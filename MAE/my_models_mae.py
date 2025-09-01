# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from MAE.util.pos_embed import get_2d_sincos_pos_embed
from MAE.vision_transformer import Block
from MAE.segment_anything.modeling.common import LayerNorm2d, MLPBlock
from typing import Optional, Tuple, Type


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    Since [cls] is useless in inpainting, we remove it.
    """

    def __init__(self,
                 # ---encoder----
                 img_size=1024,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1280,
                 depth=32,
                 num_heads=16,
                 mlp_ratio=4.,

                 out_chans=256,
                 qkv_bias= True,
                 norm_layer=nn.LayerNorm,
                 act_layer= nn.GELU,

                 use_abs_pos = True,  #
                 use_rel_pos = True,

                 rel_pos_zero_init = True,
                 window_size = 14,
                 global_attn_indexes=[7,15,23,31],

                 # ----decoder----
                 decoder_embed_dim=1024,
                 decoder_depth=26, decoder_num_heads=16,

                 # -----保留----
                 norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False
                 ):

        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # 这里可以直接拿sam的
        # 将图像划分为patch：本质上就是经过卷积+permute
        # 注意这里不要和mae的混了
        self.sam_patch_embed = SamPatchEmbed(
            kernel_size=(patch_size, patch_size), # 卷积核大小(16,16)
            stride=(patch_size, patch_size), # 卷积核步长(16,16)
            in_chans=in_chans, # 输入图像通道数=3
            embed_dim=embed_dim, # patch嵌入维度 768
        )
        # 位置嵌入
        self.sam_pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:  # 绝对位置  应该有的
            # Initialize absolute positional embedding with pretrain image size.
            self.sam_pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )  # 可学习参数[1, 64, 64, 768]  patch_size=16  img_size=1024

        # sam Block模块
        self.sam_blocks = nn.ModuleList()
        for i in range(depth):  # depth为12
            sam_block = Sam_Block(
                dim=embed_dim,  # 嵌入维度 768
                num_heads=num_heads,  # 多头的数目=12
                mlp_ratio=mlp_ratio,  # mlp隐藏层的维度变换因子=4
                qkv_bias=qkv_bias,  # qkv全连接层的偏置=True
                norm_layer=norm_layer,  # 归一化层：nn.layernorm
                act_layer=act_layer,  # 激活函数层：nn.gelu
                use_rel_pos=use_rel_pos,  #是否添加相对位置嵌入=false
                rel_pos_zero_init=rel_pos_zero_init,  # 零初始化相对位置参数=true
                # 这里有根据global_attn_indexes做  但我看不同规模的模型 这个参数不一样
                # 虽然我们这里使用的h模型 但跟随博主 我们先看b模型
                # b模型这里的global_attn_indexes为[2, 5, 8, 11]，window_size都为14
                # 因为i从depth而来，depth=12
                # 则下面的window_size 12个block中分别为 [14,14,0,14,14,0,14,14,0,14,14,0]
                window_size=window_size if i not in global_attn_indexes else 0,
                # 输入大小(16,16)
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.sam_blocks.append(sam_block)

        # 输出neck模块
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,  # 768
                out_chans,  # 256
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,  # 256
                out_chans,  # 256
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )




        # ---------------原来的--------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, mask_decoder=mask_decoder)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.random_mask = random_mask
        self.mask_decoder = mask_decoder

        if init:
            self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        print("pos_embed.shape:",pos_embed.shape)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):  # 本身mae的代码patchipy我看也没有在网络中使用  而是在后面做loss时  利用这个函数将image分成patch  然后逐个patch做loss
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        print("----unpatchify----")
        print("x.shape:",x.shape)
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        print("imgs.shape:",imgs.shape)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def adaptive_random_masking(self, x, mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        mask: [N, 1, 256, 256]
        """
        print("----adaptive_random_masking----")
        print("x.shape:",x.shape)
        print("mask.shape:",mask.shape)
        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))
        mask = F.interpolate(mask, size=[s, s], mode='area')
        mask[mask > 0] = 1  # [N,1,S,S]
        mask = mask.reshape(N, L)  # [N,L]
        print("mask.shape:", mask.shape)
        len_keep = int(L * (1 - mask_ratio))
        print("len_keep:",len_keep)
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # make noise with mask in largest 1
        noise = torch.clamp(noise + mask, 0.0, 1.0)
        print("noise.shape:",noise.shape)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        print("ids_shuffle.shape:",ids_shuffle.shape)
        print("ids_restore.shape:",ids_restore.shape)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        print("ids_keep.shape:",ids_keep.shape)
        print("ids_keep.unsqueeze(-1).repeat(1, 1, D).shape:",ids_keep.unsqueeze(-1).repeat(1, 1, D).shape)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        print("x_masked.shape:",x_masked.shape)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        print("x_masked.shape:",x_masked.shape)
        print("mask.shape:",mask.shape)
        print("ids_restore.shape:",ids_restore.shape)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask, mask_ratio):
        print("----forward_encoder----")
        print("x.shape:",x.shape)
        print("mask.shape:",mask.shape)
        # embed patches
        x = self.patch_embed(x)
        print("patch_embed x.shape:",x.shape)
        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        if self.random_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.adaptive_random_masking(x, mask, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x)
        print("x.shape:",x.shape)
        x = self.norm(x)
        print("x.shape:", x.shape)
        print("mask.shape:",mask.shape)
        print("ids_restore.shape:",ids_restore.shape)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        '''
        forward decoder during training needs ids_restore
        '''
        # embed tokens
        print("----forward_decoder----")
        print("x.shape:",x.shape)
        x = self.decoder_embed(x)
        print("x.shape:",x.shape)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        print("mask_tokens.shape:",mask_tokens.shape)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        print("x_.shape:",x_.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        print("x_.shape:", x_.shape)
        x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        print("x.shape:",x.shape)
        return x

    # test时：1,3,256,256   1,1,256,256
    def forward_encoder_with_mask(self, x, mask):
        print("----forward_encoder_with_mask----")
        # embed patches
        x = self.patch_embed(x)
        print("x.shape:",x.shape)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask = F.interpolate(mask, size=[s, s], mode='area')
        print("mask.shape:",mask.shape)
        mask_small = mask.clone()
        mask[mask > 0] = 1  # [N,1,S,S]
        mask_small[mask_small < 1] = 0
        mask = mask.reshape(N, L).unsqueeze(1).unsqueeze(1)  # [N,1,1,L]
        print("mask_small.shape:",mask_small.shape)
        print("mask.shape:",mask.shape)
        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x, mask)
        x = self.norm(x)  # N,L,D
        mask = mask.squeeze(1).squeeze(1)  # N, L
        mask_small = mask_small.reshape(N, L).unsqueeze(1).unsqueeze(1) # [N,1,1,L]
        print("x.shape:",x.shape)
        print("mask.shape:",mask.shape)
        print("mask_small.shape:",mask_small.shape)
        return x, mask, mask_small

    def forward_decoder_with_mask(self, x, mask):
        print("----forward_decoder_with_mask----")
        x = self.decoder_embed(x)  # N,L,D
        N, L, D = x.shape  # batch, length, dim
        mask = mask.unsqueeze(-1)  # N,L,1
        # append mask tokens to sequence
        print("self.mask_token.shape:",self.mask_token.shape)
        print("self.mask_token.repeat(N, L, 1).shape:",self.mask_token.repeat(N, L, 1).shape)
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, _ = blk(x)

        print("x.shape:",x.shape)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        print("x.shape:", x.shape)
        return x

    def forward_decoder_return_feature(self, x, mask, mask_small):
        # embed tokens
        x = self.decoder_embed(x)  # N,L,D
        N, L, D = x.shape  # batch, length, dim
        mask = mask.unsqueeze(-1)  # N,L,1
        # append mask tokens to sequence
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask

        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        scores = []
        for blk in self.decoder_blocks:
            if self.mask_decoder:
                x, score = blk(x, mask_small)
            else:
                x, score = blk(x)
            scores.append(score.unsqueeze(1))
        scores = torch.mean(torch.cat(scores, dim=1), dim=1)  # [B,256,256]
        x = self.decoder_norm(x)
        return x, scores

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        print("loss.shape:",loss.shape)
        return loss

    def forward(self, imgs, mask, mask_ratio=0.75):
        """
        return loss, pred img, mask. Used during training.
        """
        print("----model forward----")
        print("imgs.shape:",imgs.shape)
        print("mask.shape:",mask.shape)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward_return_feature(self, imgs, mask):
        """
        return pred(feature), scores(attention). Used during finetuning.
        """
        latent, new_mask, mask_small = self.forward_encoder_with_mask(imgs, mask)
        pred, scores = self.forward_decoder_return_feature(latent, new_mask, mask_small)  # [N, L, D]
        N, L, D = pred.shape  # batch, length, dim
        s = int(np.sqrt(L))
        pred = pred.reshape(N, s, s, D).permute(0, 3, 1, 2)
        return pred, scores

    def forward_return_image(self, imgs, mask):
        """
        test时：1,3,256,256   1,1,256,256
        return Image, new_mask. Used during testing.
        """
        print("----forward_return_image----")
        print("imgs.shape:",imgs.shape)
        print("mask.shape:",mask.shape)
        latent, new_mask, _ = self.forward_encoder_with_mask(imgs, mask)
        print("latent.shape:",latent.shape)
        print("new_mask.shape:",new_mask.shape)
        image = self.forward_decoder_with_mask(latent, new_mask)  # [N, L, D]
        print("image.shape:",image.shape)
        image = self.unpatchify(image)
        return image, new_mask



class SamPatchEmbed(nn.Module):
    """
    Image to Patch Embedding. 本质上就是经过卷积+permute
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("----patch embedding----")
        print("x.shape:",x.shape)
        x = self.proj(x)   # [1, 3, 1024, 1024]——>[1, 768, 64, 64]
        print("x.shape:",x.shape)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)  # [1, 64, 64, 768]  [bs，高度上的patches数量, 宽度上的patches数量, 嵌入维度]
        print("x.shape:", x.shape)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    不重叠窗口划分
    Args:
        x (tensor): input tokens with [B, H, W, C].  # x.shape:[1, 64, 64, 768]
        window_size (int): window size. 比如 14

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape  # [1, 64, 64, 768]

    pad_h = (window_size - H % window_size) % window_size # 注意 % 为取余操作  这里为6 # 需要填充的高度=6
    pad_w = (window_size - W % window_size) % window_size # 需要填充的宽度=6
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 三维数据填充(left, right, top, bottom, front, back)
        # 填充为: [1, 70, 70, 768]
    Hp, Wp = H + pad_h, W + pad_w  # 70,70
    # 重塑为[1, 5, 14, 5, 14, 768]
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # [25, 14, 14, 768]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 返回划分窗口后的结果 以及 填充后的高宽
    return windows, (Hp, Wp)  # [25, 14, 14, 768] (70,70)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    恢复原始中间特征尺寸
    Window unpartition into original sequences and removing padding.
    # 输入 [25, 14, 14, 768],14,[70,70],(64,64)
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw  # 70,70
    H, W = hw  #64,64
    B = windows.shape[0] // (Hp * Wp // window_size // window_size) # B=1
    # [1,5,5,14,14,768]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1) # [1,70,70,768]

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous() # 去掉填充元素[1,64,64,768]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    # size[14,14]:相对位置编码,右上角为0,左下角为26,沿x=y对称
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]  # [14,14,64]

# 相对位置编码
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # 获取相对位置编码
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class Sam_Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,  # 输入维度 768
        num_heads: int,  # 多少头 12
        mlp_ratio: float = 4.0,  # mlp是不是用变换因子
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,  # window大小随着不同i而不同
        input_size: Optional[Tuple[int, int]] = None,  # 输入大小(16,16)  应该是计算相对位置编码的
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.  **这里  是不是用窗口注意力块
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        # 归一化层nn.LayerNorm
        self.norm1 = norm_layer(dim)
        # attention模块
        self.attn = SAMAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            # 14,14
            # 若有window size 则进行窗口注意力 否则全局
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        # 归一化层nn.LayerNorm
        self.norm2 = norm_layer(dim)
        # MLP模块, mlp_ratio=4, act_layer=nn.GELU
        # 两层全连接层
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size  # 14或者0  随着不同i值不同

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape:[1, 64, 64, 768]
        shortcut = x
        x = self.norm1(x)
        # Window partition  不重叠窗口划分
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]  # 64,64
            # 输入x.shape:[1, 64, 64, 768]
            # 输出 [25, 14, 14, 768] (70,70)
            x, pad_hw = window_partition(x, self.window_size)

        # 注意力机制
        # 输入:[25, 14, 14, 768]
        # 输出:[25, 14, 14, 768]
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            # 输入: [25, 14, 14, 768],14,[70,70],(64,64)
            # 输出:去掉填充元素[1, 64, 64, 768]
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x  # 残差
        x = x + self.mlp(self.norm2(x))

        return x  # [1, 64, 64, 768]


class SAMAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,  # 嵌入维度 768
        num_heads: int = 8,  # 头数
        qkv_bias: bool = True,  # qkv全连接层偏置
        use_rel_pos: bool = False,  # false
        rel_pos_zero_init: bool = True,  # 0初始化相对位置编码
        input_size: Optional[Tuple[int, int]] = None,  # 比如14,14或者16,16
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads  # 12
        head_dim = dim // num_heads  # 768/12=64
        self.scale = head_dim**-0.5  # 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # (768, 768*3)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:  # 相对位置编码
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            print("self.rel_pos_h.shape",self.rel_pos_h.shape)
            print("self.rel_pos_w.shape",self.rel_pos_w.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x.shape:[25, 14, 14, 768]
        B, H, W, _ = x.shape

        # qkv with shape (3, B, nHead, H * W, C)

        # [25, 14, 14, 768] ->[25, 14, 14, 768*3]->[25, 14*14,3,12,64]->[3,25,12,196,64]
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        # [3,25,12,196,64]->[3,25*12,14*14,64]
        # 然后unbind延第一个维度解绑为3个[300,196,64]  则分别给赋予给q、k、v
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # 经典注意力公式 Q*K^T/根号dk
        attn = (q * self.scale) @ k.transpose(-2, -1)  # attn.shape为[300,196,196]

        # 使用相对位置编码
        # 这里看传参是不使用的
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)  # [300,196,196]
        # [300,196,196]-> [300,196,64]->[25,12,14,14,64]->[25,14,14,12,64]->[25,14,14,768]
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)  # [25,14,14,768]

        return x
