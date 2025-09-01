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
from .segment_anything.modeling.common import LayerNorm2d, MLPBlock
from typing import Optional, Tuple, Type
from timm.models.vision_transformer import Mlp, DropPath
from MAE.discriminator_model import Discriminator
from MAE.loss import AdversarialLoss,PerceptualLoss,StyleLoss
import math

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    Since [cls] is useless in inpainting, we remove it.
    """

    def __init__(self,
                 # ---encoder----
                 # 右边注释的是vit-b的参数 没有注释的就是不动了
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1280,  # 768
                 depth=32,   # 12
                 num_heads=16,  # 12
                 mlp_ratio=4.,

                 out_chans=256,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,

                 use_abs_pos=True,
                 use_rel_pos=True,

                 rel_pos_zero_init=True,
                 window_size=14,
                 global_attn_indexes=[7, 15, 23, 31], # [2,5,8,11]

                 # ----decoder----
                 decoder_embed_dim=1024,  # 512
                 decoder_depth=10,        # 8
                 decoder_num_heads=16,

                 # -----保留----
                 norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False
                 ):

        super().__init__()
        self.patch_size=patch_size
        print("embded dim:",embed_dim)  # 768 已经确定
        print("decoder depth:",decoder_depth)  # 768 已经确定
        # ---------sam encoder-----------------------------------------------------------------

        # MAE encoder specifics
        # 这里可以直接拿sam的
        # 将图像划分为patch：本质上就是经过卷积+permute
        # 注意这里不要和mae的混了
        self.sam_patch_embed = SamPatchEmbed(
            kernel_size=(patch_size, patch_size),  # 卷积核大小(16,16)
            stride=(patch_size, patch_size),  # 卷积核步长(16,16)
            in_chans=in_chans,  # 输入图像通道数=3
            embed_dim=embed_dim,  # patch嵌入维度 768
        )

        # 位置嵌入
        # 表示self.sam_pos_embed可以是指定的类型nn.Parameter  或者是none
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
                use_rel_pos=use_rel_pos,  # 是否添加相对位置嵌入=false
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

        # # 输出neck模块
        # self.sam_neck = nn.Sequential(
        #     nn.Conv2d(
        #         embed_dim,  # 768
        #         out_chans,  # 256
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        #     nn.Conv2d(
        #         out_chans,  # 256
        #         out_chans,  # 256
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        # )


        # self.decoder1= nn.Conv2d(
        #         out_chans,  # 256
        #         out_chans,  # 256
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     )

        # --------decoder部分-------------------
        # 选择和mae的保持一致吧
        # 在mask token之前还要有维度变化  encoder的输出是1,64,64,1280    其实因为encoder会做mask  所以应该是1,32,32,1280
        # 1280-->1024
        # 768-->512
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)


        # 先要有mask token那些
        # 1024
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 再结合完mask token之后变为1,64,64,1024后进行
        self.grid_size=img_size//patch_size  # 256÷16=16

        # 这个后面要记得初始化 # decode_pos_embed
        # 1,16*16(256),512
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.grid_size*self.grid_size, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # 8层

        # 现在应该是1,256,512
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, mask_decoder=mask_decoder)
            for i in range(decoder_depth)])

        # 512
        self.decoder_norm = norm_layer(decoder_embed_dim)   # 1,64*64,512

        # 1024,768
        # 512-->768
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        # # 输出neck模块
        # 我想着最后经过unpatch得到的应该是b,3,1024,1024
        # 因为我们是针对256的
        # 所以这里最后是1024/4=256
        # layernorm我也不知道应该加不加  我只是模仿sam的neck
        # self.decoder_neck = nn.Sequential(
        #     nn.Conv2d(
        #         3,  # 768
        #         3,  # 256
        #         kernel_size=4,
        #         stride=4,   # 注意不要忘记这个参数
        #         bias=False,
        #     ),
        #     LayerNorm2d(3),
        #     nn.Conv2d(
        #         3,  # 256
        #         3,  # 256
        #         kernel_size=1,
        #         stride=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(3),
        # )

        # ----------这里要新增一个模块---------
        self.mfblock1=MFBlock()  # 不需要初始化参数 已经写死
        self.mfblock2=MFBlock()  # 不需要初始化参数 已经写死
        self.mfblock3=MFBlock()  # 不需要初始化参数 已经写死
        self.mfblock4=MFBlock()  # 不需要初始化参数 已经写死


        # ------其余---------------------
        self.norm_pix_loss = norm_pix_loss
        self.random_mask = random_mask
        self.mask_decoder = mask_decoder


        # ---------初始化-----------------
        if init:
            self.initialize_weights()

    def initialize_weights(self):
        print("initailize.........")
        # encoder的pos_embed 已经加载预训练模型了 所以encoder的pos_embed可以无需初始化了

        # decoder的pos_embed
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int((self.grid_size*self.grid_size) ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # 本来有对encoder的patch_embed的参数初始化 但我们有预训练模型 因此无需

        # 对于self.mask_token我们也是需要的
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm  可以的
        # 应该是先弄此后  然后又加载本身预训练模型参数 因此参数不会被覆盖  这里还要再验证一下
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

    def unpatchify(self, x):  # 后面decoder再经norm和pred的结果为bs、64*64、768  通过此函数就可以变成bs、3、1024、1024
        print("----unpatchify----")
        print("x.shape:",x.shape)  # 1,4096,768
        p = self.patch_size     # 16
        h = w = int(x.shape[1] ** .5)  # 64
        assert h * w == x.shape[1]

        # 1,64,64,16,16
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        # 1,3,64*16,64*16
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        print("imgs.shape:",imgs.shape)
        return imgs  # 1,3,1024,1024

    def adaptive_random_masking(self, x, mask, mask_ratio): # x为img做完patch embedding+pos_emb的结果 mask就是数据集读入 0.75
        # 第一维为bs 这里都假设为1
        # 我们的x应该是1,256,768(已经核对)
        # mask为1,1,256,256
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        print("----adaptive_random_masking----")
        print("x.shape:",x.shape)  # 1,256,768
        print("mask.shape:",mask.shape)

        N, L, D = x.shape  # batch, length, dim  1,256,768
        s = int(np.sqrt(L))   # 64

        # mask 要根据x来合理设置大小
        # x经过embedding等操作后的大小为64*64
        # 因此mask也要是此
        mask = F.interpolate(mask, size=[s, s], mode='area')  # 16*16大小的mask damn 感觉比较细致吧
        mask[mask > 0] = 1  # [N,1,S,S]

        mask = mask.reshape(N, L)  # [N,L]  1,16*16
        print("mask.shape:", mask.shape)  # [1,256]
        len_keep = int(L * (1 - mask_ratio))  # 256*0.25=64  计算剩余的好的非mask部分的数量
        print("len_keep:",len_keep)

        # noise 和mask 相加
        # 1,256
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # make noise with mask in largest 1
        noise = torch.clamp(noise + mask, 0.0, 1.0)
        print("noise.shape:",noise.shape)

        # 这里特别特别重要重要
        # 1,256
        # sort noise for each sample
        # 按照noise排序的索引
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 这个是按照ids_shuffle进行排序的索引  用来恢复原始序列的
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        print("ids_shuffle.shape:",ids_shuffle.shape)  # 和noise的shape相同都为[1,4096]
        print("ids_restore.shape:",ids_restore.shape)  # 和noise的shape相同都为[1,4096]

        # 1,64
        # keep the first subset
        # 取前1024个 也就是0.25比例的
        ids_keep = ids_shuffle[:, :len_keep]
        print("ids_keep.shape:",ids_keep.shape)  # 1,64
        print("ids_keep.unsqueeze(-1).repeat(1, 1, D).shape:",ids_keep.unsqueeze(-1).repeat(1, 1, D).shape)

        # 形成unmask token
        # 1,64,768
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        print("x_masked.shape:",x_masked.shape) #  1,64,768

        # generate the binary mask: 0 is keep, 1 is remove
        # 1,256
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # 形成真网络的mask  注意和数据集的不同
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        print("x_masked.shape:",x_masked.shape) # 1,1024,768
        print("mask.shape:",mask.shape)  # 1,256
        print("ids_restore.shape:",ids_restore.shape) # 1,256

        # 1,64,768(做过mask后的x,剩下的为unmask token);1,256(记录mask);1,256一种顺序，后面帮助复原的
        return x_masked, mask, ids_restore




    def forward_encoder(self, x, mask, mask_ratio):
        # 注意 输入参数都是由dataset拿到的
        # 输入参数为[1,3,256,256]、[1,1,256,256]、0.75
        print("----forward_encoder----")
        print("x.shape:", x.shape) #  [1,3,256,256]
        print("mask.shape:", mask.shape)  # 1,1,256,256

        # 将图像划分为patch
        x = self.sam_patch_embed(x)  # [1, 16, 16, 768]
        # 添加位置嵌入  可学习的位置嵌入
        if self.sam_pos_embed is not None:
            x = x + self.sam_pos_embed   # [1, 16, 16, 768]

        # 后面要进行mask  因此这里要变为1,16*16(256),768
        x_3d=x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3])

        # 输入为1,256,768;1,1,256,256;0.75
        # 输出:1,64,768(做过mask后的x 即unmask token);1,16*16(256)(记录mask);1,16*16一种顺序，后面帮助复原的
        x, mask, ids_restore = self.adaptive_random_masking(x_3d, mask, mask_ratio)

        print("x.shape:",x.shape)  # [1, 64, 768]
        print("x.shape[0]:",x.shape[0])
        print("x.shape[1]:",x.shape[1])
        print("x.shape[2]:",x.shape[2])
        print("mask.shape:",mask.shape)  # [1, 256]
        print("ids_restore.shape:",ids_restore.shape)  # [1, 256]

        # 因为sam的encoder其实是对四维的 因此我给他再次reshape一下
        # 要对x进行reshape成为1,8,8,768
        x=x.reshape(x.shape[0],int(math.sqrt(x.shape[1])),int(math.sqrt(x.shape[1])),x.shape[2])

        one_encoder_feature=[]
        # attention模块
        for i,blk in enumerate(self.sam_blocks):
            x = blk(x)  # [1,8,8,768]
            print("sam encoder x.shape:",x.shape)
            one_encoder_feature.append(x)

        print("------------------~~~~~~~~~~~~~")
        print("len(encoder_feature):",len(one_encoder_feature))  # 12

        # x = self.sam_neck(x.permute(0, 3, 1, 2))  # [1, 256, 8, 8]


        # 返回 [1,8,8,768] ;1,16*16(256)(记录mask);1,16*16一种顺序，后面帮助复原的
        return x,mask,ids_restore,one_encoder_feature   #  一气呵成giao！


    # 1.15继续写decoder部分
    # 输入：[1,8,8,768]、一种顺序[1,256]
    def forward_decoder(self, x, ids_restore,one_encoder_feature):

        print("----forward_decoder----")
        print("x.shape:", x.shape)  # encoder的结果x是[1, 8,8,768]
        print("ids_restore.shape:",ids_restore.shape)   # [1,256]



        # 我们先把x进行reshape成三维的  因为后面mask token加入时候得是三维的
        x_3d=x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3])  # 1,64,768

        # linear层变换输出通道数
        x=self.decoder_embed(x_3d)   # 1,64,512

        print("x.shape:", x.shape)


        # **加入 mask token
        # self.mask_token本身为1,1,512
        mask_tokens=self.mask_token.repeat(x.shape[0],ids_restore.shape[1]+1-x.shape[1],1)  # shape为 1,193,512
        x_=torch.cat([x,mask_tokens],dim=1)  # 1,257,512

        # ids_restore shape 为:[1,256]
        # index的shape为[1,256,1]-->[1,256,512]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))   # 最终x_的shape为[1,256,512]

        x=x_


        # add pos embed
        x = x + self.decoder_pos_embed  # [1,256,512]

        # apply Transformer blocks
        for i,blk in enumerate(self.decoder_blocks):
            # i从0开始
            x, _ = blk(x)
            if (i+1)%2==0 and i!=9:
                print("mfblock------------------------------------------------------++++++++++")
                one_encoder_feature_thing = one_encoder_feature[i]
                if i==1:
                    x = self.mfblock1(one_encoder_feature_thing, x)
                if i==3:
                    x = self.mfblock2(one_encoder_feature_thing, x)
                if i==5:
                    x = self.mfblock3(one_encoder_feature_thing, x)
                if i==7:
                    x = self.mfblock4(one_encoder_feature_thing, x)


        x = self.decoder_norm(x)  # 就是个layernorm 1,256,512

        # predictor projection
        x = self.decoder_pred(x)  # 1,256,768(16*16*3)

        return x  # 1,256,768

    def unpatchify(self, x):
        """
        x:[1, 4096, 768]
        """
        print("----unpatchify----")
        print("x.shape:",x.shape)
        p = self.patch_size   # 16
        h = w = int(x.shape[1] ** .5)  # 64
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))  # 1,64,64,16,16,3
        x = torch.einsum('nhwpqc->nchpwq', x)  # 1,3,64,16,64,16
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))  # 1,3,1024,1024
        print("imgs.shape:",imgs.shape)
        return imgs

    def mean_flat(self,tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [1, 3, 256, 256] 原始图像
        pred: [1, 3, 256,256]  预测图像和unmask部分结合结果
        mask: [1,1,256,256], 0 is keep, 1 is remove,   mask 0 1
        """
        pass
        # terms = {}
        # # -------------判别器----------------
        # self.adversarial_loss = AdversarialLoss(type="nsgan")
        # self.discriminator = Discriminator(in_channels=3, use_sigmoid='True')
        # dis_input_real = imgs
        # dis_input_fake = pred.detach()
        # dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        # dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        # dis_real_loss = self.adversarial_loss(dis_real, True, True)
        # dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        # dis_loss = (dis_real_loss + dis_fake_loss) / 2
        #
        # terms["dis_loss"] = dis_loss
        #
        # # -------------生成器-------------------
        # self.perceptual_loss = PerceptualLoss()
        # self.style_loss = StyleLoss()
        # gen_l2_loss = self.mean_flat((imgs - pred) ** 2).mean()
        # gen_l2_loss = gen_l2_loss * 20
        # gen_loss = gen_l2_loss
        #
        # terms["gen_l2_loss"] = gen_l2_loss
        #
        # # 对于mask区域加入颜色loss  l1 loss  新加的
        # gen_l1_loss = self.l1_loss(pred*mask, imgs*mask)   # 只对mask区域做loss
        # gen_l1_loss=gen_l1_loss*20
        # gen_loss = gen_loss + gen_l1_loss
        #
        # terms["gen_l1_loss"] = gen_l1_loss
        #
        # # 对抗loss:由之前的smgan得到了
        # gen_input_fake = pred
        # gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        # gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * 100
        # terms["gen_gan_loss"] = gen_gan_loss
        # gen_loss = gen_loss + gen_gan_loss
        #
        # # generator perceptual loss
        # gen_content_loss = self.perceptual_loss(pred, imgs)
        # gen_content_loss = gen_content_loss * 0.35
        # terms["gen_perceptual_loss"] = gen_content_loss
        # gen_loss = gen_loss + gen_content_loss
        #
        # # generator style loss
        # gen_style_loss = self.style_loss(pred * mask, imgs * mask)
        # gen_style_loss = gen_style_loss * 250
        # terms["gen_style_loss"] = gen_style_loss
        # gen_loss = gen_loss + gen_style_loss
        #
        #
        # return gen_loss,terms


    def forward(self, imgs,imgs_1024, mask,mask_1024, mask_ratio=0.75):
        """
        return loss, pred img, mask. Used during training.
        前四个参数为数据集拿到
        imgs:256  计算loss用   [1,3,256,256]
        imgs_1024:1024大小实际用的   [1,3,1024,1024]
        mask：256  做mask token时候用   [1,1,256,256]
        这里需要补充mask_1024  只有0和1  1代表损坏的部分  为1024大小  合并时用  [1,1,1024,1024]
        """
        print("----model forward----")
        print("imgs.shape:", imgs.shape)  # [1,3,256,256]
        print("mask.shape:", mask.shape)  # [1,1,256,256]

        mask_256=mask  # 因为后面mask会被覆盖 因此前面先备份一下

        # 输入参数为[1,3,256,256]、[1,1,256,256]、0.75
        # # 返回 [1,8,8,768] ;1,16*16(256)(记录mask);1,16*16一种顺序，后面帮助复原的
        # --------encoder  要获取中间12个特征
        latent, mask, ids_restore,one_encoder_feature= self.forward_encoder(imgs, mask, mask_ratio)


        print("latent.shape:",latent.shape)  # [1,8,8,768]
        print("mask.shape:",mask.shape)  # 1,16*16(256)
        print("ids_restore.shape:",ids_restore.shape)  # 1,16*16(256)

        # 输入：[1,8,8,768]、一种顺序[1,256]
        # 输出：1,256,768
        # 这里要传入encoder特征 来操作哈
        pred = self.forward_decoder(latent, ids_restore,one_encoder_feature)

        print("pred.shape:",pred.shape)  # [1, 256, 768]

        # pred_img为1, 3, 256, 256
        pred_img=self.unpatchify(pred)

        pred_res=pred_img

        print("pred_res.shape:",pred_res.shape)  # [1, 3, 256, 256]
        print("mask.shape:",mask.shape)  # [1, 256]
        print("mask_256.shape:",mask_256.shape)  # [1, 1, 256, 256]

        return imgs,pred_res,mask  # 原图[1,3,256,256]\预测结果[1, 3, 256, 256]\网络使用的mask[1, 256]
        # 返回损失、预测的256大小的图像、mask(实际用的，不等于数据集读入的)
        # return loss,pred_res_256,mask,terms

    # test时：1,3,256,256   1,1,256,256
    def forward_encoder_with_mask1(self, x, mask):
        print("----forward_encoder_with_mask----")

        # 将图像划分为patch
        x = self.sam_patch_embed(x)  # [1, 16, 16, 768]
        # 添加位置嵌入  可学习的位置嵌入
        if self.sam_pos_embed is not None:
            x = x + self.sam_pos_embed   # [1, 16, 16, 768]

        # 后面要进行mask  因此这里要变为1,16*16(256),768
        x=x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3])

        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))  # 16
        # masking: length -> length * mask_ratio
        mask = F.interpolate(mask, size=[s, s], mode='area')
        print("mask.shape:",mask.shape)  # 1,1,16,16

        mask_small = mask.clone()
        mask[mask > 0] = 1  # [N,1,S,S]
        # mask[mask > 0] = 1  # [N,1,S,S]
        mask_11=mask.clone()
        mask_small[mask_small < 1] = 0
        mask_small11=mask   # 这个要是可视化应该用此  而不是mask其他的
        # mask_small[mask_small < 1] = 0

        # mask = mask_small.reshape(N, L).unsqueeze(1).unsqueeze(1)  # [N,1,1,L]  N就是bs 1,1,1,256
        mask = mask.reshape(N, L).unsqueeze(1).unsqueeze(1)  # [N,1,1,L]  N就是bs 1,1,1,256
        mask1=mask
        print("mask_small.shape:",mask_small.shape)  # 1,1,16,16
        print("mask.shape:",mask.shape)  # 1,1,1,256
        mask=mask.view(N,s,s,-1)  # 1,16,16,1

        x = x.reshape(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), x.shape[2])

        # -----注意：这里是最新的改进 用来获取多个encoder block的结果
        encoder_feature=[]
        one_encoder_feature = []
        # attention模块
        # for blk in self.sam_blocks:
        for i, blk in enumerate(self.sam_blocks):
            # 输入：x为1,16,16,768  mask为1,16,16,1
            x = blk(x,mask)  # 输出：测试时应该是1,16,16,768
            one_encoder_feature.append(x)
            if (i + 1) % 2 == 0:
                encoder_feature.append(x)

        # # apply Transformer blocks
        # for blk in self.blocks:
        #     x, _ = blk(x, mask)
        # x = self.norm(x)  # N,L,D
        mask1 = mask1.squeeze(1).squeeze(1)  # N, L   1,256
        mask_small = mask_small.reshape(N, L).unsqueeze(1).unsqueeze(1) # [N,1,1,L]  1,1,1,256  N就是bs
        mask_11 = mask_11.reshape(N, L).unsqueeze(1).unsqueeze(1) # [N,1,1,L]  1,1,1,256  N就是bs
        print("x.shape:",x.shape)
        print("mask.shape:",mask.shape)
        print("mask_small.shape:",mask_small.shape)

        # 返回1,16,16,768   1,256   1,1,1,256
        return x, mask1, mask_small11,encoder_feature,one_encoder_feature

    def forward_decoder_with_mask1(self, x, mask,one_encoder_feature):
        print("----forward_decoder_with_mask1----")
        print("x.shape:", x.shape)  # encoder的结果x是[1, 16,16,768]
        print("mask.shape:",mask.shape)   # [1,256]
        x_3d = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # 1,256,768



        x = self.decoder_embed(x_3d)  # 1,256,512



        # x = self.decoder_embed(x)  # N,L,D
        N, L, D = x.shape  # batch, length, dim 1,256,512
        mask = mask.unsqueeze(-1)  # N,L,1  1,256,1
        # append mask tokens to sequence
        print("self.mask_token.shape:",self.mask_token.shape)  # 1,1,512

        # 1,256,512
        print("self.mask_token.repeat(N, L, 1).shape:",self.mask_token.repeat(N, L, 1).shape)
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x, _ = blk(x)
        deocder_feature=[]
        for i,blk in enumerate(self.decoder_blocks):
            # i从0开始
            x, _ = blk(x)
            if (i+1)%2==0 and i!=9:
                print("mfblock------------------------------------------------------++++++++++")
                one_encoder_feature_thing = one_encoder_feature[i]
                if i==1:
                    x = self.mfblock1(one_encoder_feature_thing, x)
                if i==3:
                    x = self.mfblock2(one_encoder_feature_thing, x)
                    deocder_feature.append(x)
                if i==5:
                    x = self.mfblock3(one_encoder_feature_thing, x)
                    deocder_feature.append(x)
                if i==7:
                    x = self.mfblock4(one_encoder_feature_thing, x)
                    deocder_feature.append(x)


        print("x.shape:",x.shape)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        print("x.shape:", x.shape)  # 1,256,768(16*16*3)

        return x,deocder_feature


    # imgs 为经过imgenet归一化的结果  mask为0或者1
    # shape分别为1,3,256,256  1,1,256,256
    def forward_test(self, imgs, mask):
        # 返回1,16,16,768   1,256   1,1,1,256
        latent, new_mask, new_mask_11,encoder_feature,one_encoder_feature = self.forward_encoder_with_mask1(imgs, mask)
        print("latent.shape:",latent.shape)
        print("new_mask.shape:",new_mask.shape)

        # ----这里需要确保是6个
        print("len(encoder_feature):",len(encoder_feature))
        image,decoder_feature = self.forward_decoder_with_mask1(latent, new_mask,one_encoder_feature)  # [N, L, D]
        print("image.shape:",image.shape)  # 1,256,768  # 1,256,768(16*16*3)
        pred_img = self.unpatchify(image)  # # pred_img为1, 3, 256, 256
        return pred_img,new_mask_11,encoder_feature,one_encoder_feature,decoder_feature





class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mask_decoder=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.mask_decoder = mask_decoder
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        print("----attention----")
        print("x.shape:",x.shape)
        # print("mask.shape:",mask.shape)  # 在decoder中的block没有传入mask  所以这里打印shape会报错
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        print("q,k,v.shape:",q.shape,k.shape,v.shape)
        attn_before_softmax = (q @ k.transpose(-2, -1)) * self.scale
        print("attn_before_softmax.shape:",attn_before_softmax.shape)
        if mask is not None and not self.mask_decoder:
            # mask:[B,1,L(1),L]
            attn_before_softmax = attn_before_softmax.masked_fill(mask == 1, float('-inf'))
        attn = attn_before_softmax.softmax(dim=-1)
        print("attn.shape:",attn.shape)

        if self.mask_decoder:
            if mask is None:
                score = torch.mean(attn, dim=1)
            else:
                attn_before_softmax = attn_before_softmax.masked_fill(mask == 1, float('-inf'))
                score = torch.mean(attn_before_softmax.softmax(dim=-1), dim=1)
        else:
            score = torch.mean(attn, dim=1)
        print("score.shape:",score.shape)
        attn = self.attn_drop(attn)
        print("attn.shape:",attn.shape)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        print("x.shape:",x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        print("x.shape:", x.shape)
        print("score.shape:", score.shape)
        return x, score

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mask_decoder=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, mask_decoder=mask_decoder)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x.to(torch.float32)
        print("----block----")
        print("* x.dtype:", x.dtype)
        att, score = self.attn(self.norm1(x), mask)
        x = x + self.drop_path(att)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        print("x.shape:",x.shape)
        print("score.shape:",score.shape)
        return x, score



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
        print("x.shape:", x.shape)
        x = self.proj(x)  # [1, 3, 1024, 1024]——>[1, 768, 64, 64]
        print("x.shape:", x.shape)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)  # [1, 64, 64, 768]  [bs，高度上的patches数量, 宽度上的patches数量, 嵌入维度]
        print("x.shape:", x.shape)   # [1, 64, 64, 768]
        return x


def window_partition(x: torch.Tensor, window_size: int,ceshi_mask=False) -> Tuple[torch.Tensor, Tuple[int, int]]:
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
    # ceshi_mask=True  # 注意 测试和训练时候要改的
    B, H, W, C = x.shape  # [1, 16, 16, 768]

    # 下面两个实际测试中都是12
    pad_h = (window_size - H % window_size) % window_size  # 注意 % 为取余操作  这里为6 # 需要填充的高度=6
    pad_w = (window_size - W % window_size) % window_size  # 需要填充的宽度=6
    if ceshi_mask==True and (pad_h > 0 or pad_w > 0):
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), value=1)  # 注意mask的话要填充为1

    if ceshi_mask==False and (pad_h > 0 or pad_w > 0):
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 三维数据填充(left, right, top, bottom, front, back)
        # 填充为: [1, 70, 70, 768]  应该是1,28,28,768

    Hp, Wp = H + pad_h, W + pad_w  # 70,70
    # 重塑为[1, 5, 14, 5, 14, 768]
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    print("--partition----")
    print("hp:",Hp)
    print("wp:",Wp)
    print("window_size:",window_size)
    # [25, 14, 14, 768]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 返回划分窗口后的结果 以及 填充后的高宽
    return windows, (Hp, Wp)  # [4, 14, 14, 768] (28,28)


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
    H, W = hw  # 64,64
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)  # B=1
    # [1,5,5,14,14,768]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)  # [1,70,70,768]

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()  # 去掉填充元素[1,64,64,768]
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


# 新增的多尺度感知层融合模块
class MFBlock(nn.Module):
    def __init__(self, dim=512, rates=0.01):
        super(MFBlock, self).__init__()
        self.conv11=nn.Conv2d(768, 512, kernel_size=1)
        self.a1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 128, 3, padding=0, dilation=1),
            nn.ReLU(True))
        self.a2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(512, 128, 3, padding=0, dilation=2),
            nn.ReLU(True))
        self.a3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(512, 128, 3, padding=0, dilation=4),
            nn.ReLU(True))
        self.a4 = nn.Sequential(
            nn.ReflectionPad2d(6),
            nn.Conv2d(512, 128, 3, padding=0, dilation=6),
            nn.ReLU(True))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3))

    def forward(self, x,y):
        # 输入x为encoder的feature 输入y为decoder的此层的feaure
        # 第二层的encoder feature 得加入到第二层的decoder feature才更有意义哇  这样传到第三层比较好 我认为
        # x应该是1,16,16,768   y为1,256,512
        x = x.permute(0, 3, 1, 2)  # 变为1,512,16,16
        x=self.conv11(x)    # 1*1卷积 改变通道数 1,16,16,512



        out1 = self.a1(x)
        out2 = self.a2(x)
        out3 = self.a3(x)
        out4 = self.a4(x)
        out = torch.cat((out1,out2,out3,out4), dim=1)   # 经过多尺度运算后 变为1,512,16,16


        out = self.fuse(out)   # 1*1卷积  变为1,512,16,16

        if out.shape[2]<16:
            out = F.interpolate(out, size=(16, 16), mode="bicubic")

        out = out.permute(0, 2, 3, 1)  # 1,16,16,512

        out=out.reshape(out.shape[0],out.shape[1]*out.shape[2],out.shape[3])

        out=out+y  # 1,256,512

        return out  # 1,256,512



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

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # x.shape:[1, 64, 64, 768]
    #     shortcut = x
    #     x = self.norm1(x)
    #     # Window partition  不重叠窗口划分
    #     if self.window_size > 0:
    #         H, W = x.shape[1], x.shape[2]  # 64,64
    #         # 输入x.shape:[1, 64, 64, 768]
    #         # 输出 [25, 14, 14, 768] (70,70)
    #         x, pad_hw = window_partition(x, self.window_size)
    #
    #     # 注意力机制
    #     # 输入:[25, 14, 14, 768]
    #     # 输出:[25, 14, 14, 768]
    #     x = self.attn(x)
    #     # Reverse window partition
    #     if self.window_size > 0:
    #         # 输入: [25, 14, 14, 768],14,[70,70],(64,64)
    #         # 输出:去掉填充元素[1, 64, 64, 768]
    #         x = window_unpartition(x, self.window_size, pad_hw, (H, W))
    #
    #     x = shortcut + x  # 残差
    #     x = x + self.mlp(self.norm2(x))
    #
    #     return x  # [1, 64, 64, 768]

    # 测试专写
    # **------------参数修改1
    def forward(self, x: torch.Tensor,mask=None) -> torch.Tensor:
        # x.shape:[1, 64, 64, 768]
        # 测试时x的shape为1,16,16,768
        shortcut = x
        x = self.norm1(x)

        # Window partition  不重叠窗口划分
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]  # 64,64
            # 输入x.shape:[1, 64, 64, 768]
            # 输出 [4, 14, 14, 768] (70,70)
            x, pad_hw = window_partition(x, self.window_size)
            print("x partition shape:",x.shape)
            # 测试输入x的shape为1,16,16,768
            # 测试输出x的shape为4,14,14,768  代表含义是窗口数量、窗口大小、窗口大小、维度
            if mask is not None:
                # mask也要做窗口划分  输入mask的shape为1,16,16,1
                # mask划分后为4,14,14,1 分别是窗口数量、窗口大小、窗口大小、维度
                mask,mask_pad_hw=window_partition(mask, self.window_size)
                print("mask partition shape:",mask.shape)  # 4,14,14,1
                print("---%%")
                print("mask.shape:", mask.shape)
                # 调整mask的shape  这样shape变成4,14*14
                mask = mask.reshape(mask.shape[0] * mask.shape[3], self.window_size * self.window_size)

        # 注意力机制
        # 输入:[25, 14, 14, 768]
        # 输出:[25, 14, 14, 768]
        # mask = mask.view(-1, self.window_size * self.window_size)


        # **-----------修改2
        x = self.attn(x, mask)
        # Reverse window partition
        if self.window_size > 0:
            # 输入: [25, 14, 14, 768],14,[70,70],(64,64)
            # 输出:去掉填充元素[1, 64, 64, 768]
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x  # 残差
        x = x + self.mlp(self.norm2(x))

        return x  # [1, 64, 64, 768]  测试时应该是1,16,16,768


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
        self.scale = head_dim ** -0.5  # 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # (768, 768*3)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:  # 相对位置编码
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            # print("self.rel_pos_h.shape", self.rel_pos_h.shape)
            # print("self.rel_pos_w.shape", self.rel_pos_w.shape)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # 输入x.shape:[25, 14, 14, 768]
    #     B, H, W, _ = x.shape
    #
    #     # qkv with shape (3, B, nHead, H * W, C)
    #
    #     # [25, 14, 14, 768] ->[25, 14, 14, 768*3]->[25, 14*14,3,12,64]->[3,25,12,196,64]
    #     qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    #     # q, k, v with shape (B * nHead, H * W, C)
    #     # [3,25,12,196,64]->[3,25*12,14*14,64]
    #     # 然后unbind延第一个维度解绑为3个[300,196,64]  则分别给赋予给q、k、v
    #     q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
    #
    #     # 经典注意力公式 Q*K^T/根号dk
    #     attn = (q * self.scale) @ k.transpose(-2, -1)  # attn.shape为[300,196,196]
    #
    #     # 使用相对位置编码
    #     # 这里看传参是不使用的
    #     if self.use_rel_pos:
    #         attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
    #
    #     attn = attn.softmax(dim=-1)  # [300,196,196]
    #     # [300,196,196]-> [300,196,64]->[25,12,14,14,64]->[25,14,14,12,64]->[25,14,14,768]
    #     x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    #     x = self.proj(x)  # [25,14,14,768]
    #
    #     return x


    def forward(self, x: torch.Tensor,mask=None) -> torch.Tensor:
        # 输入x.shape:[4, 14, 14, 768]
        B, H, W, _ = x.shape

        # qkv with shape (3, B, nHead, H * W, C)



        # [4, 14, 14, 768] ->[4, 14, 14, 768*3]->[4, 14*14,3,12,64]->[3,4,12,196,64]
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        # [3,4,12,196,64]->[3,4*12,14*14,64]
        # 然后unbind延第一个维度解绑为3个[48,196,64]  则分别给赋予给q、k、v
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # 经典注意力公式 Q*K^T/根号dk
        # 应该也是win_num25*num_heads12,win_size14*win_size14,win_size14*win_size14
        attn = (q * self.scale) @ k.transpose(-2, -1)  # attn.shape为[48(4*12),196,196]
        print("attn.shape:",attn.shape)

        # 应用掩码  mask本身的shape为25,14*14  表示 窗口的数量、窗口大小*窗口大小
        if mask is not None:
            print("attention mask.shape:",mask.shape)  # [4, 196]
            mask = mask.view(-1, 1, H * W)  # [num_windows, 1, ws*ws]
            mask = mask.repeat(1, self.num_heads, 1)  # [num_windows, num_heads, ws*ws]
            mask = mask.view(-1, 1, H * W)  # [num_windows25*num_heads12, 1, ws*ws]
            print("attention after mask.shape:",mask.shape)  # 感觉应该是48(4*12),1,196
            attn = attn.masked_fill(mask == 1, float('-inf'))  # 广播到所有查询位置

        # 使用相对位置编码
        # 这里看传参是不使用的
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        # if mask is not None


        attn = attn.softmax(dim=-1)  # [300,196,196]
        # [300,196,196]-> [300,196,64]->[25,12,14,14,64]->[25,14,14,12,64]->[25,14,14,768]
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)  # [25,14,14,768]

        return x

if __name__ == '__main__':
    s1=MaskedAutoencoderViT(img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,

            out_chans=256,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,

            use_abs_pos=True,  #
            use_rel_pos=True,

            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=[2, 5, 8, 11],

            # ----decoder----
            decoder_embed_dim=512,
            decoder_depth=8, decoder_num_heads=16,

            # -----保留----
            norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False)
    # print(s1.state_dict().keys())

    pretrained_checkpoint = torch.load('/mnt/data/wxx/sam/checkpoint/sam_vit_b_01ec64.pth')
    filtered_state_dict = {k[len('image_encoder.'):]: v for k, v in pretrained_checkpoint.items() if
                           k.startswith('image_encoder.')}
    # print(filtered_state_dict.keys())

    # 获取模型实例中以'sam_'开头的参数键名，并去掉'sam_'前缀
    model_state_dict = s1.state_dict()
    sam_keys = {k[len('sam_'):] if k.startswith('sam_') else None: k for k in model_state_dict.keys() if
                k.startswith('sam_')}
    # print(sam_keys.keys())  # 注意 这里的键值对为去掉sam和有sam的

    # 构建最终的参数字典，只保留匹配的参数
    final_state_dict = {}
    for k, v in filtered_state_dict.items():
        if k in sam_keys:
            # print("k:",k)
            # print("sam_keys[k]:",sam_keys[k])
            final_state_dict[sam_keys[k]] = v   # sam啥：checkpoint值

    # 更新模型参数
    missing_keys, unexpected_keys = s1.load_state_dict(final_state_dict, strict=False)

    # # 打印缺失和多余的键
    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)


    # 至此  完成只加载encoder参数的代码



    t1=torch.randn(1, 3, 256, 256)
    t2=torch.randn(1, 3, 1024, 1024)
    t3=torch.randn(1, 1, 256, 256)   # 注意这个通道数是1  不是3！！！！！
    t4=torch.randn(1, 1, 1024, 1024)
    r1,r2,r3=s1.forward_encoder(t2,t3,0.75)
    # 注意没有neck了
    print("r1.shape:",r1.shape)  # torch.Size([1,32,32,1280])
    print("r2.shape:",r2.shape)  # torch.Size([1,4096])
    print("r3.shape:",r3.shape)  # torch.Size([1,4096])

    # 不过我们先这么用着 主要看decoder维度是否变化正确
    #
    # t3=torch.randn(1,64*64,1280)

