# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from MAE.segment_anything.modeling.image_encoder import ImageEncoderViT
import torch
import torch.distributed as dist
import torch.nn as nn
from math import inf
from torch.nn import functional as F

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        # loss反向传播
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)

            # 更新优化器
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class NativeScalerWithGradNormCount_finetune:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, optimizer_new, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer_new)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                self._scaler.unscale_(optimizer_new)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.step(optimizer_new)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, last=False):
    output_dir = Path(args.output_dir)
    if last:
        epoch_name = 'last'
    else:
        epoch_name = str(epoch)
    if loss_scaler is not None:
        print("save checkpoint...loss_scaler....")
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            # 无论参数是否冻结都会被保存
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),  # 优化器参数也有哇
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

def sam_encoder_checkponit(checkpoint_path):
    # 模仿sam2d做的  即使256也可以加载模型
    image_size=256
    vit_patch_size=16
    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2,5,8,11],
        window_size=14,
        out_chans=256,
    )
    # 加载预训练参数
    with open(checkpoint_path, "rb") as f:
        state_dicts = torch.load(f, map_location="cpu")

    sam_dict = image_encoder.state_dict()  # 获得实例模型的一切

    # 创建一个新的字典用于存储带有前缀的键值对
    new_dict = {}

    # 遍历原始字典的键和值
    for key, value in sam_dict.items():
        # 为键名添加前缀
        new_key = "image_encoder." + key
        new_dict[new_key] = value

    # 现在 new_dict 包含了带有前缀的键名 就是原始encoder加上了encoder前缀
    sam_dict = new_dict

    print("sam_dict:", sam_dict.keys())  # 就是原始encoder加上了encoder前缀
    print("state_dicts:",state_dicts.keys())  # sam所有的一切
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']

    # 从sam一切选出和encoder一样的组成
    new_state_dict = {k: v for k, v in state_dicts.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[
                          2] not in k}

    # print("sam_dict:",sam_dict)
    print("new_state_dict:",new_state_dict.keys())

    pos_embed = new_state_dict['image_encoder.pos_embed']

    token_size = int(image_size // vit_patch_size)

    # 对于一些情况修改组后结果new_state_dict
    if pos_embed.shape[1] != token_size:
        print("pos_embed.shape[1] != token_size...")
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed

        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]

        global_rel_pos_keys = [k for k in rel_pos_keys if
                               '2' in k or
                               '5' in k or
                               '7' in k or
                               '8' in k or
                               '11' in k or
                               '13' in k or
                               '15' in k or
                               '23' in k or
                               '31' in k]
        # print(sam_dict)
        for k in global_rel_pos_keys:
            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(rel_pos_params, (h_check, w_check), mode='bilinear', align_corners=False)

            new_state_dict[k] = rel_pos_params[0, 0, ...]

    # 更新原始encoder的键值对
    sam_dict.update(new_state_dict)
    print("end....")
    print("sam_dict:", sam_dict.keys())  # 只有encoder的且带image_encoder


    return sam_dict     # 原本sam encoder的所有东西 里面有neck的

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    type='vit'
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 这里
            checkpoint = sam_encoder_checkponit(args.resume)
        # **或许这里可以写个if语句对于是啥样的resume进行判断
        # 若是vit的话就是加载encoder那部分

        if type=="vit":

            # checkpoint的参数
            filtered_state_dict = {k[len('image_encoder.'):]: v for k, v in checkpoint.items() if
                                   k.startswith('image_encoder.')}
            # print(filtered_state_dict.keys())

            # 获取模型实例中以'sam_'开头的参数键名，并去掉'sam_'前缀
            model_state_dict = model_without_ddp.state_dict()  # 实例化中的
            # print("model_state_dict.keys():",model_state_dict)
            sam_keys = {k[len('sam_'):] if k.startswith('sam_') else None: k for k in model_state_dict.keys() if
                        k.startswith('sam_')}
            # print(sam_keys.keys())  # 注意 这里的键值对为去掉sam和有sam的

            # 构建最终的参数字典，只保留匹配的参数
            final_state_dict = {}
            for k, v in filtered_state_dict.items():
                if k in sam_keys:
                    # print("k:", k)
                    # print("sam_keys[k]:", sam_keys[k])
                    final_state_dict[sam_keys[k]] = v  # sam啥：checkpoint值

            print("final_state_dict:",final_state_dict.keys())  # 最开始无neck

            # 更新模型参数
            # 无neck  有neck
            # 当有neck时 则有neck
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(final_state_dict, strict=False)

            # 打印缺失和多余的键
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)


        # 若是vae的话就是下面的：
        if type=="mae":
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            print("Resume checkpoint %s" % args.resume)

            # 这个也是vae的话延伸出来的
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    print("world_size:",world_size)
    if world_size > 1:
        print("fenbushi pingjun...")
        x_reduce = torch.tensor(x).to("cuda")
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

# 传参为mae_vit_base_patch16、false、false
def get_mae_model(type, finetune=False, random_mask=False, **kwargs):
    if finetune:
        from MAE.models_mae import MaskedAutoencoderViTFinetune as mae
    else:
        from MAE.models_mae import MaskedAutoencoderViT as mae

    # 这里
    if type == 'mae_vit_base_patch16':
        model = mae(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), random_mask=random_mask,
                    **kwargs)
    elif type == 'mae_vit_large_patch16':
        model = mae(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), random_mask=random_mask,
                    **kwargs)
    elif type == 'mae_vit_huge_patch14':
        model = mae(patch_size=14, embed_dim=1280, depth=32, num_heads=16,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), random_mask=random_mask,
                    **kwargs)
    else:
        raise NotImplementedError

    return model


def add_weight_decay_finetune(model, weight_decay=1e-5, skip_list=(), opt_tag=1):
    decay = []
    no_decay = []
    new_decay = []
    new_no_decay = []
    for name, param in model.named_parameters():
        if opt_tag == 1 and 'decoder_patch_embed' in name or 'alpha' in name:
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                new_no_decay.append(param)
            else:
                new_decay.append(param)
        else:
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}], \
           [{'params': new_no_decay, 'weight_decay': 0.}, {'params': new_decay, 'weight_decay': weight_decay}]

# 一组是不需要权重衰减的参数（如偏置项、一维参数和在 skip_list 中的参数），另一组是需要权重衰减的参数。这样可以更灵活地控制不同类型的参数在训练过程中的更新策略。
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("sam frozen...")
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# 更新
def get_sam_image_encoder_model(type='sam_vit_b', finetune=False, random_mask=False, **kwargs):
    from MAE.ceshi_model import MaskedAutoencoderViT as vit
    # 这里
    if type == 'sam_vit_b':
        model = vit(
            img_size=256,  # 注意这里发生改变
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,

            out_chans=256,
            qkv_bias=True,
            # 这里的 partial 用于固定 nn.LayerNorm 的 eps 参数为 1e-6，并生成一个新的函数 norm_layer。
            # 当你调用 norm_layer 时，它会自动将 eps=1e-6 传递给 nn.LayerNorm。
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
            norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False
        )
    elif type == 'sam_vit_h':
        model=vit(
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4.,

            out_chans=256,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,

            use_abs_pos=True,  #
            use_rel_pos=True,

            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=[7, 15, 23, 31],

            # ----decoder----
            decoder_embed_dim=1024,
            decoder_depth=20, decoder_num_heads=16,

            # -----保留----
            norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False
        )
    else:
        raise NotImplementedError

    return model
