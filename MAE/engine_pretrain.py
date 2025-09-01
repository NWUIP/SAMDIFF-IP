# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
import sys
from typing import Iterable
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import torch.nn as nn
import MAE.util.lr_sched as lr_sched
import MAE.util.misc as misc
from MAE.loss import AdversarialLoss,PerceptualLoss,StyleLoss


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    # 增加discriminator模型和优化器
                    dis_model=None,
                    dis_optimizer=None
                    ):
    # ---模型训练状态
    model.train(True)
    dis_model.train(True)

    # ---log
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter   # 1

    print("accum_iter:",accum_iter)

    # ---注意：这是优化器梯度清0
    optimizer.zero_grad()
    dis_optimizer.zero_grad()
    first = True

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 损失准备工作
    adversarial_loss = AdversarialLoss(type="nsgan")
    perceptual_loss = PerceptualLoss()
    style_loss = StyleLoss()
    l1_loss = nn.L1Loss()


    # 一个for循环是一个bs
    # data_iter_step是编号  samples就是此batchsize对应的样本
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 每一个batchsize都应该进行梯度清零、反向传播、优化器更新
        optimizer.zero_grad()
        dis_optimizer.zero_grad()
        # # we use a per iteration (instead of per epoch) lr scheduler
        # # 学习率调整 nonono
        # if data_iter_step % accum_iter == 0:
        #     print("change before:",optimizer.param_groups[0]['lr'])
        # #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # #     # 获取生成器优化器的当前学习率
        #     current_lr = optimizer.param_groups[0]['lr']
        #     dis_current_lr = dis_optimizer.param_groups[0]['lr']
        # #     # 调整判别器优化器的学习率
        # #     dis_current_lr = lr_sched.adjust_discriminator_lr(dis_optimizer, current_lr)
        # #     # dis_current_lr = lr_sched.adjust_discriminator_lr(dis_optimizer, current_lr)
        #     print(f"Generator LR: {current_lr}, Discriminator LR: {dis_current_lr}")

        # 取出samples中的值  若是tensor就送到device中  因为name这个键对应的值可能并不是tensor咯
        # samples = samples.to(device, non_blocking=True)
        for k in samples: # samples是字典 里面有img、img_1024、mask、mask_1024、name
            print(k)
            if type(samples[k]) is torch.Tensor:
                samples[k] = samples[k].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  # 自动混合精度训练
            terms={}

            # ----------------得到结果----------------------------------
            # 返回# 原图[1,3,256,256]\预测结果[1, 3, 256, 256]\网络使用的mask[1, 256]
            imgs, pred, mask = model(samples['img'],samples['img_1024'], samples['mask'],samples['mask_1024'], mask_ratio=args.mask_ratio)
            print("shuchu....")
            print("imgs.shape:",imgs.shape)  # b,3,256,256
            print("pred.shape:",pred.shape)  # b,3,256,256
            print("mask.shape:",mask.shape)  # b,4096
            print("samples['mask'].shape:",samples['mask'].shape)  # b,1,256,256

            # 注意这个传参！
            # 实际我们模型的输出：loss为标量、pred为1,3,256,256，mask为1,4096
            # terms为loss字典 用于后面的loss可视化 tensorboard
            # loss, pred, mask,terms = model(samples['img'],samples['img_1024'], samples['mask'],samples['mask_1024'], mask_ratio=args.mask_ratio)
            # 注意我们把loss都写在这里判别器加在这里

            # -----------------计算loss--------------------------------
            # -----------------判别器

            # 原图
            dis_input_real = imgs
            # 最终结果
            dis_input_fake = pred.detach()
            # 对于原图的判断结果
            dis_real, _ = dis_model(dis_input_real)  # in: [rgb(3)]
            # 对于网络输出的判断结果
            dis_fake, _ = dis_model(dis_input_fake)  # in: [rgb(3)]
            # 得到判别器损失
            dis_real_loss = adversarial_loss(dis_real, True, True)
            dis_fake_loss = adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            terms["dis_loss"] = dis_loss
            print("dis_loss:",dis_loss)
            # 判别器反向传播和优化器更新
            dis_loss /= accum_iter
            dis_loss.backward()
            dis_optimizer.step()

            # if (data_iter_step + 1) % accum_iter == 0:
            #     dis_optimizer.zero_grad()


            # -------------生成器-------------------

            gen_l2_loss = mean_flat((imgs - pred) ** 2).mean()
            gen_l2_loss = gen_l2_loss * 20
            gen_loss = gen_l2_loss

            terms["gen_l2_loss"] = gen_l2_loss
            print("gen_l2_loss:", gen_l2_loss)

            # 对于mask区域加入颜色loss  l1 loss  新加的
            # 注意mask
            gen_l1_loss = l1_loss(pred * samples['mask'], imgs * samples['mask'])  # 只对mask区域做loss
            gen_l1_loss = gen_l1_loss * 20
            gen_loss = gen_loss + gen_l1_loss

            terms["gen_l1_loss"] = gen_l1_loss
            print("gen_l1_loss:", gen_l1_loss)

            # 对抗loss:由之前的smgan得到了
            # 模型输出
            gen_input_fake = pred
            gen_fake, _ = dis_model(gen_input_fake)  # in: [rgb(3)]
            gen_gan_loss = adversarial_loss(gen_fake, True, False) * 100
            terms["gen_gan_loss"] = gen_gan_loss
            gen_loss = gen_loss + gen_gan_loss
            print("gen_gan_loss:", gen_gan_loss)

            # generator perceptual loss
            gen_content_loss = perceptual_loss(pred, imgs)
            gen_content_loss = gen_content_loss * 0.35
            terms["gen_perceptual_loss"] = gen_content_loss
            gen_loss = gen_loss + gen_content_loss
            print("gen_content_loss:", gen_content_loss)

            # # generator style loss
            # gen_style_loss = style_loss(pred * samples['mask'], imgs * samples['mask'])
            # gen_style_loss = style_loss(pred, imgs)
            # print("gen_style_loss:", gen_style_loss)
            # gen_style_loss = gen_style_loss * 250
            # terms["gen_style_loss"] = gen_style_loss
            # gen_loss = gen_loss + gen_style_loss
            print("gen_loss:", gen_loss)


        # 注意这个loss就是生成器的loss
        # 判别器的loss以及更新准备在ceshi_model.py中实现
        loss_value = gen_loss.item()
        dis_loss=terms["dis_loss"].item()
        gen_l2_loss=terms["gen_l2_loss"].item()
        gen_l1_loss=terms["gen_l1_loss"].item()
        gen_gan_loss=terms["gen_gan_loss"].item()
        gen_perceptual_loss=terms["gen_perceptual_loss"].item()
        # gen_style_loss=terms["gen_style_loss"].item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        gen_loss /= accum_iter

        # loss反向传播
        # 优化器更新
        # 之前已经设置过require_grad为false了  放心在反向传播和优化器更新都不涉及
        loss_scaler(gen_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)




        print("(data_iter_step + 1) % accum_iter:",(data_iter_step + 1) % accum_iter)
        # # 再次梯度清零
        # if (data_iter_step + 1) % accum_iter == 0:
        #     optimizer.zero_grad()
        #     dis_optimizer.zero_grad()

        torch.cuda.synchronize(device=1)

        metric_logger.update(loss=loss_value)

        # 学习率更新
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # tensorboard画图
        # **这里可以加新的图
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        dis_loss_reduce = misc.all_reduce_mean(dis_loss)
        gen_l2_loss_reduce = misc.all_reduce_mean(gen_l2_loss)
        gen_l1_loss_reduce = misc.all_reduce_mean(gen_l1_loss)
        gen_gan_loss_reduce = misc.all_reduce_mean(gen_gan_loss)
        gen_perceptual_loss_reduce = misc.all_reduce_mean(gen_perceptual_loss)
        # gen_style_loss_reduce = misc.all_reduce_mean(gen_style_loss)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('gen_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('dis_loss', dis_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_l2_loss', gen_l2_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_l1_loss', gen_l1_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_gan_loss', gen_gan_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_perceptual_loss', gen_perceptual_loss_reduce, epoch_1000x)
            # log_writer.add_scalar('gen_style_loss', gen_style_loss_reduce, epoch_1000x)

            log_writer.add_scalar('lr', lr, epoch_1000x)

        if data_iter_step % 100 == 0:
            print("----first and is_process----")
            first = False  # 意味着此if一个epoch进入一次
            print("hasattr(model, 'module'):",hasattr(model, 'module'))   # false
            model_without_ddp = model.module if hasattr(model, 'module') else model
            os.makedirs(args.output_dir + '/samples', exist_ok=True)


            # ----可视化结果-----
            # --预测--------
            y = pred[:4]    # 1,3,256,256
            y = torch.einsum('nchw->nhwc', y).detach()   # 1,256,256,3


            # --使用的mask(注意和数据集使用的不同，具体请看encoder)----------
            # visualize the mask
            mask = mask[:4].detach()  # 1,256

            # 1,256,1--> 1,256,768
            mask = mask.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_size ** 2 * 3)  # (N, H*W, p*p*3)

            # 1,256,768-->1,3,256,256
            mask = model_without_ddp.unpatchify(mask)  # 1 is removing, 0 is keeping
            print("mask.shape:",mask.shape)

            # 1,3,256,256
            resized_mask = mask
            # resized_mask = F.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)
            print("resized_mask.shape:", resized_mask.shape)
            # unique_values = torch.unique(resized_mask)
            #
            # # 打印唯一值
            # print("唯一值:", unique_values)

            # 这个为网络专用的mask1,256,256,3
            mask = torch.einsum('nchw->nhwc', resized_mask).detach()  # .cpu()


            #---原图------------
            # 1,256,256,3
            x = torch.einsum('nchw->nhwc', samples['img'][:4])
            # ---数据集mask--------------
            mask_data=torch.einsum('nchw->nhwc', samples["mask"][:4])
            # 数据集masked image
            im_masked = x * (1 - mask_data)
            im_masked = im_masked.cpu()
            print("im_masked.shape:",im_masked.shape)
            im_masked = torch.cat(tuple(im_masked), dim=0)
            print("im_masked.shape:", im_masked.shape)

            # 数据集mask和完整结合MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask_data) + y * mask_data
            im_paste = im_paste.cpu()
            im_paste = torch.cat(tuple(im_paste), dim=0)
            print("im_paste.shape:",im_paste.shape)


            # -----下面为实际的mask
            # masked image  实际网络mask x
            im_masked1 = x * (1 - mask)
            im_masked1 = im_masked1.cpu()
            print("im_masked1.shape:",im_masked1.shape)
            im_masked1 = torch.cat(tuple(im_masked1), dim=0)
            print("im_masked1.shape:", im_masked1.shape)

            print("mask.shape:",mask.shape)

            # MAE reconstruction pasted with visible patches
            # 实际网络mask x 和网络预测
            im_paste1 = x * (1 - mask) + y * mask
            im_paste1 = im_paste1.cpu()
            im_paste1 = torch.cat(tuple(im_paste1), dim=0)
            print("im_paste1.shape:",im_paste1.shape)

            x = x.cpu()
            y = y.cpu()
            x = torch.cat(tuple(x), dim=0)
            y = torch.cat(tuple(y), dim=0)
            print("x.shape:",x.shape)
            print("y.shape:",y.shape)

            # 原图、数据集mask的缺损图、网络输出图、网络输出图和数据集mask图进行结合、实际网络输入的mask图、实际网络mask和y图融合图
            images = torch.cat([x.float(), im_masked.float(), y.float(), im_paste.float(),im_masked1.float(),im_paste1.float()], dim=1)
            images = torch.clip((images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).int()
            images = images.numpy().astype(np.uint8)

            path = os.path.join(args.output_dir, 'samples')
            name = os.path.join(path, str(epoch).zfill(5)+'_'+str(data_iter_step).zfill(10) + ".jpg")
            print('\nsaving sample ' + name)

            print("images.shape:",images.shape)
            cv2.imwrite(name, images[:, :, ::-1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_finetune(model: torch.nn.Module,
                             data_loader: Iterable, optimizer, optimizer_new,
                             device: torch.device, epoch: int, loss_scaler,
                             log_writer=None, args=None, start_epoch=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('alpha', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_new.zero_grad()
    first = True

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_finetune(optimizer, data_iter_step / len(data_loader) + epoch, args, start_from=start_epoch)
            lr_sched.adjust_learning_rate_finetune(optimizer_new, data_iter_step / len(data_loader) + epoch, args, start_from=start_epoch)

        # samples = samples.to(device, non_blocking=True)
        for k in samples:
            if type(samples[k]) is torch.Tensor:
                samples[k] = samples[k].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask, irr_mask, partial_mask = model(samples['img'], samples['mask'], mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, optimizer_new, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            optimizer_new.zero_grad()

        torch.cuda.synchronize(device=1)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(alpha=model.module.alpha.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('alpha', model.module.alpha.item(), epoch_1000x)

        if first and misc.is_main_process():
            first = False
            os.makedirs(args.output_dir + '/samples', exist_ok=True)
            model_without_ddp = model.module if hasattr(model, 'module') else model
            y = model_without_ddp.unpatchify(pred[:4])
            y = torch.einsum('nchw->nhwc', y).detach()

            # visualize the mask
            mask = mask[:4].detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            mask = model_without_ddp.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach()  # .cpu()

            partial_mask = partial_mask[:4].detach()
            partial_mask = partial_mask.repeat(1, 1, model_without_ddp.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            partial_mask = model_without_ddp.unpatchify(partial_mask)  # 1 is removing, 0 is keeping
            partial_mask = torch.einsum('nchw->nhwc', partial_mask).detach()  # .cpu()

            irr_mask = irr_mask[:8].detach()
            irr_mask = torch.einsum('nchw->nhwc', irr_mask).detach()  # .cpu()

            x = torch.einsum('nchw->nhwc', samples['img'][:8])

            # masked image
            im_masked = x * (1 - mask)
            im_masked = im_masked.cpu()
            im_masked = torch.cat(tuple(im_masked), dim=0)

            im_masked2 = x * (1 - irr_mask)
            im_masked2 = im_masked2.cpu()
            im_masked2 = torch.cat(tuple(im_masked2), dim=0)

            im_masked3 = x * (1 - partial_mask)
            im_masked3 = im_masked3.cpu()
            im_masked3 = torch.cat(tuple(im_masked3), dim=0)

            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask
            im_paste = im_paste.cpu()
            im_paste = torch.cat(tuple(im_paste), dim=0)
            x = x.cpu()
            y = y.cpu()
            x = torch.cat(tuple(x), dim=0)
            y = torch.cat(tuple(y), dim=0)

            images = torch.cat([x.float(), im_masked.float(), im_masked2.float(), im_masked3.float(), y.float(), im_paste.float()], dim=1)
            images = torch.clip((images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).int()
            images = images.numpy().astype(np.uint8)

            path = os.path.join(args.output_dir, 'samples')
            name = os.path.join(path, str(epoch).zfill(10) + ".jpg")
            print('\nsaving sample ' + name)
            cv2.imwrite(name, images[:, :, ::-1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
