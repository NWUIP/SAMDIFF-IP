import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from PIL import Image
import argparse
from guided_diffusion.metrics_my1 import EdgeAccuracy
from torchvision.utils import save_image
import numpy as np
import torch as th
import functools
import torch.distributed as dist
from guided_diffusion.my_util import stitch_images, create_dir, stitch_images1
from guided_diffusion.dataset_my import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from MAE.util import misc
from guided_diffusion.gaussian_diffusion import GaussianDiffusion

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', random_mask=False, finetune=False):
    model = misc.get_sam_image_encoder_model(type='sam_vit_b')
    checkpoint = th.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print("sam.checkpoint:", checkpoint.keys())
    print("msg:", msg)
    return model

def main(model_path, save_dir):

    th.cuda.device_count()

    args = create_argparser().parse_args()
    args1 = create_argparser().parse_args()
    args1.in_channels = 6
    args1.out_channels = 3
    args1.model_path = model_path

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    print("here....")
    print(args_to_dict(args, model_and_diffusion_defaults().keys()))
    print(args_to_dict(args1, model_and_diffusion_defaults().keys()))

    print(args1.model_path)


    model1, diffusion1 = create_model_and_diffusion(
        **args_to_dict(args1, model_and_diffusion_defaults().keys())
    )
    state_dict1 = th.load(args1.model_path, map_location=th.device('cpu'))
    model1.load_state_dict(state_dict1, strict=True)
    print("len(model1.load_state_dict(state_dict1)):", len(model1.load_state_dict(state_dict1)))
    if len(model1.load_state_dict(state_dict1)) == 0:
        print("All parameters have been loaded successfully.")
    else:
        print("There are some parameters that were not loaded.")

    model1.to(dist_util.dev())
    sam_chkpt_dir = ''
    sam_model = prepare_model(sam_chkpt_dir, random_mask=False, finetune=False).to("cpu")

    if args.use_fp16:
        model1.convert_to_fp16()
    model1.eval()

    test_data = load_data(
        data_dir=args.test_datadir,
        mask_dir=args.test_maskdir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
        deterministic=True,
        mask_train=False,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_images = []

    while len(all_images) * args.batch_size < args.num_samples:
        image, mask, mask_image, edge, mask_edge, gray_maskimage, gray_image, sam_yuantu, sam_mask, _ = next(test_data)
        image = image.to(dist_util.dev())
        mask = mask.to(dist_util.dev())
        mask_image = mask_image.to(dist_util.dev())
        edge = edge.to(dist_util.dev())
        mask_edge = mask_edge.to(dist_util.dev())
        gray_maskimage = gray_maskimage.to(dist_util.dev())
        gray_image = gray_image.to(dist_util.dev())
        sam_yuantu = sam_yuantu.to(dist_util.dev())
        sam_mask = sam_mask.to(dist_util.dev())


        image2 = image

        mask_image2 = mask_image


        model_kwargs = {}

        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        logger.log("image sampling...")

        jiazao_time1 = th.tensor([249])
        tensor1 = jiazao_time1.long().to(dist_util.dev())

        compute_losses1 = functools.partial(
            diffusion1.training_losses1test,
            model1,
            sam_model,
            image,
            mask_image,
            sam_yuantu,
            sam_mask,
            tensor1
        )

        sample3, sam_result_0255, input_x_t, input_damage_image, input_edges_tensor_guiyi, sam_image_input, _ = compute_losses1()


        image2 = ((image2 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image2 = image2.permute(0, 2, 3, 1)

        mask_image2 = ((mask_image2 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        mask_image2 = mask_image2.permute(0, 2, 3, 1)

        sam_result_0255 = sam_result_0255.clamp(0, 255).to(th.uint8)
        sam_result_0255 = sam_result_0255.permute(0, 2, 3, 1)

        input_x_t = ((input_x_t + 1) * 127.5).clamp(0, 255).to(th.uint8)
        input_x_t = input_x_t.permute(0, 2, 3, 1)

        input_damage_image = ((input_damage_image + 1) * 127.5).clamp(0, 255).to(th.uint8)
        input_damage_image = input_damage_image.permute(0, 2, 3, 1)

        input_edges_tensor_guiyi = ((input_edges_tensor_guiyi + 1) * 127.5).clamp(0, 255).to(th.uint8)
        input_edges_tensor_guiyi = input_edges_tensor_guiyi.permute(0, 2, 3, 1)

        sample33 = sample3
        sample33 = ((sample33 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample33 = sample33.permute(0, 2, 3, 1)
        sample33 = sample33.contiguous()

        image1 = Image.fromarray(sample33.squeeze().cpu().numpy())
        idx = len(all_images)
        file_name = f'{idx:05}.jpg'
        file_path = os.path.join(save_dir, file_name)
        image1.save(file_path)

        all_images.append("1")

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=2000,
        batch_size=1,
        use_ddim=False,
        test_datadir="",
        test_maskdir="",
        image_size=256,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    steps = []
    for step in steps:
        model_path = f""
        save_dir = f""
        main(model_path, save_dir)