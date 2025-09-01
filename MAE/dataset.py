import os
import random
import sys
from glob import glob
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append('..')
import cv2


def to_int(x):
    return tuple(map(int, x))


class ImgMaskDataset(Dataset):
    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False, mask_rates=None, image_size=256):

        self.is_train = is_train
        self.pt_dataset = pt_dataset

        self.image_id_list = []
        with open(self.pt_dataset) as f:
            for line in f:
                self.image_id_list.append(line.strip())

        if is_train:
            if len(mask_path) > 1:
                self.irregular_mask_list = []
                with open(mask_path[0]) as f:
                    for line in f:
                        self.irregular_mask_list.append(line.strip())
                self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
                self.segment_mask_list = []
                with open(mask_path[1]) as f:
                    for line in f:
                        self.segment_mask_list.append(line.strip())
                self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
            else:
                total_masks = []
                with open(mask_path[0]) as f:
                    for line in f:
                        total_masks.append(line.strip())
                random.shuffle(total_masks)
                self.irregular_mask_list = total_masks[:len(total_masks) // 2]
                self.segment_mask_list = total_masks[len(total_masks) // 2:]
        else:
            self.mask_list = glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.image_size = image_size
        self.training = is_train
        self.mask_rates = mask_rates

        self.transform_train = transforms.Compose([
            transforms.Resize(image_size, interpolation=3),  # 3 is bicubic
            # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_id_list)

    def load_mask(self, index):
        imgh, imgw = self.image_size, self.image_size  # 请注意这里还是256的

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rates[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                print("self.irregular_mask_list[mask_index]:",self.irregular_mask_list[mask_index])
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
                mask_1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            elif rdv < self.mask_rates[1]:
                mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                print("self.segment_mask_list[mask_index]:",self.segment_mask_list[mask_index])
                mask = cv2.imread(self.segment_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
                mask_1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            else:
                mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                print("self.segment_mask_list[mask_index1]:",self.segment_mask_list[mask_index1])
                print("self.irregular_mask_list[mask_index2]:",self.irregular_mask_list[mask_index2])
                mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float64)
                mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float64)
                mask1 = cv2.resize(mask1, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask2 = cv2.resize(mask2, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)
                mask_1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

            # 最终应该还是走这里的  因为512！=256
            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
                mask_1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

            

            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation

            mask_1024 = (mask_1024 > 127).astype(np.uint8) * 255  # threshold due to interpolation

            return mask,mask_1024

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def __getitem__(self, idx):
        selected_img_name = self.image_id_list[idx]
        img = cv2.imread(selected_img_name)
        while img is None:
            print('Bad image {}...'.format(selected_img_name))
            idx = random.randint(0, len(self.image_id_list) - 1)
            img = cv2.imread(self.image_id_list[idx])

        img = img[:, :, ::-1]
        img = Image.fromarray(img)

        if self.training is True:
            # 注意，下面这些注释是在此py文件的main函数中跑的  没有使用dataloader 因此注释中没有bs这个维度
            img = self.transform_train(img)  # 得到256
            print("img.shape:",img.shape)  # torch.Size([3, 256, 256])
            # 自己加
            # 将256x256的图像放大到1024x1024 因为sam的输入需要满足此
            img_1024 = F.resize(img, (1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC)
            print("img_1024.shape:", img_1024.shape)  # torch.Size([3, 1024, 1024])


        # load mask
        mask,mask_1024 = self.load_mask(idx)
        # augment data
        if self.training is True:
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()
                mask_1024 = mask_1024[:, ::-1, ...].copy()
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()
                mask_1024 = mask_1024[::-1, :, ...].copy()

        mask = self.to_tensor(mask)
        mask_1024 = self.to_tensor(mask_1024)
        print("mask.shape:",mask.shape)  # torch.Size([1, 256, 256])
        print("mask_1024.shape:",mask_1024.shape)  # torch.Size([1, 256, 256])

        # 已经确认mask和mask_1024是只有0和1两个值的
        # # 使用unique方法获取唯一值
        # unique_values = torch.unique(mask_1024)
        #
        # # 打印唯一值
        # print(unique_values)

        meta = {'img': img,     # ([3, 256, 256])  图片已经被归一化  有归一化的均值和方差
                'img_1024':img_1024,  # ([3, 1024, 1024])
                'mask': mask,   # ([1, 256, 256]) 值仅为0或者1
                'mask_1024':mask_1024,  # # ([1, 1024, 1024]) 值仅为0或者1
                'name': os.path.basename(selected_img_name)}


        return meta


if __name__ == '__main__':
    dataset_train = ImgMaskDataset('/mnt/data/wxx/0MAE-FAR/train_celeba.txt', mask_path=['/mnt/data/wxx/0MAE-FAR/train_mask.txt'], is_train=True, mask_rates=[0.4, 0.8, 1.0],
                                   image_size=256)
    print(len(dataset_train))
    meta = dataset_train.__getitem__(1)

    print(meta.keys())
    print(meta['img'].shape)
    print(meta['img_1024'].shape)
    print(meta['mask'].shape)

    import os
    import numpy as np
    import torch
    from torchvision.transforms.functional import to_pil_image
    from PIL import Image



    # **图片的归一化和保存的函数
    def save_tensor_as_image(tensor, output_path, name):
        images = torch.einsum('chw->hwc', tensor)
        # images = torch.cat(tuple(x), dim=0)
        # images = torch.cat(
        #     [x.float()], dim=1)
        images = torch.clip((images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0,
                            255).int()
        images = images.numpy().astype(np.uint8)
        path = os.path.join(output_path, name)
        print("images.shape:",images.shape)
        print('\nsaving sample ' + path)
        cv2.imwrite(path, images[:, :, ::-1])

    # **mask的归一化和保存的函数
    def mask_save_tensor_as_image(tensor, output_path, name):
        images = torch.einsum('chw->hwc', tensor)
        # images = torch.cat(tuple(x), dim=0)
        # images = torch.cat(
        #     [x.float()], dim=1)
        images = torch.clip(images * 255, 0,
                            255).int()
        images = images.numpy().astype(np.uint8)
        path = os.path.join(output_path, name)
        print("images.shape:",images.shape)
        print('\nsaving sample ' + path)
        cv2.imwrite(path, images)


    # 定义输出路径和文件名
    output_dir = "/mnt/data/wxx/sam-inpaint/dataset_1"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    # 保存每个张量为图片
    save_tensor_as_image(meta['img'], output_dir, "tensor1.jpg")
    save_tensor_as_image(meta['img_1024'], output_dir, "tensor2.jpg")
    mask_save_tensor_as_image(meta['mask'], output_dir, "tensor3.jpg")
    print("name:",meta['name'])

    '''
    dict_keys(['img', 'img_1024', 'mask', 'name'])
    torch.Size([3, 256, 256])
    torch.Size([3, 1024, 1024])
    torch.Size([1, 256, 256])
    '''
