import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import kornia
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'bmp', 'PNG', 'gif'])


def rgb_to_ycbcr(img):
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr

def randrot(img):
    # 随机选择旋转模式
    mode = np.random.randint(0, 3)  # Rotating in 90-degree increments
    return rot(img, mode)

def randfilp(img):
    # 随机选择翻转模式
    mode = np.random.randint(0, 2)  # Flipping either vertically or horizontally
    return flip(img, mode)

def rot(img, rot_mode):
    # 根据rot_mode选择旋转模式
    if rot_mode == 0:  # 90 degrees clockwise
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:  # 180 degrees
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:  # 270 degrees clockwise (or 90 degrees counterclockwise)
        img = img.transpose(-2, -1)
        img = img.flip(-1)
    return img

def flip(img, flip_mode):
    # 根据flip_mode选择翻转模式
    if flip_mode == 0:
        img = img.flip(-2)  # Vertical flip
    elif flip_mode == 1:
        img = img.flip(-1)  # Horizontal flip
    return img

class DataLoaderTrain(Dataset):
    def __init__(self, img_dir, csv_path):
        super(DataLoaderTrain, self).__init__()

        ir_files = sorted(os.listdir(os.path.join(img_dir, 'ir')))
        vis_files = sorted(os.listdir(os.path.join(img_dir, 'vis')))

        self.ir_filenames = [os.path.join(img_dir, 'ir', x) for x in ir_files if is_image_file(x)]
        self.vis_filenames = [os.path.join(img_dir, 'vis', x) for x in vis_files if is_image_file(x)]

        self.sizex = len(self.ir_filenames)
        self.csv_path = csv_path
        self.text_mapping = {}  # 变为字典以存储映射关系
        self.target_mapping = {}
        # self.crop = torchvision.transforms.RandomCrop(224)
        self.crop = transforms.Resize((224, 224))

        if self.csv_path and os.path.exists(self.csv_path):
            self.load_csv_mapping()
        else:
            print(f"Warning: CSV path {self.csv_path} not found or not provided.")

    def __len__(self):
        return self.sizex

    def load_csv_mapping(self):
        try:
            df = pd.read_csv(self.csv_path)
            # 去除列名可能带有的前后空格，防止匹配失败
            df.columns = [c.strip() for c in df.columns]

            for index, row in df.iterrows():
                if 'Filename' in row and 'Analysis_Summary' in row and 'Target' in row:
                    raw_filename = str(row['Filename'])
                    # 关键修改：去掉 CSV 里文件名的后缀，变成 '00002'，这样才能和图片读取时匹配
                    name_key = os.path.splitext(raw_filename)[0]
                    self.text_mapping[name_key] = str(row['Analysis_Summary'])
                    self.target_mapping[name_key] = str(row['Target'])
        except Exception as e:
            print(f"Error reading CSV: {e}")

    def __getitem__(self, index):
        index_ = index % self.sizex

        ir_path = self.ir_filenames[index_]
        vis_path = self.vis_filenames[index_]

        # 统一转为 RGB 格式
        ir_img = Image.open(ir_path).convert('RGB')
        vis_img = Image.open(vis_path).convert('RGB')

        w_vi, h_vi = vis_img.size  # (width, height)
        w_ir, h_ir = ir_img.size
        new_w = max(16, (w_vi // 16) * 16)
        new_h = max(16, (h_vi // 16) * 16)

        # 先默认等于原图
        image_vis = vis_img
        image_ir = ir_img

        if (w_vi != new_w) or (h_vi != new_h):
            image_vis = vis_img.resize((new_w, new_h), resample=Image.BICUBIC)
        if (w_ir != new_w) or (h_ir != new_h):
            image_ir = ir_img.resize((new_w, new_h), resample=Image.BICUBIC)

        vis_img = TF.to_tensor(image_vis).unsqueeze(0)  # [1, 3, H, W]
        ir_img = TF.to_tensor(image_ir).unsqueeze(0)  # [1, 3, H, W]

        vis_ir = torch.cat([vis_img, ir_img], dim=1)

        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)
        vis_ir = self.crop(vis_ir)

        vis_img, ir_img = torch.split(vis_ir, [3, 3], dim=1)

        # 提取当前图片文件名（无后缀），例如 '00002'
        filename = os.path.splitext(os.path.split(ir_path)[-1])[0]

        # 从字典中获取文本，如果没有匹配到，则返回空字符串
        text_label = self.text_mapping.get(filename, "")
        target = self.target_mapping.get(filename, "")

        return ir_img.squeeze(0), vis_img.squeeze(0), filename, text_label, target


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, csv_path):
        super(DataLoaderTest, self).__init__()
        inp_ir_files = sorted(os.listdir(os.path.join(inp_dir, 'ir')))
        inp_rgb_files = sorted(os.listdir(os.path.join(inp_dir, 'vis')))

        self.inp_ir_filenames = [os.path.join(inp_dir, 'ir', x) for x in inp_ir_files if is_image_file(x)]
        self.inp_rgb_filenames = [os.path.join(inp_dir, 'vis', x) for x in inp_rgb_files if is_image_file(x)]

        self.inp_ir_size = len(self.inp_ir_filenames)
        self.inp_rgb_size = len(self.inp_rgb_filenames)

        self.csv_path = csv_path
        self.text_mapping = {}

        if self.csv_path and os.path.exists(self.csv_path):
            self.load_csv_mapping()
        else:
            print(f"Warning: CSV path {self.csv_path} not found or not provided.")

    def __len__(self):
        return self.inp_ir_size

    def load_csv_mapping(self):
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = [c.strip() for c in df.columns]

            for index, row in df.iterrows():
                if 'Filename' in row and 'Analysis_Summary' in row:
                    raw_filename = str(row['Filename'])
                    name_key = os.path.splitext(raw_filename)[0]
                    self.text_mapping[name_key] = str(row['Analysis_Summary'])
        except Exception as e:
            print(f"Error reading CSV: {e}")

    def __getitem__(self, index):
        path_ir_inp = self.inp_ir_filenames[index]
        path_rgb_inp = self.inp_rgb_filenames[index]

        inp_ir = Image.open(path_ir_inp).convert('RGB')
        inp_rgb = Image.open(path_rgb_inp).convert('RGB')

        # 测试集：固定中心裁剪，不加随机翻转等增强
        # inp_rgb = center_crop_arr(inp_rgb, 504)
        # inp_ir = center_crop_arr(inp_ir, 504)

        inp_ir = TF.to_tensor(inp_ir)
        inp_rgb = TF.to_tensor(inp_rgb)

        filename = os.path.splitext(os.path.split(path_ir_inp)[-1])[0]
        text_label = self.text_mapping.get(filename, "")

        vis224 = F.interpolate(inp_rgb.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        ir224 = F.interpolate(inp_ir.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

        return inp_ir, inp_rgb, filename, text_label, vis224.squeeze(0), ir224.squeeze(0)


