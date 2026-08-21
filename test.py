import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import utils
from dataloader.datacheck import get_test_data
import warnings
from model.fusion_transformer import SGDAFusion
from utils.test_utils import Entropy, Standard_Dev, Spatial_freq, Average_grad, Sum_Corr_Diff, VIF_Fusion
warnings.filterwarnings("ignore", category=UserWarning)
import kornia
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='Image Fusion using FreeFusion')
parser.add_argument('--input_dir', default='/public/yzr/A/', type=str, help='Directory for results')
parser.add_argument('--result_dir', default='./results/RGBT', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./checkpoints/save_pth/night/models/model_latest.pth',
                    type=str, help='Path to weights')
# parser.add_argument('--gpus', default='6', type=str, help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


datasets = ['RGBTPerson']
# 初始化累加器
EN_sum = SD_sum = SF_sum = AG_sum = SCD_sum = VIF_sum = 0
total_imgs = 0  # 实际统计处理过的图片数量

hub_path = './dinov2/'
model_type = 'dinov2_vitb14_reg'
pth_path = './dinov2/dinov2_vitb14_reg4_pretrain.pth'
clip_path = './ViT-B-32.pt'
csv_dir = '/public/yzr/A/RGBTPerson/degradation.csv'

model = SGDAFusion(
    dino_repo=hub_path,
    model_type=model_type,
    clip_path=clip_path,
    device='cuda'
)

pretrained_dict = torch.load(args.weights, map_location='cpu')
state_dict = pretrained_dict['state_dict']
model.load_state_dict(state_dict)
model.cuda()
for param in model.parameters():
    param.requires_grad = False
print("===>Testing using weights: ", args.weights)

import cv2


def save_pic(outputpic, save_path):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    cv2.imwrite(save_path, outputpic)
def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img


for dataset in datasets:
    img_dir_test = os.path.join(args.input_dir, dataset)
    test_dataset = get_test_data(img_dir_test, csv_dir)
    # 建议: shuffle=False 保持顺序，num_workers根据CPU核心数调整
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                             pin_memory=True)

    utils.mkdir(args.result_dir)

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):

            inp_ir = data_test[0].cuda()
            inp_rgb = data_test[1].cuda()
            filenames = data_test[2]
            vis_text = data_test[3]
            vis224 = data_test[4].cuda()
            ir224 = data_test[5].cuda()

            current_batch_size = inp_ir.size(0)
            total_imgs += current_batch_size

            fus = model(inp_rgb, inp_ir, vis_text, Is_train=False)
            # print(fus.max(), fus.min(), fus.mean())

            # b, c, h, w = fus.shape
            # fus_flat = fus.view(b, -1)
            # q_min = torch.quantile(fus_flat, 0.01, dim=-1, keepdim=True).view(b, 1, 1, 1)
            # q_max = torch.quantile(fus_flat, 0.99, dim=-1, keepdim=True).view(b, 1, 1, 1)
            #
            # # 截断离群点并进行 Min-Max 拉伸
            # fus_norm = torch.clamp(fus, q_min, q_max)
            # fus_norm = (fus_norm - q_min) / (q_max - q_min + 1e-8)

            # fusion_ycbcr = kornia.color.rgb_to_ycbcr(fus)
            # vis_ycbcr = kornia.color.rgb_to_ycbcr(inp_rgb)
            # vis_y = vis_ycbcr[:, 0:1, :, :]
            # ir_y = inp_ir[:, 0:1, :, :]
            # fusion_y = fusion_ycbcr[:, 0: 1, :, :]
            #
            # # 2. 加权累加
            # EN_sum += Entropy(fusion_y).item() * current_batch_size
            # SD_sum += Standard_Dev(fusion_y).item() * current_batch_size
            # SF_sum += Spatial_freq(fusion_y).item() * current_batch_size
            # AG_sum += Average_grad(fusion_y).item() * current_batch_size
            #
            # # SCD 和 VIF 通常需要原始输入也在 0-1 之间，确保输入数据归一化正确
            # SCD_sum += Sum_Corr_Diff(fusion_y, ir_y, vis_y).item() * current_batch_size
            # VIF_sum += VIF_Fusion(fusion_y, ir_y, vis_y).item() * current_batch_size

            for batch_idx in range(current_batch_size):
                # 取出单张图像的 tensor
                single_img_tensor = fus[batch_idx, :, :, :]  # 取归一化后的 tensor
                save_path = os.path.join(args.result_dir, filenames[batch_idx] + '.png')

                # 转 Numpy 并保存
                fused_img = tensor2numpy(single_img_tensor)
                save_pic(fused_img, save_path)

    # # 计算全局平均值
    # EN_avg = EN_sum / total_imgs
    # SD_avg = SD_sum / total_imgs
    # SF_avg = SF_sum / total_imgs
    # AG_avg = AG_sum / total_imgs
    # SCD_avg = SCD_sum / total_imgs
    # VIF_avg = VIF_sum / total_imgs

    # print(f"Total Images: {total_imgs}")
    # print(
    #     f"EN_avg: {EN_avg:.4f}",
    #     f"\nSD_avg: {SD_avg:.4f}",
    #     f"\nSF_avg: {SF_avg:.4f}",
    #     f"\nAG_avg: {AG_avg:.4f}",
    #     f"\nSCD_avg: {SCD_avg:.4f}",
    #     f"\nVIF_avg: {VIF_avg:.4f}"
    # )

