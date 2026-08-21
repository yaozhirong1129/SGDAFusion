import os
import time
import random
import argparse
import warnings
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from config import Config
from dataloader.datacheck import get_training_data
from model.fusion_transformer import SGDAFusion
from utils import mkdir
from utils.loss import FusionLoss

warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 图像后处理与保存工具函数
# ==========================================
def tensor2numpy(img_tensor):
    """将 [C, H, W] 维度的 Tensor 转换为 OpenCV 可用的 [H, W, C] numpy 数组"""
    img = img_tensor.cpu().detach().numpy()
    if len(img.shape) == 3:
        img = np.transpose(img, [1, 2, 0])
    return img


def save_pic(outputpic, save_path):
    """截断异常值并保存图像"""
    outputpic = np.clip(outputpic, 0.0, 1.0)
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)

    # RGB 转 BGR 以便 OpenCV 写入
    if len(outputpic.shape) == 3 and outputpic.shape[2] == 3:
        outputpic = outputpic[:, :, ::-1]

    cv2.imwrite(save_path, outputpic.astype(np.uint8))


def set_seed(seed=1234):
    """固定随机种子确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# 参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="SGDAFusion Training Configuration")

    # 核心配置文件
    parser.add_argument('--config', type=str, default='training.yml', help='Path to training config YAML')

    # 学习率与调度策略
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup training epochs')
    parser.add_argument('--warmup_multiplier', type=float, default=1.0, help='Multiplier for gradual warmup')

    # 日志与保存频率
    parser.add_argument('--log_freq', type=int, default=100, help='Step frequency for logging scalar losses')
    parser.add_argument('--vis_freq', type=int, default=200, help='Step frequency for visual sample generation')
    parser.add_argument('--n_show', type=int, default=4, help='Number of sample rows to visualize')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='Epoch frequency for saving checkpoints')

    # 训练控制
    parser.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES index')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')

    return parser.parse_args()


# ==========================================
# 训练主流程
# ==========================================
def main():
    args = parse_args()

    # GPU 与基础环境配置
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    opt = Config(args.config)

    # 目录创建
    data_name = opt.Datasets.data
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, data_name, 'models')
    sample_dir = os.path.join(opt.TRAINING.SAVE_DIR, data_name, 'samples')
    mkdir(model_dir)
    mkdir(sample_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    csv_dir = opt.TRAINING.CSV_DIR

    # 模型初始化
    model = SGDAFusion(
        dino_repo=opt.DinoSetting.hub_path,
        model_type=opt.DinoSetting.model_type,
        clip_path=opt.clipSetting.clip_path,
        device='cuda'
    ).cuda()

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.OPTIM.LR_INITIAL,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0001
    )

    # 学习率调度器 (Warmup + Cosine Annealing)
    cosine_epochs = max(1, opt.OPTIM.NUM_EPOCHS - args.warmup_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=opt.OPTIM.LR_MIN
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=args.warmup_multiplier,
        total_epoch=args.warmup_epochs,
        after_scheduler=scheduler_cosine
    )

    # 损失函数 (重命名避免覆盖类名)
    criterion = FusionLoss(opt.clipSetting.clip_path).cuda()

    # 断点续训恢复
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cuda')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"===> Resumed from Epoch {start_epoch}")
        else:
            print(f"===> Checkpoint not found at {args.resume}, starting from scratch.")

    # 数据加载器
    train_dataset = get_training_data(train_dir, csv_dir)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    # TensorBoard 日志记录器
    log_dir = f"./checkpoints/log/{datetime.now().strftime('%b%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    print(f"===> Start Training: Epoch {start_epoch} to {opt.OPTIM.NUM_EPOCHS}")

    global_step = 0
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        model.train()
        for i, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{opt.OPTIM.NUM_EPOCHS}")):
            optimizer.zero_grad(set_to_none=True)

            ir_img, vis_img, _, text, target = batch_data
            ir_img = ir_img.cuda()
            vis_img = vis_img.cuda()

            fusion = model(vis_img, ir_img, text)

            loss, loss_grad, loss_ssim, loss_color, loss_int, loss_consist, loss_clip = criterion(
                vis_img, ir_img, fusion, target
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # 标量 Loss 监控
            if global_step % args.log_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/ssim', loss_ssim.item(), global_step)
                writer.add_scalar('Loss/grad', loss_grad.item(), global_step)
                writer.add_scalar('Loss/color', loss_color.item(), global_step)
                writer.add_scalar('Loss/int', loss_int.item(), global_step)
                writer.add_scalar('Loss/consist', loss_consist.item(), global_step)
                writer.add_scalar('Loss/clip', loss_clip.item(), global_step)
                writer.add_scalar('LearningRate', current_lr, global_step)

            # 可视化结果采样
            if global_step % args.vis_freq == 0:
                model.eval()
                with torch.no_grad():
                    n_show = min(args.n_show, ir_img.size(0))
                    ir_show = ir_img[:n_show, :1].repeat(1, 3, 1, 1)
                    vis_show = vis_img[:n_show]
                    fusion_show = fusion[:n_show]

                    # 拼接对比图 [IR | Vis | Fusion]
                    row_list = [
                        torch.cat([ir_show[idx:idx + 1], vis_show[idx:idx + 1], fusion_show[idx:idx + 1]], dim=3)
                        for idx in range(n_show)
                    ]
                    comparison = torch.cat(row_list, dim=0)
                    grid = vutils.make_grid(comparison, nrow=1, padding=2, normalize=True, scale_each=True)

                    grid_np = tensor2numpy(grid)
                    img_name = f"epoch_{epoch}_step_{global_step}.png"
                    save_pic(grid_np, os.path.join(sample_dir, img_name))

                model.train()

            global_step += 1

        # 调度器步进
        scheduler.step()

        # 模型权重保存
        if (epoch % args.save_epoch_freq == 0) or (epoch == opt.OPTIM.NUM_EPOCHS):
            save_payload = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(save_payload, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
            torch.save(save_payload, os.path.join(model_dir, "model_latest.pth"))

        # Epoch 统计打印
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: [{epoch}/{opt.OPTIM.NUM_EPOCHS}] | Time: {time.time() - epoch_start_time:.2f}s | "
              f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.8f}")

    writer.close()


if __name__ == '__main__':
    main()