import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
# from pytorch_msssim import ssim
import math
import clip

class FusionLoss(nn.Module):
    def __init__(self, clip_path):
        super(FusionLoss, self).__init__()
        self.Gradloss = GradientMaxLoss()
        self.colorloss = L_color()
        self.Intloss = L_Intensity()
        self.Consistloss = L_Intensity_Consist()
        self.ssimloss = L_SSIM()
        self.cliploss = Loss_Clip(clip_path=clip_path)

    def forward(self, vis, ir, fusion, target, grad_rate=10, ssim_rate=1, color_rate=20, int_rate=20, consist_rate=1, clip_rate=1):

        fusion_ycbcr = kornia.color.rgb_to_ycbcr(fusion)
        vis_ycbcr = kornia.color.rgb_to_ycbcr(vis)
        vis_y = vis_ycbcr[:, 0:1, :, :]
        ir_y = ir[:, 0:1, :, :]
        fusion_y = fusion_ycbcr[:, 0: 1, :, :]

        loss_grad = self.Gradloss(vis_y, ir_y, fusion_y) * grad_rate

        loss_ssim = (self.ssimloss(vis, fusion) + self.ssimloss(ir_y, fusion_y)) * ssim_rate

        loss_color = self.colorloss(vis, fusion) * color_rate

        loss_int = self.Intloss(vis_y, ir_y, fusion_y) * int_rate
        loss_consist = self.Consistloss(vis_y, ir_y, fusion_y, ir_compose=1, consist_mode='l1')*consist_rate

        loss_clip = self.cliploss(fusion, target) * clip_rate

        loss = loss_grad + loss_ssim + loss_color + loss_int + loss_consist + loss_clip
        # print(loss, loss_grad, loss_ssim, loss_color, loss_int, loss_consist)

        return loss, loss_grad, loss_ssim, loss_color, loss_int, loss_consist, loss_clip

class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image

class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to('cuda')
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.to('cuda')
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret



class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused))/2
        else:
            Loss_intensity = (F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused))/2
        return Loss_intensity

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def std1(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma_sq = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-10)) # 避免负数和零
    return sigma

def mse(img1, img2, window_size=9):
    padd = window_size // 2
    (_, _, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2
    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)
    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, img_vis, img_ir, img_fuse, mask=None):
        mse_ir = mse(img_ir, img_fuse)
        mse_vi = mse(img_vis, img_fuse)

        std_ir = std1(img_ir)
        std_vi = std1(img_vis)

        brightness_ir = torch.mean(img_ir, dim=1, keepdim=True)
        brightness_vi = torch.mean(img_vis, dim=1, keepdim=True)

        # 核心魔法：保持天空等亮色区域不被拉暗
        brightness_diff = brightness_vi - brightness_ir
        std_diff = std_vi - std_ir
        weight_map = torch.sigmoid((brightness_diff + std_diff))

        # print(weight_map.max(), weight_map.min(), weight_map.mean())

        if mask is not None:
            weight_map = weight_map * mask + (1 - mask) * weight_map

        # 交叉加权：权重大惩罚可见光误差，权重小惩罚红外误差
        res = weight_map * mse_vi + (1 - weight_map) * mse_ir
        return res.mean()

import torchvision.transforms as T
class Loss_Clip(nn.Module):
    def __init__(self, clip_path):
        super().__init__()
        self.clip, _ = clip.load(clip_path, device="cuda")
        for param in self.clip.parameters():
            param.requires_grad = False

        self.normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        self.resize = T.Resize((224, 224), antialias=True)

    @torch.no_grad()
    def get_text_feature(self, text):
        device = next(self.clip.parameters()).device
        text_tokens = clip.tokenize(text, truncate=True).to(device)
        # 直接返回未归一化的特征即可
        return self.clip.encode_text(text_tokens)

    def forward(self, fusion, target_text):
        fusion_bounded = torch.sigmoid(fusion)
        # print(fusion_bounded.max(), fusion_bounded.min(), fusion_bounded.mean())
        fusion_resized = self.resize(fusion_bounded)
        fusion_normalized = self.normalize(fusion_resized)

        # 提取图像特征 (未归一化)
        image_feature = self.clip.encode_image(fusion_normalized)

        # 提取文本特征 (未归一化)
        text_feature = self.get_text_feature(target_text)

        # 直接使用 F.cosine_similarity 计算
        cos_loss = 1.0 - F.cosine_similarity(image_feature.float(), text_feature.float(), dim=-1).mean()

        return cos_loss


