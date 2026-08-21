import torch
import torch.nn as nn
from .encoder import TransformerBlock
import torch.nn.functional as F

class ImageDecoder(nn.Module):
    def __init__(self, out_channels=3, dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=1, heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2, bias=False,
                 LayerNorm_type='WithBias'):
        super(ImageDecoder, self).__init__()

        # 各层级的通道维度计算
        dim4 = int(dim * 2 ** 3)  # 384 (假设 dim=48)
        dim3 = int(dim * 2 ** 2)  # 192
        dim2 = int(dim * 2 ** 1)  # 96
        dim1 = int(dim)  # 48

        # --- Decoder L4 ---
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=dim4, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[3])])

        # --- Level-3 ---
        self.feature_fusion_3 = Fusion_Embed(embed_dim=dim3)
        self.up4_3 = Upsample(dim4)  # Upsample内部会把通道减半: 384 -> 192
        self.reduce_chan_level3 = nn.Conv2d(dim4, dim3, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim3, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        # --- Level-2 ---
        self.feature_fusion_2 = Fusion_Embed(embed_dim=dim2)
        self.up3_2 = Upsample(dim3)  # 192 -> 96
        self.reduce_chan_level2 = nn.Conv2d(dim3, dim2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        # --- Level-1 ---
        self.feature_fusion_1 = Fusion_Embed(embed_dim=dim1)
        self.up2_1 = Upsample(dim2)  # 96 -> 48

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        # --- Refinement + Output ---
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, fusion_l4, vis_skips, ir_skips, attn_map=None):
        """
        参数:
            fusion_l4: (B, 384, H/8, W/8) 已经在外部交互融合好的 L4 主特征
            vis_skips: tuple 包含 (vis_l3, vis_l2, vis_l1)
            ir_skips: tuple 包含 (ir_l3, ir_l2, ir_l1)
        """
        vis_l3, vis_l2, vis_l1 = vis_skips
        ir_l3, ir_l2, ir_l1 = ir_skips

        # ---- Level-4 ----
        out_dec_level4 = self.decoder_level4(fusion_l4)  # (B, 384, H/8, W/8)

        # ---- Level-3 ----
        inp_dec_level3 = self.up4_3(out_dec_level4)  # -> (B, 192, H/4, W/4)
        out_enc_level3 = self.feature_fusion_3(vis_l3, ir_l3, attn_map=attn_map)  # 融合跳跃连接
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)  # (B, 384, H/4, W/4)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # (B, 192, H/4, W/4)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # ---- Level-2 ----
        inp_dec_level2 = self.up3_2(out_dec_level3)  # -> (B, 96, H/2, W/2)
        out_enc_level2 = self.feature_fusion_2(vis_l2, ir_l2, attn_map=attn_map)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)  # (B, 192, H/2, W/2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # (B, 96, H/2, W/2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # ---- Level-1 ----
        inp_dec_level1 = self.up2_1(out_dec_level2)  # -> (B, 48, H, W)
        out_enc_level1 = self.feature_fusion_1(vis_l1, ir_l1, attn_map=attn_map)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)  # (B, 96, H, W)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # (B, 96, H, W)

        # ---- Refinement & Output ----
        out_dec_level1 = self.refinement(out_dec_level1)  # (B, 96, H, W)
        out = self.output(out_dec_level1)  # (B, out_channels, H, W)

        return out


# class Fusion_Embed(nn.Module):
#     def __init__(self, embed_dim, bias=False):
#         super(Fusion_Embed, self).__init__()
#
#         self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)
#
#     def forward(self, x_A, x_B):
#         x = torch.concat([x_A, x_B], dim=1)
#         x = self.fusion_proj(x)
#         return x

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()
        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B, attn_map=None):
        # 1. 正常的特征拼接和降维
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)

        # 2. 如果提供了文本注意力图，则进行空间引导
        if attn_map is not None:
            B, C, H, W = x.shape

            # 当前 attn_map 维度为 [Batch, 77, H_ori, W_ori]
            # 我们在第1维度（dim=1，即 77个Token 的维度）上取最大值
            # keepdim=True 保证输出维度是 [Batch, 1, H_ori, W_ori]
            attn_single_channel, _ = torch.max(attn_map, dim=1, keepdim=True)

            # 缩放 1通道的 attn_single_channel 以匹配当前特征图的尺寸 (H, W)
            attn_resized = F.interpolate(attn_single_channel, size=(H, W), mode='bilinear', align_corners=False)

            # (可选) 对注意力图做一下归一化或限制，防止数值过大导致梯度爆炸
            # 例如让它严格分布在 0~1 之间
            attn_resized = torch.sigmoid(attn_resized)

            # 乘以特征：不在意的地方乘 1(保持原样)，在意的地方乘 1~2(特征增强)
            x = x * (1.0 + attn_resized)

        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)