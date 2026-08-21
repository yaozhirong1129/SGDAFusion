import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from clip import *
from .encoder import ImageEncoder
from .decoder import ImageDecoder
import math

def load_dino_model(repo_path, model_name='dinov2_vitb14_reg', weights_path=None, device='cuda'):
    model = torch.hub.load(repo_path, model_name, source='local', pretrained=False)
    # 加载权重
    if weights_path:
        print(f"loading pth: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()  # 默认为评估模式
    return model

class DinoFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feats = {}
        self.hook_handles = []
        # 指定需要提取特征的目标层
        self.target_layers = {0}  # 比如 1, 5, 6, 10, 11
        self.register_hooks()

    def register_hooks(self):
        for i, block in enumerate(self.model.blocks):
            if i not in self.target_layers:
                continue

            def get_block_hook(layer_idx):
                def hook(module, input, output):
                    self.feats[f'block_{layer_idx}'] = output.detach()
                return hook

            handle = block.register_forward_hook(get_block_hook(i))
            self.hook_handles.append(handle)

    def extract(self, x):
        self.feats = {}
        with torch.no_grad():
            self.model(x)
        return self.feats

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

class SGDAFusion(nn.Module):
    def __init__(self, dino_repo, model_type, clip_path, device='cuda', embed_dim=768):
        super().__init__()
        print("Loading DINO...")
        self.dino_model = load_dino_model(dino_repo, model_type, device=device)
        for param in self.dino_model.parameters():
            param.requires_grad = False
        self.extractor = DinoFeatureExtractor(self.dino_model)

        self.visImageEncoder = ImageEncoder(inp_channels=3, dim=48, num_blocks=[2, 2, 2, 2], heads=[1, 2, 4, 8],
                                   ffn_expansion_factor=2, bias=False, LayerNorm_type='withBias')
        self.irImageEncoder = ImageEncoder(inp_channels=3, dim=48, num_blocks=[2, 2, 2, 2], heads=[1, 2, 4, 8],
                                   ffn_expansion_factor=2, bias=False, LayerNorm_type='withBias')

        self.clip, _ = clip.load(clip_path, device=device)
        for param in self.clip.parameters():
            param.requires_grad = False

        self.decoder = ImageDecoder(out_channels=3)

        self.joint_sft_l4 = JointSimDiffSFT(feat_dim=384, semantic_dim=768)
        self.joint_sft_l3 = JointSimDiffSFT(feat_dim=192, semantic_dim=768)
        self.joint_sft_l2 = JointSimDiffSFT(feat_dim=96, semantic_dim=768)
        self.joint_sft_l1 = JointSimDiffSFT(feat_dim=48, semantic_dim=768)

        self.fusion = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GELU()
        )
        self.Text_guide = Text_Guide(self.clip)

        self.parallel_fuse_l4_vis = ParallelAdapterFusion(dim=384)
        self.parallel_fuse_l4_ir = ParallelAdapterFusion(dim=384)

        self.parallel_fuse_l3_vis = ParallelAdapterFusion(dim=192)
        self.parallel_fuse_l3_ir = ParallelAdapterFusion(dim=192)

        self.parallel_fuse_l2_vis = ParallelAdapterFusion(dim=96)
        self.parallel_fuse_l2_ir = ParallelAdapterFusion(dim=96)

        self.parallel_fuse_l1_vis = ParallelAdapterFusion(dim=48)
        self.parallel_fuse_l1_ir = ParallelAdapterFusion(dim=48)

    @torch.no_grad()
    def get_text_feature(self, text):
        device = next(self.clip.parameters()).device
        text_tokens = clip.tokenize(text, truncate=True).to(device)
        text_feature = self.clip.encode_text(text_tokens)
        return F.normalize(text_feature.float(), dim=-1)

    def get_dino_block(self):
        return len(self.dino_model.blocks)

    def resize(self, x, size):
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
        return x

    def Pre_sematic(self, vis_dino, ir_dino):
        # 1. Base Prior (共性先验)
        base_prior = torch.cat([vis_dino, ir_dino], dim=1)  # [B, 1536, H, W]

        # 2. Diff Prior (差异先验)
        diff_v2i = F.relu(vis_dino - ir_dino)
        diff_i2v = F.relu(ir_dino - vis_dino)

        diff_prior = torch.cat([diff_v2i, diff_i2v], dim=1)  # [B, 1536, H, W]

        return base_prior, diff_prior

    def forward(self, vis_img, ir_img, text, Is_train=True):
        if Is_train:
            feats_vis = self.extractor.extract(vis_img)
            feats_ir = self.extractor.extract(ir_img)
            vis_dino = feats_vis['block_0']
            ir_dino = feats_ir['block_0']
            vis_dino = rearrange(vis_dino[:, 5:, :], 'b (h w) n -> b n h w', h=16)  # [B, 768, 16, 16]
            ir_dino = rearrange(ir_dino[:, 5:, :], 'b (h w) n -> b n h w', h=16)  # [B, 768, 16, 16]
            del feats_vis, feats_ir
        else:
            feats_vis = self.extractor.extract(self.resize(vis_img, 448))
            feats_ir = self.extractor.extract(self.resize(ir_img, 448))
            vis_dino = feats_vis['block_0']
            ir_dino = feats_ir['block_0']
            vis_dino = rearrange(vis_dino[:, 5:, :], 'b (h w) n -> b n h w', h=32)  # [B, 768, 32, 32]
            ir_dino = rearrange(ir_dino[:, 5:, :], 'b (h w) n -> b n h w', h=32)  # [B, 768, 32, 32]
            del feats_vis, feats_ir

        # 获取 DINO 的 base_prior 和 diff_prior
        base, diff = self.Pre_sematic(vis_dino, ir_dino)

        # 1. 提取 CNN 多尺度局部特征
        vis_l4, vis_l3, vis_l2, vis_l1 = self.visImageEncoder(vis_img)
        ir_l4, ir_l3, ir_l2, ir_l1 = self.irImageEncoder(ir_img)

        vis_feats_list = [vis_l1, vis_l2, vis_l3, vis_l4]
        ir_feats_list = [ir_l1, ir_l2, ir_l3, ir_l4]

        # ==========================================================
        # 2. 分支 A：DATGM
        # ==========================================================
        text_mod_vis, text_mod_ir = self.Text_guide(
            vis_img, ir_img, vis_feats_list, ir_feats_list, text
        )
        t_vis_l1, t_vis_l2, t_vis_l3, t_vis_l4 = text_mod_vis
        t_ir_l1, t_ir_l2, t_ir_l3, t_ir_l4 = text_mod_ir

        # ==========================================================
        # 3. 分支 B：MSFGM
        # ==========================================================
        sft_vis_l4, sft_ir_l4 = self.joint_sft_l4(vis_l4, ir_l4, base, diff)
        sft_vis_l3, sft_ir_l3 = self.joint_sft_l3(vis_l3, ir_l3, base, diff)
        sft_vis_l2, sft_ir_l2 = self.joint_sft_l2(vis_l2, ir_l2, base, diff)
        sft_vis_l1, sft_ir_l1 = self.joint_sft_l1(vis_l1, ir_l1, base, diff)

        # ==========================================================
        # 4. Feature Fusion: VTFM
        # ==========================================================
        out_vis_l4 = self.parallel_fuse_l4_vis(t_vis_l4, sft_vis_l4)
        out_ir_l4 = self.parallel_fuse_l4_ir(t_ir_l4, sft_ir_l4)

        out_vis_l3 = self.parallel_fuse_l3_vis(t_vis_l3, sft_vis_l3)
        out_ir_l3 = self.parallel_fuse_l3_ir(t_ir_l3, sft_ir_l3)

        out_vis_l2 = self.parallel_fuse_l2_vis(t_vis_l2, sft_vis_l2)
        out_ir_l2 = self.parallel_fuse_l2_ir(t_ir_l2, sft_ir_l2)

        out_vis_l1 = self.parallel_fuse_l1_vis(t_vis_l1, sft_vis_l1)
        out_ir_l1 = self.parallel_fuse_l1_ir(t_ir_l1, sft_ir_l1)

        # ==========================================================
        # 5. to Decoder
        # ==========================================================
        fusion_l4 = self.fusion(torch.cat([out_vis_l4, out_ir_l4], dim=1))

        vis_skips = (out_vis_l3, out_vis_l2, out_vis_l1)
        ir_skips = (out_ir_l3, out_ir_l2, out_ir_l1)

        out = self.decoder(fusion_l4, vis_skips, ir_skips)

        return out

class JointSimDiffSFT(nn.Module):
    """
    双流联合 SFT 模块:
    1. 基于 Base 语义保护 Vis/IR 各自的基础结构
    2. 基于 Diff 语义进行跨模态特征交互 (Cross-Injection)
    """
    def __init__(self, feat_dim, semantic_dim=768):
        super().__init__()

        self.compress_base = nn.Sequential(
            nn.Conv2d(semantic_dim * 2, semantic_dim, kernel_size=1),
            nn.InstanceNorm2d(semantic_dim, affine=True),
            nn.GELU(),
            nn.Conv2d(semantic_dim, feat_dim, kernel_size=1),
            nn.InstanceNorm2d(feat_dim, affine=True),
            nn.GELU()
        )
        self.compress_diff = nn.Sequential(
            nn.Conv2d(semantic_dim * 2, semantic_dim, kernel_size=1),
            nn.InstanceNorm2d(semantic_dim, affine=True),
            nn.GELU(),
            nn.Conv2d(semantic_dim, feat_dim, kernel_size=1),
            nn.InstanceNorm2d(feat_dim, affine=True),
            nn.GELU()
        )

        # ==========================================
        # Stage 1: Base Modulation (各自保护共性)
        # ==========================================
        # Vis 分支
        self.base_gamma_v = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.base_beta_v = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        # IR 分支
        self.base_gamma_i = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.base_beta_i = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)

        # ==========================================
        # Stage 2: Diff Cross-Interaction (跨模态交互)
        # ==========================================
        # Vis <- IR (Vis 向 IR 借取特征)
        self.cross_proj_v = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1) # 提取 IR 中的信息
        self.diff_gamma_v = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1) # Diff 门控
        self.diff_beta_v = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)  # Diff 偏置

        # IR <- Vis (IR 向 Vis 借取特征)
        self.cross_proj_i = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.diff_gamma_i = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.diff_beta_i = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)

        self.norm_v = nn.InstanceNorm2d(feat_dim, affine=False)
        self.norm_i = nn.InstanceNorm2d(feat_dim, affine=False)

    def forward(self, x_vis, x_ir, base_prior, diff_prior):
        # 1. 轻量化降维 prior [B, 1536, 32, 32] -> [B, feat_dim, 32, 32]
        base_cond = self.compress_base(base_prior)
        diff_cond = self.compress_diff(diff_prior)

        # 2. 空间插值对齐到当前 CNN 尺度
        target_size = x_vis.shape[-2:]
        if base_cond.shape[-2:] != target_size:
            base_cond = F.interpolate(base_cond, size=target_size, mode='bilinear', align_corners=False)
            diff_cond = F.interpolate(diff_cond, size=target_size, mode='bilinear', align_corners=False)

        norm_x_vis = self.norm_v(x_vis)
        norm_x_ir = self.norm_i(x_ir)

        # --------------------------------------------------
        # Stage 1: 基础特征稳定 (Self-Modulation)
        # --------------------------------------------------
        v_base = norm_x_vis * (self.base_gamma_v(base_cond) + 1.0) + self.base_beta_v(base_cond)
        i_base = norm_x_ir * (self.base_gamma_i(base_cond) + 1.0) + self.base_beta_i(base_cond)

        # --------------------------------------------------
        # Stage 2: 差异区域的跨模态注意力注入 (Cross-Interaction)
        # 逻辑：利用对方的 base 特征，通过 diff_cond 控制注入强度
        # --------------------------------------------------
        # Vis 接收 IR 信息
        ir_to_vis = self.cross_proj_v(i_base)
        v_diff_inj = ir_to_vis * (self.diff_gamma_v(diff_cond) + 1.0) + self.diff_beta_v(diff_cond)
        v_out = v_base + v_diff_inj

        vis_to_ir = self.cross_proj_i(v_base)
        i_diff_inj = vis_to_ir * (self.diff_gamma_i(diff_cond) + 1.0) + self.diff_beta_i(diff_cond)
        i_out = i_base + i_diff_inj

        return v_out, i_out

class Text_Guide(nn.Module):
    def __init__(self, clip_model, channels=[48, 96, 192, 384], embed_dim=512):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False

        # 1. 跨模态交互模块
        self.cross_vis = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cross_ir = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. 多尺度通道与空间注意力模块
        self.vis_channel_mlps = nn.ModuleList()
        self.ir_channel_mlps = nn.ModuleList()
        self.norm_vis_layers = nn.ModuleList()
        self.norm_ir_layers = nn.ModuleList()

        # 新增：用于将空间特征映射到 CLIP 维度，计算空间响应
        self.spatial_projs_vis = nn.ModuleList()
        self.spatial_projs_ir = nn.ModuleList()

        for dim in channels:
            self.vis_channel_mlps.append(nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.SiLU(),
                nn.Linear(256, dim)
            ))
            self.ir_channel_mlps.append(nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.SiLU(),
                nn.Linear(256, dim)
            ))
            self.norm_vis_layers.append(nn.InstanceNorm2d(dim, affine=False))
            self.norm_ir_layers.append(nn.InstanceNorm2d(dim, affine=False))

            # 新增映射层
            self.spatial_projs_vis.append(nn.Conv2d(dim, embed_dim, kernel_size=1))
            self.spatial_projs_ir.append(nn.Conv2d(dim, embed_dim, kernel_size=1))

    @torch.no_grad()
    def get_global_features(self, vis_img, ir_img, text):
        device = next(self.clip.parameters()).device

        text_tokens = clip.tokenize(text, truncate=True).to(device)
        text_feat = self.clip.encode_text(text_tokens).float()

        # resize 输入图像以匹配 CLIP
        vis_img_224 = F.interpolate(vis_img, size=(224, 224), mode='bilinear', align_corners=False)
        ir_img_224 = F.interpolate(ir_img, size=(224, 224), mode='bilinear', align_corners=False)

        vis_feat = self.clip.encode_image(vis_img_224.type(self.clip.dtype)).float()
        ir_feat = self.clip.encode_image(ir_img_224.type(self.clip.dtype)).float()

        return F.normalize(vis_feat, dim=-1), F.normalize(ir_feat, dim=-1), F.normalize(text_feat, dim=-1)

    def forward(self, global_vis, global_ir, vis_feats, ir_feats, text):
        vis_g, ir_g, text_g = self.get_global_features(global_vis, global_ir, text)

        vis_text_joint = vis_g * text_g
        ir_text_joint = ir_g * text_g

        cross_gate_vis = self.cross_vis(torch.cat([vis_text_joint, ir_text_joint], dim=-1))
        vis_refined = vis_text_joint + cross_gate_vis * ir_text_joint

        cross_gate_ir = self.cross_ir(torch.cat([ir_text_joint, vis_text_joint], dim=-1))
        ir_refined = ir_text_joint + cross_gate_ir * vis_text_joint

        out_vis_feats = []
        out_ir_feats = []

        for i in range(len(vis_feats)):
            # 1. 计算通道注意力权重 -> [B, C, 1, 1]
            vis_ch_weight = torch.sigmoid(self.vis_channel_mlps[i](vis_refined)).unsqueeze(-1).unsqueeze(-1)
            ir_ch_weight = torch.sigmoid(self.ir_channel_mlps[i](ir_refined)).unsqueeze(-1).unsqueeze(-1)

            # 2. 计算空间注意力权重 -> [B, 1, H, W]
            # 将图像特征投射到文本特征空间
            vis_spatial_q = self.spatial_projs_vis[i](vis_feats[i])
            ir_spatial_q = self.spatial_projs_ir[i](ir_feats[i])

            # 文本向量作为 Key，维度转为 [B, embed_dim, 1, 1]
            vis_text_k = vis_refined.unsqueeze(-1).unsqueeze(-1)
            ir_text_k = ir_refined.unsqueeze(-1).unsqueeze(-1)

            # 内积计算空间响应图 (点乘相似度)
            vis_sp_weight = torch.sigmoid(torch.sum(vis_spatial_q * vis_text_k, dim=1, keepdim=True))
            ir_sp_weight = torch.sigmoid(torch.sum(ir_spatial_q * ir_text_k, dim=1, keepdim=True))

            # 3. 联合调制
            v_modulated = self.norm_vis_layers[i](vis_feats[i]) * vis_ch_weight * vis_sp_weight
            i_modulated = self.norm_ir_layers[i](ir_feats[i]) * ir_ch_weight * ir_sp_weight

            out_vis_feats.append(v_modulated)
            out_ir_feats.append(i_modulated)

        return out_vis_feats, out_ir_feats

class ParallelAdapterFusion(nn.Module):
    """
    针对双并行分支设计的空间级 MoE 融合模块
    动态评估 Text (语义) 和 SFT (结构) 特征的像素级重要性
    """

    def __init__(self, dim):
        super().__init__()

        self.router = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, kernel_size=1, bias=False),
            nn.InstanceNorm2d(dim // 2, affine=True),
            nn.GELU(),
            nn.Conv2d(dim // 2, 2, kernel_size=3, padding=1)  # 输出通道为2，对应2个专家
        )

        self.expert_text = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU()
        )
        self.expert_sft = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU()
        )

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, feat_text, feat_sft):
        # 1. 状态拼接，交由 Router 进行评估
        concat_feat = torch.cat([feat_text, feat_sft], dim=1)

        # 2. 生成路由日志并计算权重 [B, 2, H, W]
        routing_logits = self.router(concat_feat)

        # 使用 Softmax 确保在通道维度上（2个专家）权重和为 1
        routing_weights = F.softmax(routing_logits, dim=1)

        # 分离权重 [B, 1, H, W]
        weight_text = routing_weights[:, 0:1, :, :]
        weight_sft = routing_weights[:, 1:2, :, :]

        # 3. 专家特征增强
        out_text = self.expert_text(feat_text)
        out_sft = self.expert_sft(feat_sft)

        # 4. 空间加权融合
        fused_feat = weight_text * out_text + weight_sft * out_sft

        # 5. 特征平滑
        return self.out_proj(fused_feat)
