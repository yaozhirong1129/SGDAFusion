import torch


def Entropy(input_tensor):
    """
    计算 Tensor 的信息熵
    策略: 无论输入范围是多少，强制拉伸(Min-Max Normalize)到 0-255
    """
    # 1. 维度处理
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    batch_size, channels, h, w = input_tensor.shape
    entropy_list = []

    for b in range(batch_size):
        for c in range(channels):
            img = input_tensor[b, c]

            # --- 核心步骤: 统一归一化 ---
            min_val = img.min()
            max_val = img.max()

            # 防止纯色图像导致除以0
            if max_val - min_val < 1e-6:
                entropy_list.append(torch.tensor(0.0, device=img.device))
                continue

            # (x - min) / (max - min) -> [0, 1]
            img = (img - min_val) / (max_val - min_val)

            # 映射到 [0, 255] 并转整数
            img = (img * 255).long()

            # --- 计算直方图与熵 ---
            hist = torch.bincount(img.view(-1), minlength=256).float()
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            entropy = -torch.sum(prob * torch.log2(prob))

            entropy_list.append(entropy)

    return torch.stack(entropy_list).mean()

def Standard_Dev(input_tensor):
    """
    计算 Tensor 的标准差 (Standard Deviation)

    参数:
        input_tensor: (B, C, H, W) 或 (C, H, W)
    返回:
        平均 SD 值 (scalar)
    """
    # 1. 维度处理
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    batch_size, channels, h, w = input_tensor.shape
    sd_list = []

    for b in range(batch_size):
        for c in range(channels):
            img = input_tensor[b, c]

            # --- 2. 统一归一化 (Min-Max) 到 0-255 ---
            # 这一步是为了让 SD 的数值符合一般论文的量级 (通常在 30~60 之间)
            min_val = img.min()
            max_val = img.max()

            if max_val - min_val < 1e-6:
                # 纯色图像 SD 为 0
                sd_list.append(torch.tensor(0.0, device=img.device))
                continue

            # 拉伸到 0-255 (保持 float 类型计算精度)
            img_normalized = (img - min_val) / (max_val - min_val) * 255.0

            # --- 3. 计算标准差 ---
            # torch.std 默认计算无偏估计 (除以 N-1)，这在统计学上更准确
            sd_val = torch.std(img_normalized)
            sd_list.append(sd_val)

    return torch.stack(sd_list).mean()

def Spatial_freq(input_tensor):
    """
    计算 Tensor 的空间频率 (Spatial Frequency)
    公式: SF = sqrt(RF^2 + CF^2)
    其中 RF 为行频率 (水平梯度), CF 为列频率 (垂直梯度)

    参数:
        input_tensor: (B, C, H, W) 或 (C, H, W)
    返回:
        平均 SF 值 (scalar)
    """
    # 1. 维度处理
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    batch_size, channels, h, w = input_tensor.shape
    sf_list = []

    for b in range(batch_size):
        for c in range(channels):
            img = input_tensor[b, c]

            # --- 2. 统一归一化 (Min-Max) 到 0-255 ---
            min_val = img.min()
            max_val = img.max()

            if max_val - min_val < 1e-6:
                # 纯色图像没有梯度，SF=0
                sf_list.append(torch.tensor(0.0, device=img.device))
                continue

            # 拉伸到 0-255 (保持 float 进行梯度计算)
            img = (img - min_val) / (max_val - min_val) * 255.0

            # --- 3. 计算 RF (Row Frequency) - 水平梯度 ---
            # diff_x = I(i, j) - I(i, j-1)
            # 使用切片计算相邻像素差值
            row_diff = img[:, 1:] - img[:, :-1]
            rf = torch.sqrt(torch.mean(row_diff ** 2))

            # --- 4. 计算 CF (Column Frequency) - 垂直梯度 ---
            # diff_y = I(i, j) - I(i-1, j)
            col_diff = img[1:, :] - img[:-1, :]
            cf = torch.sqrt(torch.mean(col_diff ** 2))

            # --- 5. 计算最终 SF ---
            sf = torch.sqrt(rf ** 2 + cf ** 2)
            sf_list.append(sf)

    return torch.stack(sf_list).mean()

def Average_grad(input_tensor):
    """
    计算 Tensor 的平均梯度 (Average Gradient)
    公式: mean( sqrt( (dx^2 + dy^2) / 2 ) )

    参数:
        input_tensor: (B, C, H, W) 或 (C, H, W)
    返回:
        平均 AG 值 (scalar)
    """
    # 1. 维度处理
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    batch_size, channels, h, w = input_tensor.shape
    ag_list = []

    for b in range(batch_size):
        for c in range(channels):
            img = input_tensor[b, c]

            # --- 2. 统一归一化 (Min-Max) 到 0-255 ---
            min_val = img.min()
            max_val = img.max()

            if max_val - min_val < 1e-6:
                ag_list.append(torch.tensor(0.0, device=img.device))
                continue

            # 拉伸到 0-255 (保持 float)
            img = (img - min_val) / (max_val - min_val) * 255.0

            # --- 3. 计算梯度 (差分) ---
            # X方向梯度: I(i, j+1) - I(i, j) -> 形状 (H, W-1)
            grad_x = img[:, 1:] - img[:, :-1]

            # Y方向梯度: I(i+1, j) - I(i, j) -> 形状 (H-1, W)
            grad_y = img[1:, :] - img[:-1, :]

            # --- 4. 对齐形状以进行逐像素计算 ---
            # 为了计算 dx^2 + dy^2，我们需要取两个矩阵的交集区域 (H-1, W-1)
            # grad_x 砍掉最后一行
            g_x = grad_x[:-1, :]
            # grad_y 砍掉最后一列
            g_y = grad_y[:, :-1]

            # --- 5. 计算 AG ---
            # 这里的 2 是公式里 sqrt((dx^2+dy^2)/2) 的分母
            gradient_map = torch.sqrt((g_x ** 2 + g_y ** 2) / 2)

            ag = torch.mean(gradient_map)
            ag_list.append(ag)

    return torch.stack(ag_list).mean()

def Sum_Corr_Diff(fused_tensor, source_a_tensor, source_b_tensor):
    """
    计算 SCD (Sum of Correlation Differences)
    公式: SCD = r(F - S_A, S_B) + r(F - S_B, S_A)

    参数:
        fused_tensor:    融合图像 (B, C, H, W)
        source_a_tensor: 源图像A (通常是 IR)
        source_b_tensor: 源图像B (通常是 VIS)

    返回:
        平均 SCD 值 (scalar)
    """
    # 1. 维度对齐
    if fused_tensor.dim() == 2:
        fused_tensor = fused_tensor.unsqueeze(0).unsqueeze(0)
        source_a_tensor = source_a_tensor.unsqueeze(0).unsqueeze(0)
        source_b_tensor = source_b_tensor.unsqueeze(0).unsqueeze(0)
    elif fused_tensor.dim() == 3:
        fused_tensor = fused_tensor.unsqueeze(0)
        source_a_tensor = source_a_tensor.unsqueeze(0)
        source_b_tensor = source_b_tensor.unsqueeze(0)

    batch_size, channels, h, w = fused_tensor.shape
    scd_list = []

    for b in range(batch_size):
        for c in range(channels):
            # 获取单张图
            F = fused_tensor[b, c]
            A = source_a_tensor[b, c]
            B = source_b_tensor[b, c]

            # --- 2. 统一归一化 (可选，但推荐以消除量纲影响) ---
            # 虽然相关系数对线性变换不敏感，但为了防止 float 精度问题，统一一下更好
            # 这里统一拉伸到 0-1 即可，不需要非得 255
            F = (F - F.min()) / (F.max() - F.min() + 1e-8)
            A = (A - A.min()) / (A.max() - A.min() + 1e-8)
            B = (B - B.min()) / (B.max() - B.min() + 1e-8)

            # --- 3. 计算差分图像 ---
            # D_FA = Fused - Source A
            diff_fa = F - A
            # D_FB = Fused - Source B
            diff_fb = F - B

            # --- 4. 计算相关系数 r ---
            # r(Diff_FA, Source B)
            r1 = pearson_correlation(diff_fa, B)

            # r(Diff_FB, Source A)
            r2 = pearson_correlation(diff_fb, A)

            # SCD = r1 + r2
            scd_list.append(r1 + r2)

    return torch.stack(scd_list).mean()

def pearson_correlation(tensor_x, tensor_y):
    """
    计算两个 2D Tensor 的皮尔逊相关系数
    """
    # 展平
    x = tensor_x.view(-1)
    y = tensor_y.view(-1)

    # 减去均值 (Center data)
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    # 计算分子: sum( (x-mean_x) * (y-mean_y) )
    numerator = torch.sum(vx * vy)

    # 计算分母: sqrt( sum(vx^2) * sum(vy^2) )
    denominator = torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2))

    # 防止除以 0
    if denominator < 1e-8:
        return torch.tensor(0.0, device=x.device)

    return numerator / denominator



from piq import vif_p

def VIF_Fusion(fused_tensor, source_a_tensor, source_b_tensor):
    """
    计算融合任务的 VIF (Visual Information Fidelity)
    基于 piq 库实现。

    逻辑: Average( VIF(Fused, A) + VIF(Fused, B) )

    参数:
        fused_tensor:    (B, C, H, W) 融合图
        source_a_tensor: (B, C, H, W) 源图 A (IR)
        source_b_tensor: (B, C, H, W) 源图 B (VIS)

    返回:
        scalar: 平均 VIF 值
    """
    # 1. 维度处理
    if fused_tensor.dim() == 2:
        fused_tensor = fused_tensor.unsqueeze(0).unsqueeze(0)
        source_a_tensor = source_a_tensor.unsqueeze(0).unsqueeze(0)
        source_b_tensor = source_b_tensor.unsqueeze(0).unsqueeze(0)
    elif fused_tensor.dim() == 3:
        fused_tensor = fused_tensor.unsqueeze(0)
        source_a_tensor = source_a_tensor.unsqueeze(0)
        source_b_tensor = source_b_tensor.unsqueeze(0)

    # 2. 鲁棒归一化 (这一步至关重要，piq 要求输入必须是 [0, 1])
    # 定义一个内部函数来处理归一化
    def normalize_to_0_1(t):
        min_v, max_v = t.min(), t.max()
        if max_v - min_v > 1e-6:
            return (t - min_v) / (max_v - min_v)
        return t  # 纯色图保持原样(会导致VIF异常，但在融合里不常见)

    # 对 Batch 中每一组图进行处理
    batch_size = fused_tensor.shape[0]
    vif_sum = 0.0

    for b in range(batch_size):
        # 取出单张图并归一化到 [0, 1]

        f_norm = normalize_to_0_1(fused_tensor[b])
        a_norm = normalize_to_0_1(source_a_tensor[b])
        b_norm = normalize_to_0_1(source_b_tensor[b])

        # piq.vif_p 需要 4D 输入 (N, C, H, W)，所以我们要 unsqueeze 回去
        f_input = f_norm.unsqueeze(0)
        a_input = a_norm.unsqueeze(0)
        b_input = b_norm.unsqueeze(0)

        # 3. 计算 VIF
        # vif_p 返回的是一个标量 tensor
        # sigma_n_sq 是噪声方差，默认 2.0 即可
        score_a = vif_p(f_input, a_input, data_range=1.0)
        score_b = vif_p(f_input, b_input, data_range=1.0)

        # 融合指标通常取平均
        vif_sum += (score_a + score_b) / 2.0

    return vif_sum / batch_size








