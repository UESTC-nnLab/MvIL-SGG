import torch

def convert_pt_input_to_bcn_vectorized(pt_input):
    """
    【向量化实现】将 Point Transformer 的输入格式转换回 (B, C, N)。

    Args:
        pt_input (dict): 包含 'feat' 和 'offset' 的字典。
            - feat (torch.Tensor): (M, C) 的特征张量。
            - offset (torch.Tensor): (B,) 的偏移量张量。

    Returns:
        torch.Tensor: 重建后的 (B, C, N) 格式的特征张量。
    """
    feat = pt_input["feat"]
    offset = pt_input["offset"].cuda()
    
    # 验证批次中的点云大小是否一致
    point_counts = offset.diff(prepend=torch.tensor([0], device=offset.device))
    if not torch.all(point_counts == point_counts[0]):
        raise ValueError("无法转换，因为批次中的点云大小不一致。")
        
    # 使用 torch.tensor_split 根据偏移量将压平的特征张量分割成列表
    # offset[:-1] 提供了分割点
    # M, C -> list of B tensors, each of shape (N, C)
    list_of_nc_tensors = torch.tensor_split(feat, offset[:-1].long())
    
    # 将列表中的 (N, C) 张量堆叠成 (B, N, C)
    stacked_bnc = torch.stack(list_of_nc_tensors, dim=0)
    
    # 将维度从 (B, N, C) 转换到 (B, C, N)
    return stacked_bnc.permute(0, 2, 1)
# (这是上一节的转换函数，我们在这里再次使用它进行测试)
def convert_bcn_to_pt_input(features_bcn):
    if features_bcn.dim() != 3:
        raise ValueError("输入张量的形状必须是 (B, C, N)")
    B, C, N = features_bcn.shape
    device = features_bcn.device
    
    # 假设前3个通道是坐标
    coord_b3n = features_bcn[:, :3, :]
    coord_bn3 = coord_b3n.permute(0, 2, 1)
    coord = coord_bn3.reshape(-1, 3)

    # (B, C, N) -> (B, N, C) -> (B*N, C)
    feat_bnc = features_bcn.permute(0, 2, 1)
    feat = feat_bnc.reshape(-1, C)
    
    offset = torch.arange(1, B + 1, device=device).int() * N
    return {"coord": coord, "feat": feat, "offset": offset}


if __name__ == '__main__':
    # --- 参数设置 ---
    batch_size = 1024      # B
    num_features = 512  # C
    num_points = 128   # N

    # --- 1. 创建原始输入 ---
    # 创建一个随机特征张量 (B, C, N)
    original_features = torch.randn(batch_size, num_features, num_points)
    print(f"原始输入形状 (B, C, N): {original_features.shape}")

    # --- 2. 正向转换: (B, C, N) -> Point Transformer 格式 ---
    pt_input_data = convert_bcn_to_pt_input(original_features)
    print("\n--- 转换为 Point Transformer 格式 ---")
    print(f"坐标 'coord' 形状: {pt_input_data['coord'].shape}")
    print(f"特征 'feat' 形状: {pt_input_data['feat'].shape}")
    print(f"偏移量 'offset' 形状: {pt_input_data['offset'].shape}")

    # --- 3. 逆向转换: Point Transformer 格式 -> (B, C, N) ---
    # 我们使用更高效的向量化版本
    reconstructed_features = convert_pt_input_to_bcn_vectorized(pt_input_data)
    print("\n--- 转换回 (B, C, N) 格式 ---")
    print(f"重建后的特征形状: {reconstructed_features.shape}")

    # --- 4. 验证 ---
    # 检查重建的张量是否与原始张量几乎完全相等
    is_correct = torch.allclose(original_features, reconstructed_features)
    
    print("\n--- 验证结果 ---")
    if is_correct:
        print("✅ 验证成功！重建的张量与原始张量完全一致。")
    else:
        print("❌ 验证失败！重建的张量与原始张量不符。")

