import torch
import torch.nn as nn
import os
os.sys.path.append('/home/honsen/honsen/SceneGraph/ourPretrained_SGG')
import src.model.depthContrast.third_party.pointnet2.pointnet2_utils as pointnet2_utils
class PointPatchDropout(nn.Module):
    """
    对点云进行局部补丁丢弃（Patch Dropout）数据增强。

    该模块会随机选择一个或多个中心点，并移除这些中心点给定半径内的所有点。
    """

    def __init__(self, dropout_ratio=0.5, num_patches=1, patch_radius=0.1):
        """
        初始化函数。

        Args:
            dropout_ratio (float): 执行此增强操作的概率。例如，0.5表示有50%的几率对输入的点云进行丢弃操作。
            num_patches (int): 在每次操作中要丢弃的补丁数量。
            patch_radius (float): 定义补丁大小的球体半径。这个值需要根据你的点云坐标范围进行调整。
                                 通常，如果点云被归一化到[-1, 1]的单位球内，0.1到0.3是常见取值。
        """
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.num_patches = num_patches
        self.patch_radius = patch_radius
        
        # pointops库的FPS函数
        self.fps = pointnet2_utils.furthest_point_sample

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        对输入的点云执行增强。

        Args:
            points (torch.Tensor): 输入的点云张量，形状应为 (B, N, C)，
                                   其中 B 是批量大小, N 是点的数量, C 是特征维度（至少为3，包含x,y,z坐标）。
                                   坐标信息应在前3个维度。

        Returns:
            torch.Tensor: 经过补丁丢弃处理后的点云。如果一个点云的某些点被丢弃，
                          它会返回一个点数少于N的张量。
                          注意：由于返回的点云数量可能不一致，这通常用于返回一个列表，
                          或者需要后续处理（如padding）才能重新组成一个batch。
                          为了简化，此实现将返回一个点数固定的张量，被丢弃的点用第一个点来填充。
        """
        # 仅在随机数小于dropout_ratio时执行增强
        if not self.training or torch.rand(1) > self.dropout_ratio:
            return points

        B, N, C = points.shape
        # 确保点云数据在GPU上，因为pointops在GPU上运行
        device = points.device

        # 仅使用XYZ坐标来计算距离
        xyz = points[:, :, :3].contiguous()

        # 1. 使用最远点采样（FPS）来选择补丁的中心点
        #    fps的输入需要 (B, N, 3)，输出是中心点的索引 (B, num_patches)
        patch_centers_idx = self.fps(xyz, self.num_patches).long()
        
        # 创建一个掩码（mask），初始化为所有点都保留 (True)
        # 形状为 (B, N)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # 2. 确定每个补丁的范围并更新掩码
        #    计算每个点到所有补丁中心的距离
        #    dist_matrix 形状: (B, N, num_patches)
        dist_matrix = torch.cdist(xyz, torch.gather(xyz, 1, patch_centers_idx.unsqueeze(-1).expand(-1, -1, 3)))
        
        # 找到每个点到最近的补丁中心的距离
        # min_dist_to_center 形状: (B, N)
        min_dist_to_center, _ = torch.min(dist_matrix, dim=2)
        
        # 3. 如果一个点到最近补丁中心的距离小于半径，则将其掩码设置为False（表示丢弃）
        mask[min_dist_to_center < self.patch_radius] = False
        
        # 4. 根据掩码过滤点云
        #    为了保持batch内张量形状一致，我们不直接删除点，而是用第一个点来替换被丢弃的点。
        #    这是一种常见的处理方式。另一种方式是返回一个列表的点云。
        
        # 创建一个结果张量
        result_points = torch.empty_like(points)
        
        for i in range(B):
            # 获取当前样本的有效点
            valid_points = points[i][mask[i]]
            
            # 如果所有点都被丢弃了（不太可能，但作为边缘情况处理）
            if valid_points.shape[0] == 0:
                # 用一个全零的点来填充
                result_points[i] = torch.zeros_like(points[i])
                continue

            # 用有效点来填充结果
            num_valid = valid_points.shape[0]
            if num_valid < N:
                # 如果点数变少，用有效点进行重复填充，直到数量达到N
                # 这种填充方式比用0填充更好，因为它保留了物体的统计特性
                filler = valid_points.repeat((N + num_valid - 1) // num_valid, 1)[:N]
                result_points[i] = filler
            else:
                result_points[i] = valid_points

        return result_points


if __name__ == '__main__':
    # --- 使用示例 ---
    # 假设我们有一个batch的点云，每个点云有1024个点，每个点有3个坐标
    batch_size = 4
    num_points = 512
    
    # 创建一个在单位球内的随机点云
    pc_data = torch.randn(batch_size, num_points, 3)
    pc_data = pc_data / torch.norm(pc_data, dim=2, keepdim=True)
    
    # 将数据移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc_data = pc_data.to(device)

    # 初始化补丁丢弃模块
    # 有50%的概率丢弃2个半径为0.2的补丁
    patch_dropout = PointPatchDropout(dropout_ratio=1, num_patches=4, patch_radius=0.3)
    patch_dropout.to(device)
    patch_dropout.train() # 设置为训练模式以激活dropout

    print(f"原始点云形状: {pc_data.shape}")

    # 应用增强
    augmented_pc = patch_dropout(pc_data)

    print(f"增强后点云形状: {augmented_pc.shape}")

    # 可视化检查（可选，需要matplotlib和open3d）
    try:
        import numpy as np
        import open3d as o3d

        # 可视化第一个点云的原始和增强后的对比
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(pc_data[0].cpu().numpy())
        original_pcd.paint_uniform_color([0, 0.7, 1]) # 蓝色

        augmented_pcd = o3d.geometry.PointCloud()
        augmented_pcd.points = o3d.utility.Vector3dVector(augmented_pc[0].cpu().numpy())
        augmented_pcd.paint_uniform_color([1, 0.7, 0]) # 橙色
        
        # 将增强后的点云向右移动一点，方便对比
        augmented_pcd.translate((2, 0, 0))

        print("\n正在显示可视化结果...")
        print("左边是原始点云（蓝色），右边是经过补丁丢弃后的点云（橙色）。")
        o3d.visualization.draw_geometries([original_pcd, augmented_pcd])

    except ImportError:
        print("\n请安装 open3d 和 numpy (`pip install open3d numpy`) 来进行可视化。")
