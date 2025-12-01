import numpy as np
from scipy.interpolate import RegularGridInterpolator

def random_translate(points, translate_range=1.5):
    """
    对点云进行随机平移。

    参数:
    - points (np.ndarray): 输入的点云，形状为 (N, 3)。
    - translate_range (float): 每个轴的平移范围的最大值。平移量将在 [-range, range] 之间均匀采样。

    返回:
    - np.ndarray: 平移后的点云。
    """
    translation = np.random.uniform(-translate_range, translate_range, size=(2,))
    points[:, 0:2] = points[:, 0:2] + translation
    return points

def random_scale(points, scale_low=0.7, scale_high=1.5):
    """
    对点云进行随机缩放，缩放中心为点云的质心。

    参数:
    - points (np.ndarray): 输入的点云，形状为 (N, 3)。
    - scale_low (float): 缩放因子的下限。
    - scale_high (float): 缩放因子的上限。

    返回:
    - np.ndarray: 缩放后的点云。
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # 使用各向同性缩放（所有轴使用相同的缩放因子）
    scale = np.random.uniform(scale_low, scale_high)
    points_scaled = points_centered * scale
    
    return points_scaled + centroid

def random_jitter(points, sigma=0.022, clip=0.05):
    """
    对点云中的每个点添加随机扰动（高斯噪声）。

    参数:
    - points (np.ndarray): 输入的点云，形状为 (N, 3)。
    - sigma (float): 高斯噪声的标准差。
    - clip (float): 噪声值的裁剪上限，防止出现过大的离群点。

    返回:
    - np.ndarray: 添加扰动后的点云。
    """
    assert(clip > 0)
    noise = np.random.randn(*points.shape) * sigma
    # 将噪声限制在 [-clip, clip] 范围内
    noise_clipped = np.clip(noise, -clip, clip)
    return points + noise_clipped

def random_dropout(points,instances, dropout_ratio=0.5):
    """
    随机丢弃/下采样点云中的一部分点。

    参数:
    - points (np.ndarray): 输入的点云，形状为 (N, 3)。
    - dropout_ratio (float): 要丢弃的点的比例，范围在 [0, 1) 之间。

    返回:
    - np.ndarray: 下采样后的点云。
    """
    if dropout_ratio <= 0 or dropout_ratio >= 1:
        return points
        
    num_points = points.shape[0]
    num_to_keep = int(num_points * (1 - dropout_ratio))
    
    # 生成随机索引并选择要保留的点
    indices = np.random.choice(num_points, num_to_keep, replace=False)
    return points[indices, :], instances[indices]

def elastic_distortion(points, granularity, magnitude):
    """
    对点云进行非刚性(弹性)变形。
    该方法在一个粗糙的3D网格上生成随机位移，然后通过三线性插值计算每个点的位移。

    参数:
    - points (np.ndarray): 输入的点云，形状为 (N, 3)。
    - granularity (list/tuple): 3个整数，表示3D网格在x,y,z轴上的分段数。例如 (3, 3, 3)
    - magnitude (list/tuple): 3个浮点数，表示在x,y,z轴上随机位移的最大幅度。

    返回:
    - np.ndarray: 变形后的点云。
    """
    # 确定点云的边界
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    if min_coords[0]==max_coords[0] or min_coords[1]==max_coords[1] or min_coords[2]==max_coords[2]:
        return points

    # 创建粗糙网格的坐标轴
    grid_x = np.linspace(min_coords[0], max_coords[0], granularity[0])
    grid_y = np.linspace(min_coords[1], max_coords[1], granularity[1])
    grid_z = np.linspace(min_coords[2], max_coords[2], granularity[2])
    
    # 在网格节点上生成随机位移
    noise = np.random.randn(*granularity, 3) * np.array(magnitude)
    
    # 创建插值器
    # RegularGridInterpolator 需要网格节点坐标和对应的值
    interp_x = RegularGridInterpolator((grid_x, grid_y, grid_z), noise[..., 0], bounds_error=False, fill_value=0)
    interp_y = RegularGridInterpolator((grid_x, grid_y, grid_z), noise[..., 1], bounds_error=False, fill_value=0)
    interp_z = RegularGridInterpolator((grid_x, grid_y, grid_z), noise[..., 2], bounds_error=False, fill_value=0)
    
    # 对每个输入点进行插值，得到其位移
    displacement = np.zeros_like(points)
    displacement[:, 0] = interp_x(points)
    displacement[:, 1] = interp_y(points)
    displacement[:, 2] = interp_z(points)
    
    return points + displacement


def augment_point_cloud_union(points, scale_range=(0.8, 1.2), translate_range=0.1, jitter_sigma=0.01, jitter_clip=0.05, dropout_ratio=0.2, elastic_params=None):
    """
    一个完整的点云增广流程，按顺序应用各种增广。
    
    参数:
    - points (np.ndarray): 两个物体点云的并集，形状为 (N, 3)。
    - scale_range (tuple): 随机缩放的范围 (min, max)。
    - translate_range (float): 随机平移的最大范围。
    - jitter_sigma (float): 随机扰动的标准差。
    - jitter_clip (float): 随机扰动的裁剪值。
    - dropout_ratio (float): 随机丢弃的点的比例。
    - elastic_params (dict): 弹性变形的参数, e.g., {'granularity': (3, 3, 3), 'magnitude': (0.05, 0.05, 0.05)}。
                             如果为 None, 则不应用弹性变形。

    返回:
    - np.ndarray: 经过一系列增广后的点云。
    """
    augmented_points = points.copy()
    
    # 1. 非刚性变形 (通常作为第一步，作用于原始形状)
    if elastic_params:
        augmented_points = elastic_distortion(
            augmented_points, 
            elastic_params['granularity'], 
            elastic_params['magnitude']
        )
        
    # 2. 随机缩放
    augmented_points = random_scale(augmented_points, scale_low=scale_range[0], scale_high=scale_range[1])
    
    # 3. 随机扰动 (Jitter)
    augmented_points = random_jitter(augmented_points, sigma=jitter_sigma, clip=jitter_clip)
    
    # 4. 随机平移
    augmented_points = random_translate(augmented_points, translate_range=translate_range)
    
    # 5. 随机点丢弃/下采样 (通常作为最后一步)
    augmented_points = random_dropout(augmented_points, dropout_ratio=dropout_ratio)
    
    return augmented_points

# ==========================================================
# ====================== 使用示例 ==========================
# ==========================================================
if __name__ == '__main__':
    # 1. 创建一个模拟的点云并集
    # 假设物体1是一个立方体，物体2是一个在它旁边的球体
    
    # 物体1: 立方体
    cube_points = np.random.rand(500, 3) - 0.5  # 在 [-0.5, 0.5] 范围内
    
    # 物体2: 球体
    num_sphere_points = 500
    phi = np.random.uniform(0, np.pi, num_sphere_points)
    theta = np.random.uniform(0, 2 * np.pi, num_sphere_points)
    radius = 0.3
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    sphere_points = np.vstack([x, y, z]).T
    sphere_points += np.array([1.0, 0, 0]) # 将球体平移到立方体旁边
    
    # 将两个物体的点云合并成一个
    point_cloud_union = np.vstack([cube_points, sphere_points])
    
    print(f"原始点云并集包含 {point_cloud_union.shape[0]} 个点。")
    
    # 2. 定义弹性变形参数
    elastic_config = {
        'granularity': (4, 4, 4), # 在x,y,z轴上都分成4段
        'magnitude': (0.2, 0.2, 0.2)   # 各轴位移幅度
    }
    
    # 3. 应用完整的增广流程
    augmented_points = augment_point_cloud_union(
        point_cloud_union,
        scale_range=(0.8, 1.25),
        translate_range=0.2,
        jitter_sigma=0.01,
        jitter_clip=0.05,
        dropout_ratio=0.25,
        elastic_params=elastic_config
    )
    
    print(f"增广后的点云包含 {augmented_points.shape[0]} 个点。")
    
    # 4. 可视化 (可选)
    # 如果您安装了 open3d, 可以使用以下代码进行3D可视化
    try:
        import open3d as o3d
        
        # 原始点云
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(point_cloud_union)
        pcd_original.paint_uniform_color([0, 0.651, 0.929]) # 蓝色
        
        # 增广后的点云 (为了方便对比，我们把它平移一下)
        augmented_points_viz = augmented_points + np.array([3, 0, 0])
        pcd_augmented = o3d.geometry.PointCloud()
        pcd_augmented.points = o3d.utility.Vector3dVector(augmented_points_viz)
        pcd_augmented.paint_uniform_color([1, 0.706, 0]) # 橙色
        
        print("\n正在显示点云... 左边为原始点云，右边为增广后的点云。")
        o3d.visualization.draw_geometries([pcd_original, pcd_augmented])
        
    except ImportError:
        print("\n建议安装 'open3d' (pip install open3d) 来进行3D点云可视化。")
        # 使用 matplotlib 进行2D投影可视化
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 6))
        
        # 原始点云 XY 投影
        ax1 = fig.add_subplot(121)
        ax1.scatter(point_cloud_union[:, 0], point_cloud_union[:, 1], s=1)
        ax1.set_title("Original Points (XY Projection)")
        ax1.set_aspect('equal', 'box')
        
        # 增广后点云 XY 投影
        ax2 = fig.add_subplot(122)
        ax2.scatter(augmented_points[:, 0], augmented_points[:, 1], s=1)
        ax2.set_title("Augmented Points (XY Projection)")
        ax2.set_aspect('equal', 'box')
        
        plt.show()