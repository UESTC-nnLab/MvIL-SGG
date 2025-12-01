import open3d as o3d
import numpy as np
import copy

def create_pcd_from_numpy(points_np):
    """从 NumPy 数组创建 Open3D PointCloud 对象"""
    if not isinstance(points_np, np.ndarray) or points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError("输入必须是 Nx3 的 NumPy 数组")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    if not pcd.has_points():
        raise ValueError("无法从 NumPy 数组创建点云或数组为空")
    return pcd

def estimate_initial_scale(source_pcd, target_pcd):
    """基于包围盒对角线长度估计初始尺度因子"""
    # 计算包围盒
    aabb_source = source_pcd.get_axis_aligned_bounding_box()
    aabb_target = target_pcd.get_axis_aligned_bounding_box()

    # 计算对角线长度的近似值（或者使用 extent）
    diag_source = np.linalg.norm(aabb_source.get_extent())
    diag_target = np.linalg.norm(aabb_target.get_extent())

    if diag_source < 1e-6:
        raise ValueError("源点云尺寸过小，无法估计尺度")

    # 尺度因子 = 目标尺寸 / 源尺寸
    scale = diag_target / diag_source
    print(f":: 估计的初始尺度因子 (Target/Source): {scale:.4f}")
    return scale

    # --- 备选方法：基于平均最近邻距离 ---
    # kdtree_source = o3d.geometry.KDTreeFlann(source_pcd)
    # dists_source = []
    # for i in range(len(source_pcd.points)):
    #     [k, idx, _] = kdtree_source.search_knn_vector_3d(source_pcd.points[i], 2)
    #     if k == 2:
    #         dists_source.append(np.linalg.norm(source_pcd.points[i] - source_pcd.points[idx[1]]))
    # mean_dist_source = np.mean(dists_source) if dists_source else 1.0

    # kdtree_target = o3d.geometry.KDTreeFlann(target_pcd)
    # dists_target = []
    # for i in range(len(target_pcd.points)):
    #     [k, idx, _] = kdtree_target.search_knn_vector_3d(target_pcd.points[i], 2)
    #     if k == 2:
    #         dists_target.append(np.linalg.norm(target_pcd.points[i] - target_pcd.points[idx[1]]))
    # mean_dist_target = np.mean(dists_target) if dists_target else 1.0

    # if mean_dist_source < 1e-6:
    #      raise ValueError("源点云平均距离过小，无法估计尺度")
    # scale = mean_dist_target / mean_dist_source
    # print(f":: Estimated initial scale factor (Target/Source) based on NN distance: {scale:.4f}")
    # return scale
    # ------------------------------------

def preprocess_point_cloud(pcd, voxel_size):
    """下采样并计算法线和FPFH特征 (与之前相同)"""
    print(":: 体素下采样...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"下采样后点数: {len(pcd_down.points)}")

    radius_normal = voxel_size * 2
    print(f":: 估计法线，搜索半径 = {radius_normal}")
    # 增加检查点数，避免点数过少导致错误
    if len(pcd_down.points) < 3:
         raise ValueError("下采样后点数过少，无法估计法线")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=min(30, len(pcd_down.points)-1))
    )

    radius_feature = voxel_size * 5
    print(f":: 计算FPFH特征，搜索半径 = {radius_feature}")
    # 增加检查点数
    if len(pcd_down.points) < 3:
        raise ValueError("下采样后点数过少，无法计算FPFH特征")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=min(100, len(pcd_down.points)-1))
    )
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    """执行基于特征的全局配准 (RANSAC) (与之前相同)"""
    distance_threshold = voxel_size * 1.5 # RANSAC 距离阈值
    print(":: RANSAC 全局配准")
    print(f"   距离阈值 = {distance_threshold}")

    # 检查特征是否有效
    if source_fpfh is None or target_fpfh is None or source_fpfh.data.shape[1] == 0 or target_fpfh.data.shape[1] == 0:
        raise ValueError("FPFH 特征计算失败或为空")
    if len(source_down.points) < 3 or len(target_down.points) < 3:
        raise ValueError("全局配准需要至少3个点")


    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, # RANSAC 迭代次数的参数 (ransac_n)
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

def refine_registration(source, target, initial_transformation, voxel_size):
    """使用 ICP 进行精细配准 (与之前相同, 但作用于缩放后的源和原始目标)"""
    distance_threshold = voxel_size * 0.4 # ICP 距离阈值
    print(":: ICP 精细配准")
    print(f"   距离阈值 = {distance_threshold}")

    # 确保目标点云有法线
    if not target.has_normals():
         radius_normal = voxel_size * 2
         # 检查点数
         if len(target.points) < 3:
              raise ValueError("目标点云点数过少，无法估计法线")
         target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=min(30, len(target.points)-1)))

    # 检查点数
    if len(source.points) < 3 or len(target.points) < 3:
         raise ValueError("ICP 需要至少3个点")

    registration_result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
    return registration_result

def visualize_registration(source_orig_np, target_np, scale=1.0, transformation=np.identity(4)):
    """可视化配准前后的点云 (处理NumPy和缩放)"""
    source_vis = create_pcd_from_numpy(source_orig_np * scale) # 应用缩放
    target_vis = create_pcd_from_numpy(target_np)

    source_vis.paint_uniform_color([1, 0.706, 0])  # 源点云: 黄色
    target_vis.paint_uniform_color([0, 0.651, 0.929]) # 目标点云: 蓝色

    source_vis.transform(transformation) # 应用刚性变换

    o3d.visualization.draw_geometries([source_vis, target_vis],
                                      window_name="点云配准可视化")

def register(source_np_orig,target_np,isvisual=False):
    # 1. 创建 Open3D 点云对象
    source_pcd_orig = create_pcd_from_numpy(source_np_orig)
    target_pcd = create_pcd_from_numpy(target_np)

    # 2. 估计初始尺度
    estimated_scale = estimate_initial_scale(source_pcd_orig, target_pcd)

    # 3. 缩放源点云 (NumPy 层面操作)
    source_np_scaled = source_np_orig * estimated_scale
    source_pcd_scaled = create_pcd_from_numpy(source_np_scaled)  # 创建缩放后的 O3D 点云

    if isvisual:
        # 可视化初始对齐情况（源已缩放，但未旋转/平移）
        print("显示初始对齐情况 (源已按估计比例缩放)...")
        visualize_registration(source_np_orig, target_np, scale=estimated_scale)

    # 4. 设置刚性配准参数
    # !!! 重要：voxel_size 现在应该基于目标点云 B 或缩放后的 A 的尺度来确定 !!!
    # 使用目标点云的包围盒计算 voxel_size
    target_extent = target_pcd.get_axis_aligned_bounding_box().get_extent()
    # 检查 target_extent 是否有效
    if np.any(np.isnan(target_extent)) or np.any(target_extent <= 0):
        print("警告: 目标点云范围无效，使用默认 voxel_size = 0.05")
        voxel_size = 0.05
    else:
        diag_len_target = np.linalg.norm(target_extent)
        if diag_len_target < 1e-6:
            print("警告: 目标点云尺寸过小，使用默认 voxel_size = 0.05")
            voxel_size = 0.05
        else:
            voxel_size = diag_len_target / 30  # 调整分母 20 到 50 之间
            print(f"根据目标点云尺寸自动计算 voxel_size = {voxel_size:.4f}")

    # --- 如果自动计算不理想，请手动设置 voxel_size ---
    # voxel_size = 0.05 # 手动设置

    try:
        # 5. 预处理缩放后的源和目标，并进行全局配准
        source_down, source_fpfh = preprocess_point_cloud(source_pcd_scaled, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

        # 执行全局配准 (在缩放后的 A 和 B 之间)
        global_result = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)

        print("全局配准结果 (针对缩放后的源):")
        print(global_result)
        print("初始刚性变换矩阵 (全局配准):")
        print(global_result.transformation)

        if isvisual:
            # 可视化全局配准后的对齐结果 (作用于下采样点云)
            print("显示全局配准后的对齐情况...")
            # 注意：可视化时仍用原始未缩放的 source_np_orig, 但传入 scale 和 刚性变换
            visualize_registration(source_np_orig, target_np,
                                   scale=estimated_scale,
                                   transformation=global_result.transformation)

        # 6. 局部精细配准 (ICP)
        # 使用全局配准结果作为ICP的初始变换
        # ICP 作用于 缩放后的原始点云 (source_pcd_scaled) 和 原始目标点云 (target_pcd)
        print("对高分辨率点云进行精细配准...")
        final_rigid_result = refine_registration(source_pcd_scaled, target_pcd,
                                                 global_result.transformation,
                                                 voxel_size)

        print("精细配准 (ICP) 结果:")
        print(final_rigid_result)
        print("最终刚性变换矩阵 (ICP, 作用于缩放后的源):")
        print(final_rigid_result.transformation)

        # 7. 输出最终结果
        final_scale = estimated_scale
        final_rigid_transformation = final_rigid_result.transformation  # 4x4 矩阵

        print("\n---最终配准结果---")
        print(f"估计的尺度因子 s: {final_scale:.6f}")
        print("估计的刚性变换 T (应用于 s * A):")
        print(final_rigid_transformation)
        print(f"评估指标 (基于缩放后的源和目标):")
        print(f"  Fitness: {final_rigid_result.fitness:.4f}")
        print(f"  Inlier RMSE: {final_rigid_result.inlier_rmse:.4f}")

        if isvisual:
            # 8. 可视化最终结果
            print("显示最终精细配准后的对齐情况...")
            visualize_registration(source_np_orig, target_np,
                                   scale=final_scale,
                                   transformation=final_rigid_transformation)

        # 9. 如何应用最终变换到原始点云 A
        # source_np_final_aligned = (source_np_orig * final_scale) @ final_rigid_transformation[:3, :3].T + final_rigid_transformation[:3, 3]
        # 或者使用 Open3D:
        source_final_aligned_pcd = create_pcd_from_numpy(source_np_orig)  # 从原始A创建
        source_final_aligned_pcd.scale(final_scale, center=np.array([0, 0, 0]))  # 应用缩放
        source_final_aligned_pcd.transform(final_rigid_transformation)  # 应用刚性变换

        return final_rigid_result.transformation
        # (可选) 保存对齐后的源点云
        # o3d.io.write_point_cloud("cloud_A_aligned_scaled.ply", source_final_aligned_pcd)

    except ValueError as e:
        print(f"\n配准过程中发生错误: {e}")
        print("请检查点云数据和 voxel_size 参数。")
    except Exception as e:
        print(f"\n发生意外错误: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 假设你的点云数据已经加载到 NumPy 数组中
    # 这里用随机数据模拟，请替换为你的真实数据
    # --- 请替换为你的 NumPy 点云数据 ---
    print("生成模拟数据 (请替换为您的真实 NumPy 数据)...")
    # 模拟源点云 A (例如一个立方体)
    points_A_orig = np.random.rand(500, 3) * 0.8 # 原始 A
    # 模拟目标点云 B (A 经过缩放、旋转、平移)
    true_scale = 1.5
    angle = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    translation_vector = np.array([0.5, -0.2, 0.3])
    points_B = (points_A_orig * true_scale) @ rotation_matrix.T + translation_vector
    # 添加一些噪声
    points_A_noisy = points_A_orig + np.random.normal(0, 0.01, points_A_orig.shape)
    points_B_noisy = points_B + np.random.normal(0, 0.01, points_B.shape)
    # -------------------------------------

    # 使用你的 NumPy 数组
    source_np_orig = points_A_noisy # 原始未缩放的源 NumPy 数组
    target_np = points_B_noisy     # 目标 NumPy 数组

    register(source_np_orig,target_np)