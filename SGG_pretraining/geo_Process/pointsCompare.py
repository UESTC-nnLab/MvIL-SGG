import open3d as o3d
import numpy as np

def numpy_to_o3d_point_cloud(points_array):
    """
    将 NumPy 数组 (N, 3) 转换为 Open3D PointCloud 对象。
    """
    if not isinstance(points_array, np.ndarray):
        print("错误: 输入的不是一个 NumPy 数组。")
        return None
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        print(f"错误: NumPy 数组的形状应为 (N, 3)，但得到的是 {points_array.shape}。")
        return None
    if points_array.shape[0] == 0:
        print("错误: NumPy 数组为空。")
        pcd = o3d.geometry.PointCloud() # 返回一个空点云对象
        return pcd


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    return pcd

def get_point_cloud_dimensions(pcd):
    """
    计算点云的轴对齐包围盒 (AABB) 和相关维度。
    返回:
        aabb (open3d.geometry.AxisAlignedBoundingBox): 包围盒对象
        volume (float): 包围盒体积
        height (float): Z轴上的高度 (max_z - min_z)
        max_z (float): Z轴最大值
        min_z (float): Z轴最小值
    """
    if pcd is None or not pcd.has_points():
        return None, 0.0, 0.0, 0.0, 0.0 # 返回默认值，避免后续操作出错
    aabb = pcd.get_axis_aligned_bounding_box()
    # 检查包围盒是否有效 (例如，对于只有一个点的点云，extent可能是[0,0,0])
    if np.any(np.isinf(aabb.get_min_bound())) or np.any(np.isinf(aabb.get_max_bound())):
        print("警告: 点云包围盒无效 (可能由于点数据问题).")
        return None, 0.0, 0.0, 0.0, 0.0

    extent = aabb.get_extent() # (width, depth, height) 即 (x_range, y_range, z_range)
    volume = extent[0] * extent[1] * extent[2]
    height = extent[2]
    max_z = aabb.get_max_bound()[2]
    min_z = aabb.get_min_bound()[2]
    return aabb, volume, height, max_z, min_z

def are_point_clouds_contacting(pcd1, pcd2, aabb1, aabb2, distance_threshold=0.01):
    """
    判断两个点云是否接触。

    方法1: 检查AABB是否相交 (快速但不完全精确)
    方法2: 计算点云间的最小距离 (更精确但计算量大)
    """
    if pcd1 is None or not pcd1.has_points() or \
       pcd2 is None or not pcd2.has_points() or \
       aabb1 is None or aabb2 is None:
        return False, "数据不完整或点云为空"

    # 检查两个AABB是否有重叠区域
    min_bound1 = aabb1.get_min_bound()
    max_bound1 = aabb1.get_max_bound()
    min_bound2 = aabb2.get_min_bound()
    max_bound2 = aabb2.get_max_bound()

    # 检查是否有重叠 (确保min < max)
    overlap_x = max(0, min(max_bound1[0], max_bound2[0]) - max(min_bound1[0], min_bound2[0]))
    overlap_y = max(0, min(max_bound1[1], max_bound2[1]) - max(min_bound1[1], min_bound2[1]))
    overlap_z = max(0, min(max_bound1[2], max_bound2[2]) - max(min_bound1[2], min_bound2[2]))

    aabb_truly_intersects = overlap_x > 1e-9 and overlap_y > 1e-9 and overlap_z > 1e-9 # 使用一个小的epsilon避免浮点数精度问题

    if aabb_truly_intersects:
        # print("包围盒相交，可能接触。进行点间距离检查...")

        # 计算pcd1到pcd2的距离
        # 使用 KDTree 进行更高效的最近邻搜索
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
        min_dist_p1_to_p2 = float('inf')
        for point in np.asarray(pcd1.points):
            [k, idx, dist_sq] = pcd2_tree.search_knn_vector_3d(point, 1)
            if k > 0: # 确保找到了点
                 min_dist_p1_to_p2 = min(min_dist_p1_to_p2, np.sqrt(dist_sq[0]))


        pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
        min_dist_p2_to_p1 = float('inf')
        for point in np.asarray(pcd2.points):
            [k, idx, dist_sq] = pcd1_tree.search_knn_vector_3d(point, 1)
            if k > 0:
                min_dist_p2_to_p1 = min(min_dist_p2_to_p1, np.sqrt(dist_sq[0]))

        min_overall_distance = min(min_dist_p1_to_p2, min_dist_p2_to_p1)

        if min_overall_distance < distance_threshold:
            return True, f"点云接触 (最小距离: {min_overall_distance:.4f} < 阈值: {distance_threshold})"
        else:
            return True, f"包围盒相交但点云间最小距离 ({min_overall_distance:.4f}) >= 阈值 ({distance_threshold})"
    else:
        return False, "包围盒不相交"


def compare_point_clouds_from_numpy(points_array1, points_array2, flags, contact_distance_threshold=0.001):
    """
    比较两个以NumPy数组形式存储的点云的大小、高度和是否接触。
    """
    pcd1 = numpy_to_o3d_point_cloud(points_array1)
    pcd2 = numpy_to_o3d_point_cloud(points_array2)

    if pcd1 is None or pcd2 is None:
        print("无法比较点云，因为至少有一个NumPy数组转换失败或无效。")
        return

    aabb1, volume1, height1, max_z1, min_z1 = get_point_cloud_dimensions(pcd1)
    num_points1 = len(pcd1.points) # 如果pcd1是空点云，len(pcd1.points)也是0
    # if aabb1: # 只有当aabb1有效时才打印
    #     print(f"点数: {num_points1}")
    #     print(f"包围盒体积 (近似大小): {volume1:.4f}")
    #     print(f"Z轴高度: {height1:.4f} (从 {min_z1:.4f} 到 {max_z1:.4f})")
    #     print(f"Z轴最高点: {max_z1:.4f}")
    # elif num_points1 == 0 :
    #     print("点云1为空。")
    # else:
    #     print("无法计算点云1的维度属性 (可能是由于包围盒无效)。")


    aabb2, volume2, height2, max_z2, min_z2 = get_point_cloud_dimensions(pcd2)
    num_points2 = len(pcd2.points)
    # if aabb2:
    #     print(f"点数: {num_points2}")
    #     print(f"包围盒体积 (近似大小): {volume2:.4f}")
    #     print(f"Z轴高度: {height2:.4f} (从 {min_z2:.4f} 到 {max_z2:.4f})")
    #     print(f"Z轴最高点: {max_z2:.4f}")
    # elif num_points2 == 0:
    #     print("点云2为空。")
    # else:
    #     print("无法计算点云2的维度属性 (可能是由于包围盒无效)。")


    # 只有在两个点云都有有效属性时才进行比较
    can_compare_dimensions = aabb1 is not None and aabb2 is not None and num_points1 > 0 and num_points2 > 0

    if not can_compare_dimensions:
        print("由于一个或两个点云为空或属性计算失败，无法进行完整的比较。")
        # 仍然可以尝试接触检测，如果点云本身有效的话
    else:
        # 1. 哪个更大 (基于包围盒体积)
        if volume1 > volume2:
            flags[0]=1

        # 2. 哪个更高 (基于Z轴最大值)
        if max_z1 > max_z2:
            flags[1] = 1

    # 3. 是否有接触 (即使一个点云为空，也应能处理，are_point_clouds_contacting有检查)
    contact_status, contact_reason = are_point_clouds_contacting(pcd1, pcd2, aabb1, aabb2, distance_threshold=contact_distance_threshold)
    if contact_status:
        flags[2] = 1


    # 可视化 (可选)
    # if pcd1.has_points() and pcd2.has_points() and aabb1 and aabb2:
    #     pcd1.paint_uniform_color([0.8, 0.2, 0.2]) # 给点云上色
    #     pcd2.paint_uniform_color([0.2, 0.2, 0.8])
    #     aabb1_vis = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb1)
    #     aabb1_vis.paint_uniform_color([1, 0, 0]) # 红色
    #     aabb2_vis = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb2)
    #     aabb2_vis.paint_uniform_color([0, 1, 0]) # 绿色
    #     o3d.visualization.draw_geometries([pcd1, pcd2, aabb1_vis, aabb2_vis])
    # elif pcd1.has_points() and pcd2.has_points(): # 如果包围盒无效但点云存在
    #     pcd1.paint_uniform_color([0.8, 0.2, 0.2])
    #     pcd2.paint_uniform_color([0.2, 0.2, 0.8])
    #     o3d.visualization.draw_geometries([pcd1, pcd2])
    return flags

# --- 主程序 ---
if __name__ == "__main__":
    print("--- 示例 1: 基本分离的点云 ---")
    # 创建第一个 NumPy 点云 (例如，一个立方体)
    points_np1 = np.random.rand(100, 3) * np.array([1, 1, 0.5]) # 扁平一点的立方体
    # 创建第二个 NumPy 点云 (例如，一个向上平移且在Z轴更高的立方体)
    points_np2 = np.random.rand(150, 3) * np.array([0.8, 0.8, 1.2]) + np.array([1.5, 0.5, 1.0]) # 平移并使其Z轴更高

    compare_point_clouds_from_numpy(points_np1, points_np2, contact_distance_threshold=0.1)

    print("\n\n--- 示例 2: 接触的点云 ---")
    points_np_contact1 = np.array([
        [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0.5,0.5,0.5] # 中心点
    ])
    points_np_contact2 = np.array([
        [0.45,0.45,0.45], # 这个点与points_np_contact1的中心点接近
        [1.5,0.5,0.5],
        [0.5,1.5,0.5],
        [1.5,1.5,0.5]
    ])
    compare_point_clouds_from_numpy(points_np_contact1, points_np_contact2, contact_distance_threshold=0.1) # 阈值设大一点更容易检测到接触

    print("\n\n--- 示例 3: 一个空点云，一个有点的点云 ---")
    points_np_empty = np.empty((0,3))
    points_np_valid = np.random.rand(50,3)
    compare_point_clouds_from_numpy(points_np_empty, points_np_valid)

    print("\n\n--- 示例 4: 两个都为空的点云 ---")
    compare_point_clouds_from_numpy(points_np_empty, np.empty((0,3)))

    print("\n\n--- 示例 5: 点云只有一个点 (包围盒体积为0) ---")
    points_single1 = np.array([[1.0, 2.0, 3.0]])
    points_single2 = np.array([[1.0, 2.0, 3.5]]) # Z轴更高
    compare_point_clouds_from_numpy(points_single1, points_single2)

    print("\n\n--- 示例 6: 包围盒相交但不接触的点云 (需要调整阈值来观察) ---")
    # pcd A: 一个在原点的立方体
    cloud_A_points = np.array([
        [0,0,0],[1,0,0],[0,1,0],[1,1,0],
        [0,0,1],[1,0,1],[0,1,1],[1,1,1]
    ])
    # pcd B: 一个在旁边但包围盒有轻微重叠的立方体, 但点之间没有接触
    cloud_B_points = cloud_A_points.copy() + np.array([0.8, 0.8, 0]) # X, Y方向靠近，包围盒会重叠

    # 默认阈值较小，可能不会认为接触
    print("测试小阈值:")
    compare_point_clouds_from_numpy(cloud_A_points, cloud_B_points, contact_distance_threshold=0.05)
    # 增大阈值，可能会因为包围盒相交而被认为接触（取决于具体实现）
    # 注意: are_point_clouds_contacting 现在会先检查包围盒，然后检查点距离
    # 如果包围盒相交，即使点距离大于阈值，也会报告 "包围盒相交但点云间最小距离 ... >= 阈值"
    print("\n测试较大阈值 (不一定会改变接触结果，但会改变报告):")
    compare_point_clouds_from_numpy(cloud_A_points, cloud_B_points, contact_distance_threshold=0.5)