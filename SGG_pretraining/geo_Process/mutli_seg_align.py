import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple, Any

def get_camera_center(pose: np.ndarray) -> np.ndarray:
    """从 4x4 位姿矩阵提取相机中心 (世界坐标系下)."""
    # C = -R^T * t
    rotation_matrix = pose[:3, :3]
    translation_vector = pose[:3, 3]
    center = -rotation_matrix.T @ translation_vector
    return center

def estimate_similarity_transform(poses1: Dict[Any, np.ndarray],
                                  poses2: Dict[Any, np.ndarray],
                                  common_frame_ids: List[Any]) -> np.ndarray:
    """
    根据重叠帧的位姿估计从坐标系2到坐标系1的相似变换矩阵 (s, R, t).
    T_2_to_1: P1 = T_2_to_1 * P2

    Args:
        poses1: 第一个坐标系中的位姿字典 {frame_id: 4x4 pose matrix}.
        poses2: 第二个坐标系中的位姿字典 {frame_id: 4x4 pose matrix}.
        common_frame_ids: 重叠帧的 ID 列表.

    Returns:
        4x4 相似变换矩阵 (从坐标系2变换到坐标系1), 或者 None 如果无法计算.
    """
    if len(common_frame_ids) < 2:
        print("警告: 至少需要2个重叠帧来估计尺度。")
        # 如果只有1帧，可以尝试计算刚性变换（假设尺度为1）
        if len(common_frame_ids) == 1:
            frame_id = common_frame_ids[0]
            pose1 = poses1[frame_id]
            pose2 = poses2[frame_id]
            T_rigid = pose1 @ np.linalg.inv(pose2)
            print("警告: 仅使用1个重叠帧计算刚性变换 (尺度=1)。")
            return T_rigid
        else:
            print("错误: 没有足够的重叠帧来计算变换。")
            return None

    # --- 1. 估计尺度因子 s ---
    scales = []
    centers1 = {fid: get_camera_center(poses1[fid]) for fid in common_frame_ids}
    centers2 = {fid: get_camera_center(poses2[fid]) for fid in common_frame_ids}

    frame_ids_list = list(common_frame_ids) # 确保有顺序
    for i in range(len(frame_ids_list)):
        for j in range(i + 1, len(frame_ids_list)):
            fid_i = frame_ids_list[i]
            fid_j = frame_ids_list[j]

            dist1 = np.linalg.norm(centers1[fid_i] - centers1[fid_j])
            dist2 = np.linalg.norm(centers2[fid_i] - centers2[fid_j])

            if dist2 > 1e-6: # 避免除以零
                scales.append(dist1 / dist2)

    if not scales:
        print("错误: 无法估计尺度 (重叠帧的相机中心距离可能为零)。")
        # 尝试回退到刚性变换？
        frame_id = common_frame_ids[0]
        pose1 = poses1[frame_id]
        pose2 = poses2[frame_id]
        T_rigid = pose1 @ np.linalg.inv(pose2)
        print("警告: 无法估计尺度，回退到刚性变换 (尺度=1)。")
        return T_rigid
        # return None

    # 使用中位数或平均值来获得更鲁棒的尺度估计
    estimated_scale = np.median(scales) # 中位数更抗异常值
    print(f"估计的相对尺度 s (coord1 / coord2): {estimated_scale:.4f}")
    estimated_scale = 1

    # --- 2. 估计旋转 R ---
    # R1 = R * R2  => R = R1 * R2^T
    rotation_matrices_target = []
    for fid in common_frame_ids:
        R1 = poses1[fid][:3, :3]
        R2 = poses2[fid][:3, :3]
        rotation_matrices_target.append(R1 @ R2.T)

    # 平均旋转 (使用 SVD 方法)
    # H = sum(R_target_i)
    H = np.sum(rotation_matrices_target, axis=0)
    U, _, Vt = np.linalg.svd(H)
    estimated_rotation = U @ Vt

    # 确保是右手坐标系
    if np.linalg.det(estimated_rotation) < 0:
       Vt[-1, :] *= -1
       estimated_rotation = U @ Vt

    # --- 3. 估计平移 t ---
    # t1 = s * R * t2 + t => t = t1 - s * R * t2
    translations_target = []
    for fid in common_frame_ids:
        t1 = poses1[fid][:3, 3]
        t2 = poses2[fid][:3, 3]
        translations_target.append(t1 - estimated_scale * estimated_rotation @ t2) #

    estimated_translation = np.mean(translations_target, axis=0)

    # --- 4. 构建相似变换矩阵 ---
    T_sim_2_to_1 = np.identity(4)
    T_sim_2_to_1[:3, :3] = estimated_scale * estimated_rotation
    T_sim_2_to_1[:3, 3] = estimated_translation

    return T_sim_2_to_1

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用 4x4 变换矩阵变换 Nx3 点云."""
    if points.shape[1] != 3:
        raise ValueError("输入点云应为 Nx3 形状")
    if transform.shape != (4, 4):
        raise ValueError("变换矩阵应为 4x4 形状")

    points_h = np.hstack((points, np.ones((points.shape[0], 1)))) # 转为齐次坐标
    points_transformed_h = (transform @ points_h.T).T
    points_transformed = points_transformed_h[:, :3] / points_transformed_h[:, 3, np.newaxis] # 转回非齐次坐标
    return points_transformed

def transform_poses(poses_dict: Dict[Any, np.ndarray], transform: np.ndarray) -> Dict[Any, np.ndarray]:
    """使用 4x4 变换矩阵变换位姿字典."""
    transformed_poses = {}
    for frame_id, pose in poses_dict.items():
         # T_new = T_global * T_old
        transformed_poses[frame_id] = transform @ pose
    return transformed_poses


def align_reconstruction_segments(
    num_segments: int,
    point_clouds: List[np.ndarray],
    poses: List[Dict[Any, np.ndarray]],
    overlap_infos: List[List[Any]]
) -> Tuple[np.ndarray, List[Dict[Any, np.ndarray]], List[np.ndarray]]:
    """
    将多个独立重建的段对齐到第一个段的坐标系下。

    Args:
        num_segments: 总段数。
        point_clouds: 包含每个段点云 (Nx3 NumPy array) 的列表。
        poses: 包含每个段位姿字典 ({frame_id: 4x4 pose matrix}) 的列表。
        overlap_infos: 包含相邻段重叠帧ID列表的列表。
                       例如: overlap_infos[0] 是段0和段1的重叠帧ID列表,
                             overlap_infos[1] 是段1和段2的重叠帧ID列表...
                       长度应为 num_segments - 1。

    Returns:
        A tuple containing:
        - aligned_point_cloud: 合并后的总点云 (Mx3 NumPy array)。
        - aligned_poses: 所有段的位姿变换到第一个段坐标系后的列表。
        - cumulative_transforms: 每个段到第一个段的累积变换矩阵列表 (T_k_to_0)。
    """
    if not (len(point_clouds) == num_segments and len(poses) == num_segments):
        raise ValueError("点云和位姿列表的长度必须等于 num_segments")
    if len(overlap_infos) != num_segments - 1:
        raise ValueError("重叠信息列表的长度必须等于 num_segments - 1")

    # 初始化结果，以第一个段为参考
    aligned_point_clouds_list = [point_clouds[0]]
    aligned_poses_list = [poses[0]]
    cumulative_transforms = [np.identity(4)] # T_0_to_0

    # 逐个对齐后续段
    for k in range(1, num_segments):
        print(f"\n--- 对齐段 {k} 到段 {k-1} ---")
        prev_segment_poses = poses[k-1]
        current_segment_poses = poses[k]
        common_frame_ids = overlap_infos[k-1]

        if not common_frame_ids:
             print(f"警告: 段 {k-1} 和段 {k} 之间没有指定重叠帧，无法对齐。跳过段 {k}。")
             # 可以选择在这里停止，或者跳过这个段，或者尝试其他对齐方法
             # 这里我们选择记录一个无效变换，并跳过点云/位姿添加
             T_k_to_k_minus_1 = None # 或者 np.identity(4) 但标记为无效?
             current_cumulative_transform = cumulative_transforms[k-1] @ np.identity(4) # 保持之前的变换
        else:
            # 1. 估计从当前段(k)到前一段(k-1)的相似变换
            T_k_to_k_minus_1 = estimate_similarity_transform(
                prev_segment_poses, current_segment_poses, common_frame_ids
            )

            if T_k_to_k_minus_1 is None:
                 print(f"错误: 无法计算段 {k} 到段 {k-1} 的变换。跳过段 {k}。")
                 current_cumulative_transform = cumulative_transforms[k-1] @ np.identity(4)
            else:
                 # 2. 计算累积变换：从当前段(k)到参考段(0)
                 # T_k_to_0 = T_(k-1)_to_0 * T_k_to_(k-1)
                 # 注意: 这里的 T_k_to_k_minus_1 是将 k 坐标系的点变换到 k-1 坐标系
                 current_cumulative_transform = cumulative_transforms[k-1] @ T_k_to_k_minus_1#(np.linalg.inv(T_k_to_k_minus_1))

                 # 3. 变换当前段的点云和位姿到参考坐标系(0)
                 current_points_aligned = transform_points(point_clouds[k], current_cumulative_transform)
                 current_poses_aligned = transform_poses(poses[k], current_cumulative_transform)

                 # visual_open3d(point_clouds[k-1], current_points_aligned) #red  green

                 # 4. 添加到结果列表
                 aligned_point_clouds_list.append(current_points_aligned)
                 aligned_poses_list.append(current_poses_aligned)

        # 记录累积变换矩阵
        cumulative_transforms.append(current_cumulative_transform)


    # 合并所有对齐后的点云
    final_aligned_point_cloud = np.vstack(aligned_point_clouds_list)

    return final_aligned_point_cloud, aligned_poses_list, cumulative_transforms

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def getInputData(reconstructionPath):
    from demo import grid_downsample
    # 假设我们有3个段 (num_segments = 3)
    alllist = sorted(os.listdir(reconstructionPath))

    reconlist = [name for name in alllist if "segment" in name]
    instancelist = [name for name in alllist if "instance" in name]

    seg_file1 = "/home/honsen/tartan/msg_data/testSeq/rendered/test1"
    seg_file2 = "/home/honsen/tartan/msg_data/testSeq/rendered/test2"
    seg_file3 = "/home/honsen/tartan/msg_data/testSeq/rendered/test3"

    seg_file1 = sorted(os.listdir(seg_file1))
    seg_file2 = sorted(os.listdir(seg_file2))
    seg_file3 = sorted(os.listdir(seg_file3))

    seg_files = []

    seg_files.append(seg_file1)
    seg_files.append(seg_file2)
    seg_files.append(seg_file3)

    init_conf_threshold: float = 15.0

    points_seg_list = []
    instances_list = []
    poses_seg_list = []

    for i in range(len(reconlist)):

        seg_file = seg_files[i]

        recon_data = np.load(os.path.join(reconstructionPath,reconlist[i]),allow_pickle=True).item()
        instanceid = np.load(os.path.join(reconstructionPath,instancelist[i]))

        world_points = recon_data['world_points']
        conf = recon_data['world_points_conf']
        extrinsic = recon_data['extrinsic']

        # now (S, H, W, 3)
        S, H, W, _ = world_points.shape

        # Flatten
        points = world_points.reshape(-1, 3)
        conf_flat = conf.reshape(-1)

        ids_flat = instanceid.reshape(-1)

        cam_to_world_mat = closed_form_inverse_se3(extrinsic)  # shape (S, 4, 4) typically
        # For convenience, we store only (3,4) portion
        cam_to_world = cam_to_world_mat[:, :3, :]

        # Compute scene center and recenter
        scene_center = np.mean(points, axis=0)
        points_centered = points - scene_center
        cam_to_world[..., -1] -= scene_center  #47,3,4

        poses = np.array([np.eye(4) for i in range(cam_to_world.shape[0])])

        poses[:,0:3,:] = cam_to_world

        poses = {seg_file[i] : poses[i] for i  in range(poses.shape[0])}

        init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
        init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)

        saved_points = points_centered[init_conf_mask]
        save_instances = ids_flat[init_conf_mask]

        saved_points, save_instances = grid_downsample(saved_points, save_instances, voxel_size=0.03)

        points_seg_list.append(saved_points)
        instances_list.append(save_instances)
        poses_seg_list.append(poses)

    return points_seg_list, poses_seg_list, instances_list, seg_files

def visual_open3d(pc1,pc2):
    # (可选) 可视化对齐前后的点云
    import open3d as o3d
    pcds_before = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 红绿蓝区分段
    segments = [pc1,pc2]

    for i in range(n_segments):
        pcd = o3d.geometry.PointCloud()
        # 变换到近似位置以便观察未对齐状态
        T_approx = np.identity(4)
        T_approx[:3, 3] = np.array([i * 1.0, 0, 0])  # 简单错开
        # pcd.points = o3d.utility.Vector3dVector(transform_points(segments[i], T_approx))  # transform_points(all_point_clouds[i], T_approx)
        pcd.points = o3d.utility.Vector3dVector(segments[i]) # 如果坐标差别不大，可以直接显示
        pcd.paint_uniform_color(colors[i % len(colors)])
        pcds_before.append(pcd)

    print("显示对齐前的点云 (手动错开)...")
    o3d.visualization.draw_geometries(pcds_before, window_name="未对齐 (示意)")

# --- 示例用法 ---
if __name__ == '__main__':

    reconstructionPath = "/home/honsen/tartan/vggt/reconPath"

    points_seg_list, poses_seg_list, instances_list, seg_files = getInputData(reconstructionPath)

    overlap1 = set(seg_files[0])&set(seg_files[1])
    overlap2 = set(seg_files[1]) & set(seg_files[2])

    overlap1 = sorted(list(overlap1))
    overlap2 = sorted(list(overlap2))
    # 输入参数
    n_segments = 3
    all_point_clouds = [ points_seg_list[0], points_seg_list[1], points_seg_list[2]]
    all_poses = [ poses_seg_list[0], poses_seg_list[1],poses_seg_list[2]]

    # 重叠帧信息 [[seg0-seg1], [seg1-seg2], ...]
    overlap_frame_ids = [
        overlap1,  # 重叠帧在 segment 0 和 1
        overlap2,    # 重叠帧在 segment 1 和 2
    ]

    # 执行对齐
    try:
        aligned_cloud, aligned_poses_output, final_transforms = align_reconstruction_segments(
            num_segments=n_segments,
            point_clouds=all_point_clouds,
            poses=all_poses,
            overlap_infos=overlap_frame_ids
        )

        # all_instances = np.concatenate(instances_list)
        #
        # alldata = {'pcd':aligned_cloud,'instance':all_instances}
        #
        # np.save("scene1.npy", alldata, allow_pickle=True)

        print(f"\n成功完成对齐！")
        print(f"合并后的点云形状: {aligned_cloud.shape}")

        # 可以检查计算出的累积变换与真实变换（如果知道的话）
        print("\n计算得到的累积变换矩阵 (T_k_to_0):")
        for i, T in enumerate(final_transforms):
            print(f"--- 段 {i} 到 段 0 ---")
            print(np.round(T, 3))

        # (可选) 可视化对齐前后的点云
        import open3d as o3d
        pcds_before = []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # 红绿蓝区分段
        for i in range(n_segments):
            pcd = o3d.geometry.PointCloud()
            # 变换到近似位置以便观察未对齐状态
            T_approx = np.identity(4)
            T_approx[:3,3] = np.array([i*1.0, 0, 0]) # 简单错开
            pcd.points = o3d.utility.Vector3dVector(all_point_clouds[i]) #transform_points(all_point_clouds[i], T_approx)
            #pcd.points = o3d.utility.Vector3dVector(all_point_clouds[i]) # 如果坐标差别不大，可以直接显示
            pcd.paint_uniform_color(colors[i % len(colors)])
            pcds_before.append(pcd)
        print("显示对齐前的点云 (手动错开)...")
        o3d.visualization.draw_geometries(pcds_before, window_name="未对齐 (示意)")

        pcd_aligned = o3d.geometry.PointCloud()
        pcd_aligned.points = o3d.utility.Vector3dVector(aligned_cloud)
        print("显示对齐后的点云...")
        o3d.visualization.draw_geometries([pcd_aligned], window_name="对齐后")

        print()

    except ValueError as e:
        print(f"对齐过程中发生错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")