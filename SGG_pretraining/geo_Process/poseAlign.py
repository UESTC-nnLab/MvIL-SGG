import numpy as np

from geo_Process.TdMto2d import read_jpg_and_txt_files, load_extrinsic_from_txt


def umeyama_alignment(true_poses, est_poses):
    """
    使用 Umeyama 算法计算最优刚性变换 (R, t)，使得 true_poses ≈ R @ est_poses + t
    Args:
        true_poses: 真实位姿列表 [N x 4x4]
        est_poses: 估计位姿列表 [N x 4x4]
    Returns:
        T: 刚性变换矩阵 (4x4)
    """
    # 提取平移部分 (假设位姿是相机到世界的变换)
    true_translations = np.array([pose[:3, 3] for pose in true_poses])
    est_translations = np.array([pose[:3, 3] for pose in est_poses])

    # 中心化
    true_centroid = np.mean(true_translations, axis=0)
    est_centroid = np.mean(est_translations, axis=0)
    true_centered = true_translations - true_centroid
    est_centered = est_translations - est_centroid

    # 计算 SVD: H = U * S * V^T
    H = np.dot(est_centered.T, true_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 计算平移 t = true_centroid - R @ est_centroid
    t = true_centroid - np.dot(R, est_centroid)

    # 构造 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

if __name__ =="__main__":

    recons_file = "/home/honsen/tartan/msg_data/3rscan/ab835faa-54c6-29a1-9b55-1a5217fcba19/sequence"

    _, pose_files = read_jpg_and_txt_files(recons_file)

    true_poses = []

    for i in range(21):
        true_poses.append(load_extrinsic_from_txt(pose_files[i]))

    custom_extrinsic = np.load("/home/honsen/tartan/vggt/reconPath/segment1.npy", allow_pickle=True).item()

    est_extr = custom_extrinsic['extrinsic']

    est_poses = []

    for i in range(21):
        cust_extr = np.eye(4)

        cust_extr[0:3, :] = est_extr[i]

        cus_pose = np.linalg.inv(cust_extr)

        est_poses.append(cus_pose)

    # 计算最优变换
    T = umeyama_alignment(true_poses, est_poses)
    print("Umeyama 变换 T:\n", T)

    # 对齐所有位姿
    aligned_poses = [np.dot(T, pose) for pose in est_poses]