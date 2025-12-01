import numpy as np
import torch
import open3d as o3d
from PIL import Image
from typing import List, Dict, Tuple
import os
import trimesh
from utils import util_ply
import cv2
def load_point_cloud(pcd_path: str) -> np.ndarray:
    """加载点云数据"""
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

def load_instance_masks(mask_path: str) -> Dict[int, np.ndarray]:
    """加载实例分割掩码，返回字典{实例ID: 点索引数组}"""
    # 假设掩码是以.npy格式存储的，形状为(N,)，每个元素是实例ID
    instance_ids = np.load(mask_path)
    unique_ids = np.unique(instance_ids)
    masks = {}
    for id_ in unique_ids:
        if id_ == 0:  # 假设0表示背景
            continue
        masks[id_] = np.where(instance_ids == id_)[0]
    return masks

def project_points_to_image(
    points: np.ndarray, 
    pose: np.ndarray, 
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    instance_masks
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将3D点投影到2D图像平面
    参数:
        points: (N, 3) 点云坐标
        pose: (4, 4) 相机位姿矩阵 (世界到相机)
        intrinsics: (3, 3) 相机内参矩阵
        image_size: (H, W) 图像尺寸
    返回:
        projected: (M, 2) 投影后的2D坐标 (u, v)
        valid_indices: (M,) 有效点的原始索引
    """
    import matplotlib.pyplot as plt
    # 将点转换为齐次坐标 (N, 4)


    all_data = np.load("/home/honsen/honsen/CVPR2023-VLSAT/geo_Process/scene1.npy", allow_pickle=True).item()
    custom_extrinsic = np.load("/home/honsen/tartan/vggt/reconPath/segment1.npy", allow_pickle=True).item()

    custom_extrinsic = custom_extrinsic['extrinsic'][0]

    cust_extr = np.eye(4)

    cust_extr[0:3, :] = custom_extrinsic

    instances1 = all_data['instance']
    points1 = all_data['pcd']

    bestrans1 = [[0.9849332, -0.10805957, 0.13501749, 0.01958854],
                 [-0.17118331, -0.49835685, 0.84990395, 0.15589916],
                 [-0.02455336, -0.86021136, - 0.5093462, - 0.33497217],
                 [0., 0., 0., 1.]]

    instanceidx6 = np.where(instances1 == 6)

    point6 = points1[instanceidx6]

    points_hom = np.hstack([points1, np.ones((points1.shape[0], 1))])

    points_hom = points_hom[instanceidx6]

    extrinsic = np.linalg.inv(pose)

    # intrinsics = [[376.71405, 0.00000, 175.00000,0],
    # [0.00000, 570.28235, 175.00000,0],
    # [0.00000, 0.00000, 1.00000,0],
    # [0,0,0,1]]
    #
    # intrinsics = np.array(intrinsics)

    intrinsics[0,2] = 270.419
    intrinsics[1,2] = 492.889

    # 世界坐标到相机坐标
    w_2_c = (cust_extr @ points_hom.T)# (N, 4)
    c_2_i = intrinsics[:3, :] @ w_2_c  # n_frames x 3 x n_points
    c_2_i = c_2_i.transpose(1, 0)  # n_frames x n_points x 3

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(point6[:, 0], point6[:, 1], point6[:, 2], c='k', s=1)
    ax.set_xlabel('x (m)')
    ax.set_zlabel('z (m)')
    ax.set_ylabel('y (m)')
    plt.show()

    projected = c_2_i[..., :2] / c_2_i[..., 2:]  # n_frames x n_points x 2

    # 过滤掉图像外的点
    h, w = image_size
    h = 900
    w = 500
    in_image = (projected[:, 0] >= 0) & (projected[:, 0] < w) & \
               (projected[:, 1] >= 0) & (projected[:, 1] < h)

    projected1 = projected[in_image]

    fig = plt.figure(figsize=(8, 16))
    ax = fig.add_subplot()
    ax.scatter(projected1[:, 0], projected1[:, 1], c='k')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.show()

    return projected[in_image], in_image

def create_2d_mask_from_projection(
    projected_points: np.ndarray,
    point_indices: np.ndarray,
    instance_masks: Dict[int, np.ndarray],
    image_size: Tuple[int, int],
    colors
) -> Dict[int, np.ndarray]:
    """
    根据投影结果创建2D实例掩码
    参数:
        projected_points: (M, 2) 投影后的2D坐标
        point_indices: (M,) 对应的原始点云索引
        instance_masks: 实例分割字典 {id: 点索引数组}
        image_size: (H, W) 图像尺寸
    返回:
        2D实例掩码字典 {id: (H, W) 二值掩码}
    """
    h, w = image_size

    # 创建空的2D掩码
    mask_2d = -np.ones((h, w, 3), dtype=np.uint8)

    for i in range(len(projected_points)):

        color = colors[0]
        u = np.round(projected_points[i, 0]).astype(int)
        v = np.round(projected_points[i, 1]).astype(int)
        # 确保坐标在图像范围内
        u = np.clip(u, 0, w - 1)
        v = np.clip(v, 0, h - 1)

        mask_2d[v,u, 0]=color[2]
        mask_2d[v, u, 1] = color[1]
        mask_2d[v, u, 2] = color[0]

    cv2.imshow("qwe",mask_2d)
    cv2.waitKey(10000)

    return mask_2d

def process_scene(
    pcd_path: str,
    image_paths: List[str],
    poses: List[np.ndarray],
    intrinsics,
    output_dir: str
):
    """
    处理整个场景，为每张图像生成2D实例掩码
    参数:
        pcd_path: 点云文件路径
        mask_path: 实例掩码文件路径
        image_paths: 图像路径列表
        poses: 相机位姿列表 (世界到相机)
        intrinsics_list: 相机内参列表
        output_dir: 输出目录
    """
    # 加载数据

    plydata = trimesh.load(pcd_path, process=False)
    points = np.array(plydata.vertices)
    instance_masks = util_ply.read_labels(plydata).flatten()

    # 确保数量一致
    assert len(image_paths) == len(poses)
    
    for i, (img_path, pose) in enumerate(zip(image_paths, poses)):
        # 加载图像获取尺寸
        img = Image.open(img_path)
        img_size = (img.height, img.width)

        intrinsics = intrinsics['m_intrinsic']
        # 投影点云到图像
        projected, indices = project_points_to_image(points, pose, intrinsics, img_size,instance_masks)

        colors = get_distinct_colors(50)

        # 生成2D掩码
        masks_2d = create_2d_mask_from_projection(projected, indices, instance_masks, img_size,colors)
        
        # 保存结果
        for id_, mask in masks_2d.items():
            mask_img = Image.fromarray(mask * 255)
            output_path = f"{output_dir}/frame_{i:04d}_instance_{id_:04d}.png"
            mask_img.save(output_path)
        
        print(f"Processed frame {i+1}/{len(image_paths)}")

def read_jpg_and_txt_files(folder_path):
    """
    读取文件夹中的.jpg和.txt文件到两个单独的列表
    
    参数:
        folder_path: 要扫描的文件夹路径
        
    返回:
        jpg_files: 所有.jpg文件的完整路径列表
        txt_files: 所有.txt文件的完整路径列表
    """
    jpg_files = []
    txt_files = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件扩展名并分类
        if filename.lower().endswith('.jpg'):
            jpg_files.append(file_path)
        elif filename.lower().endswith('pose.txt'):
            txt_files.append(file_path)
    
    # 对文件名进行排序，确保对应关系
    jpg_files.sort()
    txt_files.sort()
    
    return jpg_files, txt_files

def load_extrinsic_from_txt(file_path):
    """
    从.txt文件读取4×4相机外参矩阵
    
    参数:
        file_path: .txt文件路径
        
    返回:
        4×4的NumPy数组
        
    文件格式示例:
        0.1 0.2 0.3 0.4
        0.5 0.6 0.7 0.8
        0.9 1.0 1.1 1.2
        0.0 0.0 0.0 1.0
    """
    with open(file_path, 'r') as f:
        # 读取所有行并去除空行
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 检查是否为4行
        if len(lines) != 4:
            raise ValueError(f"文件 {file_path} 应该有4行数据，但找到了 {len(lines)} 行")
            
        # 解析每行数据
        matrix = []
        for line in lines:
            # 分割字符串为数字列表
            row = [float(x) for x in line.split()]
            if len(row) != 4:
                raise ValueError(f"每行应该有4个数字，但找到了 {len(row)} 个")
            matrix.append(row)
            
    return np.array(matrix)

import colorsys
import random
def get_distinct_colors(num_colors):
    """生成视觉上可区分的颜色"""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.7 + random.random() * 0.3
        value = 0.5 + random.random() * 0.5
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

# 生成50种视觉可区分的颜色
distinct_colors = get_distinct_colors(50)
print(distinct_colors)


def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()

    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]

    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])
        m_Height = int(data[3].strip().split(' ')[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(data[4].strip().split(' ')[-1])
        m_Height = int(data[5].strip().split(' ')[-1])
        m_Shift = int(data[6].strip().split(' ')[-1])
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))

    m_frames_size = int(data[11].strip().split(' ')[-1])

    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=m_intrinsic,
        m_frames_size=m_frames_size
    )

if __name__ == "__main__":


    # 假设数据路径和参数
    pcd_path = "/home/honsen/tartan/msg_data/3rscan/ab835faa-54c6-29a1-9b55-1a5217fcba19/labels.instances.annotated.v2.ply"
    recons_file = "/home/honsen/tartan/msg_data/3rscan/ab835faa-54c6-29a1-9b55-1a5217fcba19/sequence"

    img_paths, pose_files = read_jpg_and_txt_files(recons_file)

    # image_paths = os.listdir("/home/honsen/tartan/msg_data/testSeq/rgb")
    # img_paths = []
    # for i,img_path in enumerate(image_paths):
    #     img_paths.append(os.path.join("/home/honsen/tartan/msg_data/testSeq/rgb",img_path))

    res_intrinsic = read_intrinsic("/home/honsen/tartan/msg_data/3rscan/ab835faa-54c6-29a1-9b55-1a5217fcba19/sequence/_info.txt")

    poselist = []

    for i in range (len(pose_files)):
        poselist.append(load_extrinsic_from_txt(pose_files[i]))


    output_dir = "output_masks"
    
    # 处理场景
    process_scene(pcd_path, img_paths, poselist, res_intrinsic, output_dir)