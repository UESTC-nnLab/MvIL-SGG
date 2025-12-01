import os
import zipfile
from tqdm import tqdm  # 进度条工具（可选，安装：pip install tqdm）

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_compressed_files(parent_folders, supported_extensions=['.zip']):
    """
    处理多个父文件夹下的压缩包
    :param parent_folders: 包含目标文件夹路径的列表，如 ['folder1', 'folder2']
    :param supported_extensions: 支持的压缩文件扩展名
    """

    for parent_folder in parent_folders:

        p_root = "/home/honsen/tartan/msg_data/3rscan/"

        parent_folder = p_root+parent_folder

        if not os.path.exists(parent_folder):
            print(f"警告：文件夹 '{parent_folder}' 不存在，跳过。")
            continue

        print(f"\n处理文件夹: {parent_folder}")
        for root, _, files in os.walk(parent_folder):
            for file in tqdm(files, desc=f"解压文件 ({root})"):
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                if file_ext in supported_extensions:
                    extract_dir = os.path.join(root, os.path.splitext(file)[0])
                    os.makedirs(extract_dir, exist_ok=True)

                    try:
                        extract_zip(file_path, extract_dir)

                        print(f"解压成功: {file} -> {extract_dir}")
                    except Exception as e:
                        print(f"解压失败: {file} (错误: {str(e)})")


if __name__ == "__main__":
    # 示例：指定需要扫描的多个文件夹路径
    target_folders = [
        'path/to/folder1',
        'path/to/folder2'
    ]

    process_compressed_files(target_folders)