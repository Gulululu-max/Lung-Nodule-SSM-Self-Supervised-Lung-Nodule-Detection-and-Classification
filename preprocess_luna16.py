import os
import glob
import zipfile
import numpy as np
import SimpleITK as sitk
import cv2
from tqdm import tqdm
from pathlib import Path


# 1. 压缩包所在目录 (包含 subset0.zip, subset1.zip ...)
ZIP_SOURCE_DIR = r"/home/ubuntu-user/WMQ/data/subsets"

# 2. 临时解压目录 (解压后的 .mhd/.raw 将放在这里，处理完可删除)
# 建议放在空间较大的磁盘
UNZIP_TEMP_DIR = r"/home/ubuntu-user/WMQ/data/unzipped_luna16"

# 3. 最终输出目录 (生成的 PNG 将按 subset0-subset9 结构存放在这里)
# 你的特征提取程序会读取这个目录
OUTPUT_DIR = r"/home/ubuntu-user/QMW/Data"

# 4. 目标图像尺寸
TARGET_SIZE = 504

# 5. 是否清理旧数据 (True: 每次运行前清空 OUTPUT_DIR，防止旧数据干扰)
CLEAN_OUTPUT_BEFORE_RUN = True

# ===============================================================

def unzip_all_zips(source_dir, target_dir):
    """解压所有 zip 文件到目标目录，保持 subsetX 的文件夹结构"""
    print("\n📦 检查并解压数据包...")
    os.makedirs(target_dir, exist_ok=True)
    
    zip_files = list(Path(source_dir).glob("*.zip"))
    # 也兼容 .tar.gz 如果需要，这里主要处理 .zip
    if not zip_files:
        print(f"   ⚠️ 未在 {source_dir} 找到 .zip 文件。")
        print(f"   检查是否已经解压？直接查看 {target_dir} 是否有内容。")
        # 如果没有zip，假设用户已经手动解压到了 target_dir 或者 source_dir 本身就是解压后的
        # 这里为了逻辑严谨，如果没找到zip，我们尝试直接用 source_dir 作为数据源（如果里面已经有subset文件夹）
        return False 

    print(f"   发现 {len(zip_files)} 个压缩包，开始解压到 {target_dir} ...")
    
    for zip_path in tqdm(zip_files, desc="Unzipping"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 解压到 target_dir
                # 压缩包内部通常已经是 subset0/... 结构，zipfile 会保留这个结构
                zip_ref.extractall(target_dir)
        except Exception as e:
            print(f"\n❌ 解压失败 {zip_path.name}: {e}")
            
    print("   ✅ 解压完成。")
    return True

def find_mhd_files(root_dir):
    """在 root_dir 下递归查找所有 .mhd 文件，并按 subset 分组"""
    print(f"\n🔍 正在扫描 {root_dir} 下的 .mhd 文件...")
    
    # 使用 rglob 递归查找
    mhd_paths = list(Path(root_dir).rglob("*.mhd"))
    
    if not mhd_paths:
        print(f"❌ 错误：未找到任何 .mhd 文件。")
        print(f"   请检查 {root_dir} 目录下是否包含 subset0, subset1 等文件夹及对应的 .mhd/.raw 文件。")
        return []
    
    print(f"   ✅ 共找到 {len(mhd_paths)} 个 CT 扫描文件。")
    return [str(p) for p in mhd_paths]

def preprocess_luna16():
    print("="*60)
    print("🚀 LUNA16 全自动预处理 (解压 + 转换 + 保持结构)")
    print("="*60)

    # 1. 确定数据源路径
    data_source_path = UNZIP_TEMP_DIR
    
    # 尝试解压
    has_zips = unzip_all_zips(ZIP_SOURCE_DIR, UNZIP_TEMP_DIR)
    
    # 如果没找到zip，检查 UNZIP_TEMP_DIR 是否为空，如果为空，尝试直接用 ZIP_SOURCE_DIR (以防用户已经解压在原处)
    if not has_zips:
        if not list(Path(UNZIP_TEMP_DIR).glob("subset*")):
            # 检查原目录是否有 subset 文件夹
            if list(Path(ZIP_SOURCE_DIR).glob("subset*")):
                print("   ℹ️  未找到压缩包，但发现原目录已有子集文件夹，将直接使用原目录。")
                data_source_path = ZIP_SOURCE_DIR
            else:
                print("   ❌ 既没找到压缩包，也没找到解压后的子集文件夹。请检查路径配置。")
                return

    # 验证子集文件夹是否存在
    subset_folders = [d for d in Path(data_source_path).iterdir() if d.is_dir() and d.name.startswith("subset")]
    if not subset_folders:
        print(f"❌ 错误：在 {data_source_path} 中未找到以 'subset' 开头的文件夹。")
        return
    print(f"   📂 检测到 {len(subset_folders)} 个子集目录: {[f.name for f in sorted(subset_folders)]}")

    # 2. 准备输出目录
    if os.path.exists(OUTPUT_DIR):
        if CLEAN_OUTPUT_BEFORE_RUN:
            print(f"\n🗑️  正在清空输出目录：{OUTPUT_DIR}")
            import shutil
            shutil.rmtree(OUTPUT_DIR)
        else:
            print(f"\n⚠️  输出目录已存在，将追加数据（可能导致重复）。")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 获取所有 MHD 文件
    mhd_files = find_mhd_files(data_source_path)
    if not mhd_files:
        return

    processed_count = 0
    error_count = 0

    # 4. 开始处理
    # 关键点：我们需要知道每个文件属于哪个 subset，以便在输出端重建目录结构
    for mhd_path in tqdm(mhd_files, desc="Processing Scans"):
        try:
            path_obj = Path(mhd_path)
            
            # 【核心逻辑】找到该文件相对于 data_source_path 的相对路径
            # 例如：/unzipped/subset0/123.mhd -> subset0/123.mhd
            try:
                rel_path = path_obj.relative_to(data_source_path)
            except ValueError:
                # 如果不在 data_source_path 下（极少见），取最后两级
                rel_path = Path(path_obj.parent.name) / path_obj.name

            # 提取 subset 名称 (例如 "subset0")
            subset_name = rel_path.parts[0] 
            filename = path_obj.name
            seriesuid = filename.replace(".mhd", "")

            # 【构建输出路径】
            # 目标结构：OUTPUT_DIR / subset0 / seriesuid_z.png
            out_subset_dir = os.path.join(OUTPUT_DIR, subset_name)
            os.makedirs(out_subset_dir, exist_ok=True)
            
            out_series_dir = out_subset_dir # 直接放在 subset 文件夹下
            # 如果希望每个病人一个文件夹，取消下面这行的注释：
            # out_series_dir = os.path.join(out_subset_dir, seriesuid)
            # os.makedirs(out_series_dir, exist_ok=True)

            # 读取图像
            img_itk = sitk.ReadImage(mhd_path)
            img_np = sitk.GetArrayFromImage(img_itk)
            
            if img_np.ndim != 3:
                raise ValueError(f"维度错误：{img_np.ndim}")

            # HU 截断 & 归一化
            img_np = np.clip(img_np, -1200, 600)
            img_np = ((img_np + 1200) / 1800.0 * 255).astype(np.uint8)

            depth = img_np.shape[0]

            for z_idx in range(depth):
                slice_2d = img_np[z_idx, :, :]
                slice_resized = cv2.resize(slice_2d, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
                
                save_filename = f"{seriesuid}_{z_idx}.png"
                save_path = os.path.join(out_series_dir, save_filename)
                
                cv2.imwrite(save_path, slice_resized)

            processed_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"\n❌ 处理失败 {path_obj.name}: {e}")
    
    # 5. 总结与验证
    print("\n" + "="*60)
    print("✅ 预处理完成!")
    print(f"   成功：{processed_count} | 失败：{error_count}")
    print(f"   输出位置：{OUTPUT_DIR}")
    
    # 验证目录结构是否符合预期
    print("\n🔍 验证输出结构...")
    expected_subsets = [f"subset{i}" for i in range(10)]
    found_subsets = []
    for sub in expected_subsets:
        sub_path = os.path.join(OUTPUT_DIR, sub)
        if os.path.exists(sub_path):
            count = len(glob.glob(os.path.join(sub_path, "*.png")))
            found_subsets.append(sub)
            print(f"   ✅ {sub}: 存在 ({count} 张 PNG)")
        else:
            print(f"   ❌ {sub}: 缺失")
            
    if len(found_subsets) == 10:
        print("\n🎉 完美！所有 10 个子集目录已生成，特征提取程序应能正常识别。")
    else:
        print("\n⚠️ 警告：部分子集目录缺失，请检查输入数据是否完整。")

    print("="*60)
    print("📝 下一步:")
    print("   确保 config.py 中的 slices_dir 指向:", OUTPUT_DIR)
    print("   然后运行特征提取程序。")

if __name__ == "__main__":
    preprocess_luna16()