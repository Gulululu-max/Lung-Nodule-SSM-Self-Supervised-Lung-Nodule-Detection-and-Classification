import os
import numpy as np
import torch
import cv2
import glob
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
from torchvision import transforms

def get_metadata_or_default(mhd_dir, seriesuid):
    mhd_file = glob.glob(os.path.join(mhd_dir, "**", f"{seriesuid}.mhd"), recursive=True)
    if mhd_file:
        try:
            img = sitk.ReadImage(mhd_file[0])
            origin = np.array(img.GetOrigin())[::-1]  # Reverse order
            spacing = np.array(img.GetSpacing())[::-1]  # Reverse order
            return origin, spacing
        except Exception as e:
            print(f"Error reading {mhd_file[0]}: {e}")
    print(f"⚠️ Metadata not found for {seriesuid}, using defaults.")
    return np.array([-256.0, -256.0, -500.0]), np.array([1.0, 1.0, 1.0])

class Luna16Dataset(Dataset):
    def __init__(self, mhd_dir, slices_dir, candidates_df, transform, dino_model, device, 
                 patient_ids=None, max_slices=None):
        """
        Args:
            patient_ids (list, optional): List of seriesuid to include. 
                                          If None, includes all patients.
            max_slices (int, optional): Max total slices to load (for debugging).
        """
        self.mhd_dir = mhd_dir
        self.slices_dir = slices_dir
        self.candidates_df = candidates_df
        self.transform = transform
        self.dino_model = dino_model  # 这个实际上只用于bbox推断，不需要在__getitem__中使用
        self.device = device
        self.data_info = []
        self.slice_counts = {}

        # 🔑 核心修改：构建病人 ID 集合用于快速查找
        allowed_patients = set(patient_ids) if patient_ids is not None else None
        
        if allowed_patients:
            print(f"Filtering dataset for {len(allowed_patients)} specific patients...")
            # 可选：提前过滤 candidates_df 以加速后续循环
            self.candidates_df = self.candidates_df[self.candidates_df['seriesuid'].isin(allowed_patients)]

        print("Loading dataset with 3D-aware processing... This may take some time.")
        
        # 预先构建 candidates 映射: {seriesuid: [row_data]}
        candidate_map = {}
        for _, row in self.candidates_df.iterrows():
            uid = row['seriesuid']
            if uid not in candidate_map:
                candidate_map[uid] = []
            candidate_map[uid].append(row)

        total_loaded = 0
        
        for subset in range(10):
            # ✅ 检查 max_slices 限制
            if max_slices is not None and total_loaded >= max_slices:
                break

            subset_slices_path = os.path.join(self.slices_dir, f"subset{subset}")
            if not os.path.exists(subset_slices_path):
                continue

            all_png_files = [f for f in os.listdir(subset_slices_path) if f.endswith('.png')]
            if not all_png_files:
                continue

            # 按 seriesuid 分组文件
            files_by_uid = {}
            for fname in all_png_files:
                try:
                    parts = fname.rsplit('_', 1)
                    if len(parts) != 2:
                        continue 
                    uid = parts[0]
                    
                    # 🔑 过滤：如果指定了病人列表且当前 UID 不在列表中，跳过
                    if allowed_patients is not None and uid not in allowed_patients:
                        continue
                        
                    if uid not in files_by_uid:
                        files_by_uid[uid] = []
                    files_by_uid[uid].append(fname)
                except Exception:
                    continue

            if not files_by_uid:
                continue

            print(f"   Processing subset{subset}: {len(files_by_uid)} patients found (after filtering)")

            # 遍历每个病人
            for seriesuid, file_list in tqdm(files_by_uid.items(), desc=f"Indexing subset{subset}"):
                # ✅ 内部检查 max_slices
                if max_slices is not None and total_loaded >= max_slices:
                    break

                # 1. 获取 Metadata
                origin, spacing = get_metadata_or_default(self.mhd_dir, seriesuid)
                
                # 2. 计算该病人的结节所在切片索引 (Z-index) - 这里是3D映射的核心
                target_z_indices = set()
                target_coords = {}  # 存储坐标信息用于边界框计算
                if seriesuid in candidate_map:
                    for row in candidate_map[seriesuid]:
                        try:
                            c_x, c_y, c_z = row["coordX"], row["coordY"], row["coordZ"]
                            o_z = origin[2]
                            s_z = spacing[2]
                            if s_z == 0: s_z = 1.0
                            z_idx = int(np.rint((c_z - o_z) / s_z))
                            
                            # 存储3D坐标信息用于精确边界框计算
                            target_z_indices.add(z_idx)
                            if z_idx not in target_coords:
                                target_coords[z_idx] = []
                            target_coords[z_idx].append((c_x, c_y, c_z))
                        except Exception as e:
                            print(f"Error processing coordinates for {seriesuid}: {e}")
                            continue

                self.slice_counts[seriesuid] = len(file_list)

                # 3. 遍历该病人的所有切片文件 - 现在我们只关注包含结节的切片
                for slice_file in file_list:
                    # ✅ 最内层检查 max_slices
                    if max_slices is not None and total_loaded >= max_slices:
                        break

                    try:
                        z_str = slice_file.rsplit('_', 1)[1].replace('.png', '')
                        z = int(z_str)
                    except (IndexError, ValueError):
                        continue

                    slice_path = os.path.join(subset_slices_path, slice_file)
                    
                    # 判断标签 - 只有当切片索引匹配结节Z坐标时才是阳性
                    label = 1 if z in target_z_indices else 0
                    
                    # 推断 bbox (如果是阳性) - 使用3D坐标信息
                    bbox = [0, 0, 0, 0]
                    if label == 1 and z in target_coords:
                        # 使用3D坐标信息计算更精确的边界框
                        bbox = self.infer_bbox_from_3d_coords(slice_path, target_coords[z], origin, spacing)

                    # 🔑 关键修改：编码3D上下文到3个通道
                    processed_slice_path = self._create_3d_context_slice(
                        seriesuid, z, file_list, subset_slices_path
                    )
                    
                    # 使用处理后的切片路径（包含3D上下文）
                    self.data_info.append((processed_slice_path, label, bbox, slice_path))
                    total_loaded += 1
                
                # 如果是因为 max_slices 跳出，需继续向外跳出
                if max_slices is not None and total_loaded >= max_slices:
                    break
            
            if max_slices is not None and total_loaded >= max_slices:
                break

        print(f"✅ Loaded {len(self.data_info)} slices with 3D context preservation.")
        
        if len(self.data_info) == 0:
            print("WARNING: No valid slices found! Attempting fallback...")
            self._load_fallback_data()

    def _create_3d_context_slice(self, seriesuid, z_idx, file_list, subset_path):
        """
        创建包含3D上下文的多通道切片
        模拟论文中"replicated across all three channels to simulate 3D spatial context"
        """
        # Sort file list by z index to find adjacent slices
        sorted_files = sorted(file_list, key=lambda x: int(x.rsplit('_', 1)[1].replace('.png', '')))
        
        # Find indices of adjacent slices
        current_idx = sorted_files.index(f"{seriesuid}_{z_idx}.png") if f"{seriesuid}_{z_idx}.png" in sorted_files else -1
        
        if current_idx == -1:
            # Fallback: just return the original slice
            return os.path.join(subset_path, sorted_files[0])
        
        # Get adjacent slice indices
        adj_slices = []
        for offset in [-1, 0, 1]:  # Previous, current, next slice
            adj_idx = current_idx + offset
            if 0 <= adj_idx < len(sorted_files):
                adj_slices.append(os.path.join(subset_path, sorted_files[adj_idx]))
            else:
                # If out of bounds, repeat the current slice
                adj_slices.append(os.path.join(subset_path, sorted_files[current_idx]))
        
        # Create a temporary file with 3-channel representation
        import tempfile
        import uuid
        
        # Load the three slices
        channels = []
        for slice_path in adj_slices:
            slice_img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            if slice_img is None:
                slice_img = np.zeros((512, 512), dtype=np.uint8)  # fallback
            slice_img = cv2.resize(slice_img, (512, 512))
            channels.append(slice_img)
        
        # Stack into 3-channel image
        rgb_slice = np.stack(channels, axis=-1)  # Shape: (H, W, 3)
        
        # Save to temporary file
        temp_dir = "/tmp/dino_3d_context"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = os.path.join(temp_dir, f"temp_{seriesuid}_{z_idx}_{uuid.uuid4().hex}.png")
        cv2.imwrite(temp_filename, rgb_slice)
        
        return temp_filename

    def infer_bbox_from_3d_coords(self, slice_path, coords_3d_list, origin, spacing):
        """
        基于3D坐标信息推断2D边界框
        """
        # 取第一个坐标点（如果有多个重叠结节）
        if not coords_3d_list:
            return [0, 0, 0, 0]
        
        coord_x, coord_y, coord_z = coords_3d_list[0]
        
        # Convert 3D world coordinates to 2D pixel coordinates on this slice
        # Calculate pixel position based on spacing and origin
        pixel_x = (coord_x - origin[0]) / spacing[0]
        pixel_y = (coord_y - origin[1]) / spacing[1]
        
        # Typical nodule size (in mm), convert to pixels based on spacing
        typical_nodule_diameter_mm = 8.0  # 8mm average nodule diameter
        avg_spacing = (spacing[0] + spacing[1]) / 2  # Average x-y spacing
        nodule_radius_pixels = (typical_nodule_diameter_mm / 2) / avg_spacing
        
        # Convert to normalized coordinates [0, 1]
        # Assuming the slice is 512x512 pixels (adjust if different)
        slice_width, slice_height = 512, 512
        
        # Calculate bounding box
        x_center_norm = pixel_x / slice_width
        y_center_norm = pixel_y / slice_height
        width_norm = (2 * nodule_radius_pixels) / slice_width
        height_norm = (2 * nodule_radius_pixels) / slice_height
        
        # Ensure bounds
        x_min = max(0, x_center_norm - width_norm/2)
        y_min = max(0, y_center_norm - height_norm/2)
        width = min(1, width_norm)
        height = min(1, height_norm)
        
        return [x_min, y_min, width, height]

    def _load_fallback_data(self):
        png_files = glob.glob(os.path.join(self.slices_dir, "**", "*.png"), recursive=True)
        print(f"Found {len(png_files)} PNG files in fallback.")
        if len(png_files) > 1000:
            png_files = png_files[:1000]
        for slice_path in png_files:
            self.data_info.append((slice_path, 0, [0, 0, 0, 0], slice_path))
        print(f"Added {len(self.data_info)} slices using fallback method.")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        processed_slice_path, label, bbox, original_slice_path = self.data_info[idx]
        try:
            # Load the 3D-context encoded slice (3-channel RGB)
            slice_2d = cv2.imread(processed_slice_path, cv2.IMREAD_COLOR)  # Load as 3-channel
            if slice_2d is None:
                print(f"Error loading image {processed_slice_path}, using zeroed image")
                slice_2d = np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Ensure it's 3-channel (double check)
            if len(slice_2d.shape) == 2:
                slice_2d = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
            elif slice_2d.shape[2] == 1:
                slice_2d = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
            
            # Apply transform (should work with 3-channel input)
            # IMPORTANT: This returns the IMAGE tensor, NOT the features
            slice_tensor = self.transform(slice_2d).to(self.device)
            return slice_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(bbox, dtype=torch.float32), original_slice_path
        except Exception as e:
            print(f"Error processing {processed_slice_path}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(3, 512, 512), torch.tensor(0, dtype=torch.long), torch.zeros(4, dtype=torch.float32), original_slice_path