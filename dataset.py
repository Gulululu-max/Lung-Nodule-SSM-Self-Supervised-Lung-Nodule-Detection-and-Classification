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
            origin = np.array(img.GetOrigin())[::-1]
            spacing = np.array(img.GetSpacing())[::-1]
            return origin, spacing
        except Exception as e:
            print(f"Error reading {mhd_file[0]}: {e}")
    # print(f"⚠️ Metadata not found for {seriesuid}, using defaults.") # 减少噪音
    return np.array([-256.0, -256.0, -500.0]), np.array([1.0, 1.0, 1.0])

class Luna16Dataset(Dataset):
    def __init__(self, mhd_dir, slices_dir, candidates_df, transform, dino_model, device, 
                 patient_ids=None, max_slices=None):
        """
        Args:
            patient_ids (list, optional): List of seriesuid to include. 
            max_slices (int, optional): Max total slices to load.
        """
        self.mhd_dir = mhd_dir
        self.slices_dir = slices_dir
        self.candidates_df = candidates_df
        self.transform = transform
        self.dino_model = dino_model  # 可能是 None
        self.device = device
        self.data_info = []
        self.slice_counts = {}

        allowed_patients = set(patient_ids) if patient_ids is not None else None
        
        if allowed_patients:
            # print(f"Filtering dataset for {len(allowed_patients)} specific patients...")
            self.candidates_df = self.candidates_df[self.candidates_df['seriesuid'].isin(allowed_patients)]

        # print("Loading dataset...")
        
        candidate_map = {}
        for _, row in self.candidates_df.iterrows():
            uid = row['seriesuid']
            if uid not in candidate_map:
                candidate_map[uid] = []
            candidate_map[uid].append(row)

        total_loaded = 0
        skip_bbox_count = 0
        
        for subset in range(10):
            if max_slices is not None and total_loaded >= max_slices:
                break

            subset_slices_path = os.path.join(self.slices_dir, f"subset{subset}")
            if not os.path.exists(subset_slices_path):
                continue

            all_png_files = [f for f in os.listdir(subset_slices_path) if f.endswith('.png')]
            if not all_png_files:
                continue

            files_by_uid = {}
            for fname in all_png_files:
                try:
                    parts = fname.rsplit('_', 1)
                    if len(parts) != 2:
                        continue 
                    uid = parts[0]
                    
                    if allowed_patients is not None and uid not in allowed_patients:
                        continue
                        
                    if uid not in files_by_uid:
                        files_by_uid[uid] = []
                    files_by_uid[uid].append(fname)
                except Exception:
                    continue

            if not files_by_uid:
                continue

            # print(f"   Processing subset{subset}: {len(files_by_uid)} patients...")

            for seriesuid, file_list in tqdm(files_by_uid.items(), desc=f"Indexing subset{subset}", leave=False):
                if max_slices is not None and total_loaded >= max_slices:
                    break

                origin, spacing = get_metadata_or_default(self.mhd_dir, seriesuid)
                
                target_z_indices = set()
                if seriesuid in candidate_map:
                    for row in candidate_map[seriesuid]:
                        try:
                            c_z = row["coordZ"]
                            o_z = origin[2]
                            s_z = spacing[2]
                            if s_z == 0: s_z = 1.0
                            z_idx = int(np.rint((c_z - o_z) / s_z))
                            target_z_indices.add(z_idx)
                        except Exception:
                            pass

                self.slice_counts[seriesuid] = len(file_list)

                for slice_file in file_list:
                    if max_slices is not None and total_loaded >= max_slices:
                        break

                    try:
                        z_str = slice_file.rsplit('_', 1)[1].replace('.png', '')
                        z = int(z_str)
                    except (IndexError, ValueError):
                        continue

                    slice_path = os.path.join(subset_slices_path, slice_file)
                    label = 1 if z in target_z_indices else 0
                    
                    # ✅ 核心修改：智能处理 Bbox
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    if label == 1:
                        if self.dino_model is not None:
                            # 只有当模型存在时才进行耗时的特征推断
                            bbox = self.infer_bbox_from_features(slice_path)
                        else:
                            # 🔑 关键：如果是端到端模式 (dino_model=None)，使用默认 Bbox
                            # 这里使用 [0, 0, 1, 1] 表示整个图像都是 ROI，或者中心一个小框
                            # 对于分类任务，Bbox Loss 可能被忽略或使用默认值，不会导致崩溃
                            bbox = [0.0, 0.0, 1.0, 1.0] 
                            skip_bbox_count += 1
                    else:
                        # 阴性样本不需要 Bbox
                        bbox = [0.0, 0.0, 0.0, 0.0]

                    self.data_info.append((slice_path, label, bbox))
                    total_loaded += 1
                
                if max_slices is not None and total_loaded >= max_slices:
                    break
            
            if max_slices is not None and total_loaded >= max_slices:
                break

        print(f"✅ Loaded {len(self.data_info)} slices. (Skipped Bbox inference for {skip_bbox_count} samples due to missing model)")
        
        if len(self.data_info) == 0:
            print("WARNING: No valid slices found! Attempting fallback...")
            self._load_fallback_data()

    def infer_bbox_from_features(self, slice_path):
        # ✅ 防御性检查：防止外部调用时模型为 None
        if self.dino_model is None:
            return [0.0, 0.0, 1.0, 1.0]

        try:
            slice_2d = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            if slice_2d is None:
                return [0.0, 0.0, 1.0, 1.0]
            
            slice_2d_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
            slice_tensor = self.transform(slice_2d_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.dino_model(slice_tensor)
            
            # DINOv2 ViT-L/14 output shape check
            # Usually [B, Num_Tokens, Dim]. We need to handle patch extraction carefully.
            # The original logic assumes specific patching on 504x504 image.
            # Ensure slice_tensor is correctly shaped for patching.
            
            patch_size = 14
            # Note: This patching logic assumes the input tensor is already resized to 504x504 by transform
            # If transform resizes to 224 or other, this logic breaks. 
            # Assuming transform keeps it at 504 or resizes to 504 based on your previous code context.
            
            h, w = slice_tensor.shape[2], slice_tensor.shape[3]
            patches = []
            coords = []
            
            for i in range(0, h, patch_size):
                for j in range(0, w, patch_size):
                    if i + patch_size > h or j + patch_size > w:
                        continue
                    patch = slice_tensor[:, :, i:i+patch_size, j:j+patch_size]
                    patches.append(patch)
                    coords.append((i, j))
            
            if not patches:
                return [0.0, 0.0, 1.0, 1.0]
                
            patches_tensor = torch.cat(patches, dim=0)

            with torch.no_grad():
                patch_features = self.dino_model(patches_tensor)
            
            # Handle feature dimensions (CLS token vs flattened)
            # Assuming we take CLS token for comparison: features[:, 0, :]
            if len(features.shape) == 3:
                slice_feature = features[0, 0, :] # CLS token
                patch_features_cls = patch_features[:, 0, :] # CLS tokens for patches
            else:
                # Fallback if shapes are different
                slice_feature = features.squeeze(0)
                patch_features_cls = patch_features.squeeze(1) if len(patch_features.shape) > 2 else patch_features

            similarities = torch.nn.functional.cosine_similarity(patch_features_cls, slice_feature.unsqueeze(0), dim=1)
            max_sim_idx = similarities.argmax().item()
            
            num_patches_per_row = w // patch_size
            center_patch_y = (max_sim_idx // num_patches_per_row) * patch_size + patch_size // 2
            center_patch_x = (max_sim_idx % num_patches_per_row) * patch_size + patch_size // 2

            box_size = 20 
            half_box = box_size // 2
            x_min = max(0, center_patch_x - half_box)
            x_max = min(w, center_patch_x + half_box)
            y_min = max(0, center_patch_y - half_box)
            y_max = min(h, center_patch_y + half_box)

            bbox = [x_min / float(w), y_min / float(h), (x_max - x_min) / float(w), (y_max - y_min) / float(h)]
            return bbox
        except Exception as e:
            # print(f"Error inferring bbox for {slice_path}: {e}")
            return [0.0, 0.0, 1.0, 1.0]

    def _load_fallback_data(self):
        png_files = glob.glob(os.path.join(self.slices_dir, "**", "*.png"), recursive=True)
        if len(png_files) > 1000:
            png_files = png_files[:1000]
        for slice_path in png_files:
            self.data_info.append((slice_path, 0, [0.0, 0.0, 0.0, 0.0]))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        slice_path, label, bbox = self.data_info[idx]
        try:
            slice_2d = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            if slice_2d is None:
                slice_2d = np.zeros((504, 504), dtype=np.uint8)
            slice_2d_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
            slice_tensor = self.transform(slice_2d_rgb).to(self.device)
            return slice_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(bbox, dtype=torch.float32), slice_path
        except Exception as e:
            return torch.zeros(3, 504, 504).to(self.device), torch.tensor(0, dtype=torch.long), torch.zeros(4, dtype=torch.float32).to(self.device), slice_path