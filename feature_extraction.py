import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from dataset import Luna16Dataset
from config import mhd_dir, slices_dir, candidates_file, transform, features_dir, device, batch_size
from sklearn.model_selection import train_test_split

# Load Pre-trained DINOv2 Model
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
dinov2_vitl14.eval()
# 利用预训练的 DINOv2 深度学习模型，从 LUNA16 肺部 CT 扫描图像中提取高维特征向量，并将这些特征保存为 .npy 文件，以便后续用于训练分类器

# Load Candidates
df_candidates = pd.read_csv(candidates_file)

# 3. 🔑 按病人 ID (SeriesUID) 划分数据
print("Splitting data by Patient ID (SeriesUID)...")
unique_patients = df_candidates['seriesuid'].unique()

# 70% Train, 30% (Val+Test)
train_patients, temp_patients = train_test_split(
    unique_patients, 
    test_size=0.3, 
    random_state=42, 
    shuffle=True
)

# 50% of 30% -> 15% Val, 15% Test
val_patients, test_patients = train_test_split(
    temp_patients, 
    test_size=0.5, 
    random_state=42, 
    shuffle=True
)

print(f"Total Patients: {len(unique_patients)}")
print(f"Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

# 4. 初始化三个独立的数据集 (传入 patient_ids)
# 注意：正式运行时去掉 max_slices 或设为 None
train_dataset = Luna16Dataset(
    mhd_dir, slices_dir, df_candidates, transform, dinov2_vitl14, device, 
    patient_ids=train_patients, 
    max_slices=None # 正式跑全量设为 None
)

val_dataset = Luna16Dataset(
    mhd_dir, slices_dir, df_candidates, transform, dinov2_vitl14, device, 
    patient_ids=val_patients, 
    max_slices=None
)

test_dataset = Luna16Dataset(
    mhd_dir, slices_dir, df_candidates, transform, dinov2_vitl14, device, 
    patient_ids=test_patients, 
    max_slices=None
)



# Load Dataset
# dataset = Luna16Dataset(mhd_dir, slices_dir, df_candidates, transform, dinov2_vitl14, device) #, max_slices=2000)
# train_size = int(0.7 * len(dataset))
# val_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)                 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                         

# Function to extract and save features
def extract_features(loader, split_name):
    features = []
    labels = []
    bboxes = []
    slice_paths = []
    with torch.no_grad():
        for slice_tensors, label_batch, bbox_batch, paths in tqdm(loader, desc=f"Extracting features for {split_name}"):
            slice_tensors = slice_tensors.to(device)
            feature_batch = dinov2_vitl14(slice_tensors).cpu().numpy()
            features.append(feature_batch)
            labels.append(label_batch.numpy())
            bboxes.append(bbox_batch.numpy())
            slice_paths.extend(paths)
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    bboxes = np.concatenate(bboxes, axis=0)
    np.save(os.path.join(features_dir, f"{split_name}_features.npy"), features)
    np.save(os.path.join(features_dir, f"{split_name}_labels.npy"), labels)
    np.save(os.path.join(features_dir, f"{split_name}_bboxes.npy"), bboxes)
    np.save(os.path.join(features_dir, f"{split_name}_paths.npy"), np.array(slice_paths))
    print(f"Saved {split_name} features: {features.shape}, labels: {labels.shape}, bboxes: {bboxes.shape}")

# Extract features for all splits
extract_features(train_loader, "train")
extract_features(val_loader, "val")
extract_features(test_loader, "test")