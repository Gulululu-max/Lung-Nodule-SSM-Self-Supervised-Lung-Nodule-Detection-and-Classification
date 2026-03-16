cuda 11.8 
python 3.10


# def __init__(self, mhd_dir, slices_dir, candidates_df, transform, dino_model, device):
    #     self.mhd_dir = mhd_dir
    #     self.slices_dir = slices_dir
    #     self.candidates_df = candidates_df
    #     self.transform = transform
    #     self.dino_model = dino_model
    #     self.device = device
    #     self.data_info = []
    #     self.slice_counts = {}

    #     print("Loading dataset... This may take some time.")
        
    #     # 预先构建 candidates 映射，加速查找: {seriesuid: [z_index_list]}
    #     # 注意：这里暂时无法精确计算 z_index，因为缺少 spacing。
    #     # 策略：先记录该病人有哪些结节坐标，稍后在读取切片时尝试匹配，
    #     # 或者简单地将该病人所有切片标记为潜在阳性（如果只需弱标签）。
    #     # 为了严谨，我们保留获取 metadata 的逻辑来转换坐标。
    #     candidate_map = {}
    #     for _, row in candidates_df.iterrows():
    #         uid = row['seriesuid']
    #         if uid not in candidate_map:
    #             candidate_map[uid] = []
    #         candidate_map[uid].append(row) # 存储整行数据以便后续计算

    #     for subset in range(10):
    #         subset_slices_path = os.path.join(self.slices_dir, f"subset{subset}")
    #         if not os.path.exists(subset_slices_path):
    #             print(f"⚠️ Warning: Subset directory not found: {subset_slices_path}")
    #             continue

    #         # ✅ 修改点 1: 直接获取该目录下所有 PNG 文件，而不是找子文件夹
    #         all_png_files = [f for f in os.listdir(subset_slices_path) if f.endswith('.png')]
            
    #         if not all_png_files:
    #             print(f"   No PNG files found in {subset_slices_path}")
    #             continue

    #         print(f"   Processing {subset}: {len(all_png_files)} files found")

    #         # 按 seriesuid 分组文件，避免重复读取 metadata
    #         # 结构: {seriesuid: ['file1.png', 'file2.png', ...]}
    #         files_by_uid = {}
    #         for fname in all_png_files:
    #             try:
    #                 # 假设文件名格式: {seriesuid}_{z}.png
    #                 # 从右边分割一次，分离出 _z.png
    #                 parts = fname.rsplit('_', 1)
    #                 if len(parts) != 2:
    #                     continue 
    #                 uid = parts[0]
    #                 if uid not in files_by_uid:
    #                     files_by_uid[uid] = []
    #                 files_by_uid[uid].append(fname)
    #             except Exception:
    #                 continue

    #         # 遍历每个病人
    #         for seriesuid, file_list in tqdm(files_by_uid.items(), desc=f"Indexing subset{subset}"):
    #             # 1. 获取 Metadata (用于坐标转换)
    #             origin, spacing = get_metadata_or_default(self.mhd_dir, seriesuid)
                
    #             # 2. 计算该病人的结节所在切片索引 (Z-index)
    #             target_z_indices = set()
    #             if seriesuid in candidate_map:
    #                 for row in candidate_map[seriesuid]:
    #                     try:
    #                         # 将物理坐标转换为切片索引
    #                         # coordZ 是物理坐标，origin[2] 是起始位置，spacing[2] 是层厚
    #                         # 注意：SimpleITK 的 GetOrigin/Spacing 顺序可能与 CSV 不同，
    #                         # 你的原代码用了 [::-1] 反转，这里保持一致
    #                         c_z = row["coordZ"]
    #                         o_z = origin[2]
    #                         s_z = spacing[2]
                            
    #                         if s_z == 0: s_z = 1.0 # 防止除以零
                            
    #                         z_idx = int(np.rint((c_z - o_z) / s_z))
    #                         target_z_indices.add(z_idx)
    #                     except Exception as e:
    #                         # print(f"Error calculating Z for {seriesuid}: {e}")
    #                         pass

    #             self.slice_counts[seriesuid] = len(file_list)

    #             # 3. 遍历该病人的所有切片文件
    #             for slice_file in file_list:
    #                 try:
    #                     # 解析当前切片的 Z 索引
    #                     # 文件名: {uid}_{z}.png -> 取最后一部分去掉 .png
    #                     z_str = slice_file.rsplit('_', 1)[1].replace('.png', '')
    #                     z = int(z_str)
    #                 except (IndexError, ValueError):
    #                     continue

    #                 slice_path = os.path.join(subset_slices_path, slice_file)
                    
    #                 # 判断标签
    #                 label = 1 if z in target_z_indices else 0
                    
    #                 # 如果是阳性，尝试推断 bbox (阴性则跳过以节省时间，或给默认值)
    #                 bbox = [0, 0, 0, 0]
    #                 if label == 1:
    #                     # 注意：infer_bbox_from_features 比较耗时，如果数据量大可能会慢
    #                     # 如果只是为了跑通，可以先给默认值，或者只在训练时动态计算
    #                     bbox = self.infer_bbox_from_features(slice_path)

    #                 self.data_info.append((slice_path, label, bbox))

    #     print(f"✅ Loaded {len(self.data_info)} slices.")
    #     if len(self.data_info) == 0:
    #         print("WARNING: No valid slices found! Attempting fallback...")
    #         self._load_fallback_data()