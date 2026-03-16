import os
import json
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

# 🔑 关键设置：必须在导入 torch 后，且在定义 main 函数之前设置启动方法
# 这解决了 "Cannot re-initialize CUDA in forked subprocess" 错误
multiprocessing.set_start_method('spawn', force=True)

# 导入自定义模块 (确保这些文件在同一个目录下或在 PYTHONPATH 中)
from model import DinoDetector
from dataset import Luna16Dataset
from config import (
    mhd_dir, slices_dir, candidates_file, transform, device, 
    batch_size, learning_rate, weight_decay, num_epochs, 
    checkpoint_dir, annotated_dir, features_dir
)

# ==========================================
# 主程序入口函数
# ==========================================
def main():
    # 确保输出目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    # 设置 matplotlib 支持中文 (防止乱码)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False

    print("="*60)
    print("🚀 Starting DINOv2 Feature Extraction & Classifier Training")
    print("="*60)

    # ==========================================
    # 1. 初始化日志系统
    # ==========================================
    start_time_global = time.time()
    training_log = {
        "start_time": datetime.datetime.now().isoformat(),
        "config": {
            "device": str(device),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "model_architecture": "DinoDetector (ViT-L/14 Backbone)",
            "data_split_strategy": "Patient-Level (No Data Leakage)",
            "num_workers": 4  # 记录使用的 worker 数量
        },
        "epoch_history": [],
        "best_model_info": {},
        "pytorch_test_results": {},
        "classical_models_results": {},
        "end_time": None,
        "total_training_duration_seconds": None
    }

    # ==========================================
    # 2. 加载 DINOv2 模型与计算特征维度
    # ==========================================
    print("\n1. Loading DINOv2 Backbone...")
    # 注意：这里加载到 device，但在 spawn 模式下，子进程不会继承这个上下文，
    # 子进程只会加载 Dataset 逻辑，模型推理主要在 main 进程 (GPU) 进行。
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    dinov2_vitl14.eval()

    # 计算特征维度
    dummy_slice = np.zeros((504, 504, 3), dtype=np.uint8)
    dummy_tensor = transform(dummy_slice).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = dinov2_vitl14(dummy_tensor)
        if len(output.shape) == 3:
            feature_vec = output[:, 0, :] # CLS Token
        else:
            feature_vec = output.flatten()
        
    feature_dim = feature_vec.shape[1] if len(feature_vec.shape) == 2 else feature_vec.shape[0]
    print(f"✅ Feature dimension determined: {feature_dim}")

    # ==========================================
    # 3. 加载数据并按病人划分
    # ==========================================
    print("\n2. Loading Dataset and Splitting by Patient ID...")
    df_candidates = pd.read_csv(candidates_file)
    unique_patients = df_candidates['seriesuid'].unique()
    print(f"   Total unique patients: {len(unique_patients)}")

    # 按 7:1.5:1.5 比例划分病人
    train_p, temp_p = train_test_split(unique_patients, test_size=0.3, random_state=42)
    val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

    print(f"   Train patients: {len(train_p)}, Val patients: {len(val_p)}, Test patients: {len(test_p)}")

    # 实例化完整数据集
    # 注意：Dataset 内部不要做 .to(device) 操作，只返回 CPU Tensor
    full_dataset = Luna16Dataset(
        mhd_dir, slices_dir, df_candidates, transform, 
        dino_model=None, 
        device='cpu', # 强制 dataset 使用 cpu，避免在 init 时初始化 cuda
        patient_ids=None 
    )

    # 构建索引映射辅助函数
    def get_indices_for_patients(dataset, patient_ids):
        indices = []
        patient_set = set(patient_ids)
        for idx, (path, _, _) in enumerate(dataset.data_info):
            fname = os.path.basename(path)
            # 假设文件名格式为 {uid}_{z}.png 或类似，取前缀作为 uid
            uid = fname.rsplit('_', 1)[0]
            if uid in patient_set:
                indices.append(idx)
        return indices

    train_indices = get_indices_for_patients(full_dataset, train_p)
    val_indices = get_indices_for_patients(full_dataset, val_p)
    test_indices = get_indices_for_patients(full_dataset, test_p)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # 🔑 关键：设置 num_workers > 0 以启用多进程加速
    # spawn 模式下，num_workers=4 是常见稳妥选择
    num_workers = 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,      # 加速 CPU->GPU 传输
        persistent_workers=True # 避免每个 epoch 重启 worker，进一步提升速度
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )

    print(f"✅ Data Slices - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ==========================================
    # 4. 初始化分类模型
    # ==========================================
    model = DinoDetector(feature_dim).to(device)
    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ==========================================
    # 5. 训练循环
    # ==========================================
    print(f"\n3. Training for {num_epochs} epochs...")

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "dino_detector_best.pth")

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for slice_tensors, labels, bboxes, _ in pbar:
            # 🔑 关键：在循环内部将数据移动到 GPU
            slice_tensors = slice_tensors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 在线提取特征 (GPU)
            with torch.no_grad():
                features_raw = dinov2_vitl14(slice_tensors)
                if len(features_raw.shape) == 3:
                    features = features_raw[:, 0, :] # CLS Token
                else:
                    features = features_raw.flatten(1)
            
            class_output, bbox_output = model(features)
            
            loss_class = criterion_class(class_output, labels)
            mask = labels == 1
            if mask.sum() > 0:
                loss_bbox = criterion_bbox(bbox_output[mask], bboxes[mask])
            else:
                loss_bbox = torch.tensor(0.0).to(device)
                
            loss = loss_class + loss_bbox
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(class_output, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # --- Validate ---
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for slice_tensors, labels, bboxes, _ in val_loader:
                slice_tensors = slice_tensors.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bboxes = bboxes.to(device, non_blocking=True)
                
                features_raw = dinov2_vitl14(slice_tensors)
                if len(features_raw.shape) == 3:
                    features = features_raw[:, 0, :]
                else:
                    features = features_raw.flatten(1)
                
                class_output, bbox_output = model(features)
                
                loss_class = criterion_class(class_output, labels)
                mask = labels == 1
                loss_bbox = criterion_bbox(bbox_output[mask], bboxes[mask]) if mask.sum() > 0 else torch.tensor(0.0).to(device)
                loss = loss_class + loss_bbox
                
                total_val_loss += loss.item()
                _, predicted = torch.max(class_output, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 更新日志
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(avg_val_loss),
            "val_accuracy": float(val_acc),
            "lr": optimizer.param_groups[0]['lr']
        }
        training_log["epoch_history"].append(epoch_log)
        
        print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            training_log["best_model_info"] = {
                "epoch": epoch + 1,
                "val_accuracy": float(best_val_acc),
                "model_path": best_model_path
            }
            print(f"💾 New best model saved at Epoch {epoch+1}!")

    # ==========================================
    # 6. 测试集评估
    # ==========================================
    print("\n4. Evaluating on Test Set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for slice_tensors, labels, bboxes, _ in test_loader:
            slice_tensors = slice_tensors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            features_raw = dinov2_vitl14(slice_tensors)
            if len(features_raw.shape) == 3:
                features = features_raw[:, 0, :]
            else:
                features = features_raw.flatten(1)
            
            class_output, _ = model(features)
            probs = torch.softmax(class_output, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(class_output, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    pt_metrics = {
        "Accuracy": float(accuracy_score(all_labels, all_preds)),
        "Precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "Recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "F1-Score": float(f1_score(all_labels, all_preds, zero_division=0)),
        "AUC": float(roc_auc_score(all_labels, all_probs))
    }
    training_log["pytorch_test_results"] = pt_metrics

    print(f"PyTorch DinoDetector Test Results:")
    for k, v in pt_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ==========================================
    # 7. 经典机器学习模型对比
    # ==========================================
    print("\n5. Training Classical Classifiers for Comparison...")
    
    def extract_features_for_sklearn(loader):
        feats = []
        labs = []
        for slice_tensors, labels, _, _ in loader:
            slice_tensors = slice_tensors.to(device, non_blocking=True)
            with torch.no_grad():
                features_raw = dinov2_vitl14(slice_tensors)
                if len(features_raw.shape) == 3:
                    features = features_raw[:, 0, :].cpu().numpy()
                else:
                    features = features_raw.flatten(1).cpu().numpy()
            feats.append(features)
            labs.append(labels.numpy())
        return np.vstack(feats), np.hstack(labs)

    print("   Extracting features for Sklearn models...")
    train_feat_np, train_lab_np = extract_features_for_sklearn(train_loader)
    test_feat_np, test_lab_np = extract_features_for_sklearn(test_loader)

    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"   Training {name}...")
        clf.fit(train_feat_np, train_lab_np)
        preds = clf.predict(test_feat_np)
        probs = clf.predict_proba(test_feat_np)[:, 1] if hasattr(clf, "predict_proba") else preds
        
        acc = float(accuracy_score(test_lab_np, preds))
        prec = float(precision_score(test_lab_np, preds, zero_division=0))
        rec = float(recall_score(test_lab_np, preds, zero_division=0))
        f1 = float(f1_score(test_lab_np, preds, zero_division=0))
        auc = float(roc_auc_score(test_lab_np, probs))
        
        results[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}
        print(f"   ✅ {name} - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    training_log["classical_models_results"] = results

    # ==========================================
    # 8. 打印最终对比表格
    # ==========================================
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON ON TEST SET (Patient-Level Split)")
    print("="*80)
    print(f"| {'Model':<15} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'AUC':>10} |")
    print("-" * 80)
    row_pt = f"| {'DinoDetector':<15} | {pt_metrics['Accuracy']:>10.4f} | {pt_metrics['Precision']:>10.4f} | {pt_metrics['Recall']:>10.4f} | {pt_metrics['F1-Score']:>10.4f} | {pt_metrics['AUC']:>10.4f} |"
    print(row_pt)
    for name, m in results.items():
        row = f"| {name:<15} | {m['acc']:>10.4f} | {m['prec']:>10.4f} | {m['rec']:>10.4f} | {m['f1']:>10.4f} | {m['auc']:>10.4f} |"
        print(row)
    print("="*80)

    # ==========================================
    # 9. 生成训练曲线图
    # ==========================================
    print("\n6. Generating Training Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss', color='blue', marker='o')
    axes[0].plot(val_losses, label='Val Loss', color='orange', marker='s')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(train_accuracies, label='Train Accuracy', color='green', marker='o')
    axes[1].plot(val_accuracies, label='Val Accuracy', color='red', marker='s')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curve_save_path = os.path.join(checkpoint_dir, "training_curves.png")
    plt.savefig(curve_save_path, dpi=300)
    print(f"📈 Training curves saved to: {curve_save_path}")
    plt.close()

    # ==========================================
    # 10. 保存完整日志
    # ==========================================
    end_time_global = time.time()
    training_log["end_time"] = datetime.datetime.now().isoformat()
    training_log["total_training_duration_seconds"] = round(end_time_global - start_time_global, 2)

    log_save_path = os.path.join(checkpoint_dir, "training_log.json")
    with open(log_save_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=4, ensure_ascii=False)

    print(f"📝 Detailed training log saved to: {log_save_path}")
    print(f"⏱️ Total Training Time: {training_log['total_training_duration_seconds']} seconds")
    print("\n🎉 All Tasks Completed Successfully!")

# 🔑 关键入口保护
# 当使用 'spawn' 启动子进程时，子进程会重新导入此脚本。
# 如果没有这个判断，子进程会再次执行 main()，导致无限递归创建进程并报错。
# 只有直接运行脚本时 (__name__ == '__main__') 才执行 main()。
if __name__ == '__main__':
    main()