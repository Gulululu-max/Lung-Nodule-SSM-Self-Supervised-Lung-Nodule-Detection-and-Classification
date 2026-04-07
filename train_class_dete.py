import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 导入自定义模块
from model import DinoDetector
from config import device, batch_size, learning_rate, weight_decay, num_epochs, checkpoint_dir, features_dir, annotated_dir

# ================= 0. 初始化与配置 =================
start_time_total = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(checkpoint_dir, f"run_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)

# 定义日志文件路径
log_csv_path = os.path.join(log_dir, "training_log.csv")
log_json_path = os.path.join(log_dir, "training_config.json")
report_json_path = os.path.join(log_dir, "final_report.json")
curve_img_path = os.path.join(log_dir, "training_curves.png")

print("="*70)
print(f"🚀 Starting Professional Training Session: {timestamp}")
print("="*70)

# 记录所有超参数和环境信息
config_info = {
    "timestamp": timestamp,
    "device": str(device),
    "hyperparameters": {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "feature_dim": "auto_detected"
    },
    "paths": {
        "features_dir": features_dir,
        "checkpoint_dir": log_dir,
        "annotated_dir": annotated_dir
    }
}

# ================= 1. 数据集加载 (带路径) =================
class FastFeatureDataset(Dataset):
    def __init__(self, feat_path, label_path, bbox_path, path_list_file):
        self.features = np.load(feat_path)
        self.labels = np.load(label_path)
        self.bboxes = np.load(bbox_path)
        
        if os.path.exists(path_list_file):
            self.image_paths = np.load(path_list_file, allow_pickle=True)
        else:
            print(f"⚠️ Warning: Path file {path_list_file} missing. Visualization disabled.")
            self.image_paths = [None] * len(self.features)
            
        assert len(self.features) == len(self.labels), "Feature and Label length mismatch!"
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.bboxes[idx], dtype=torch.float32),
            self.image_paths[idx]
        )

try:
    train_ds = FastFeatureDataset(
        os.path.join(features_dir, "train_features.npy"),
        os.path.join(features_dir, "train_labels.npy"),
        os.path.join(features_dir, "train_bboxes.npy"),
        os.path.join(features_dir, "train_paths.npy")
    )
    val_ds = FastFeatureDataset(
        os.path.join(features_dir, "val_features.npy"),
        os.path.join(features_dir, "val_labels.npy"),
        os.path.join(features_dir, "val_bboxes.npy"),
        os.path.join(features_dir, "val_paths.npy")
    )
    test_ds = FastFeatureDataset(
        os.path.join(features_dir, "test_features.npy"),
        os.path.join(features_dir, "test_labels.npy"),
        os.path.join(features_dir, "test_bboxes.npy"),
        os.path.join(features_dir, "test_paths.npy")
    )
    config_info["data_stats"] = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds)
    }
except Exception as e:
    print(f"❌ Critical Error loading data: {e}")
    sys.exit(1)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)#, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)#, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

feature_dim = train_ds.features.shape[1]
config_info["hyperparameters"]["feature_dim"] = feature_dim
print(f"✅ Data Loaded. Feature Dim: {feature_dim}")

# ================= 2. 模型与优化器 =================
model = DinoDetector(feature_dim).to(device)
criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 保存初始配置
with open(log_json_path, 'w') as f:
    json.dump(config_info, f, indent=4, default=str)
print(f"💾 Config saved to: {log_json_path}")

# ================= 3. 辅助函数：画图 =================
def draw_bboxes_on_slice(slice_path, bbox_pred, bbox_gt, output_path):
    if not slice_path or not os.path.exists(slice_path):
        return False
    img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    
    # 假设 bbox 是归一化的 [x, y, w, h]
    # 预测框 (绿色, 粗)
    xp, yp, wp, hp = bbox_pred
    cv2.rectangle(img_rgb, (int(xp*w), int(yp*h)), (int((xp+wp)*w), int((yp+hp)*h)), (0, 255, 0), 2)
    
    # 真值框 (红色, 细)
    xg, yg, wg, hg = bbox_gt
    cv2.rectangle(img_rgb, (int(xg*w), int(yg*h)), (int((xg+wg)*w), int((yg+hg)*h)), (0, 0, 255), 1)
    
    cv2.imwrite(output_path, img_rgb)
    return True

# ================= 4. 训练循环 (带详细日志) =================
history = {
    "epoch": [],
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [],
    "lr": [],
    "time_per_epoch": []
}

best_val_loss = float('inf')
best_epoch = 0
vis_count_total = 0

print(f"\n🏋️ Training Start (Max Epochs: {num_epochs})...")
print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<12} | {'Val Acc':<10} | {'Time':<8} | {'LR':<10}")
print("-" * 90)

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # --- Train ---
    model.train()
    t_loss_sum, t_correct, t_total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
    for feats, labels, bboxes, paths in pbar:
        feats, labels, bboxes = feats.to(device), labels.to(device), bboxes.to(device)
        
        optimizer.zero_grad()
        cls_out, bbox_out = model(feats)
        
        loss_cls = criterion_class(cls_out, labels)
        mask = (labels == 1)
        loss_bbox = criterion_bbox(bbox_out[mask], bboxes[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=device)
        loss = loss_cls + loss_bbox
        
        loss.backward()
        optimizer.step()
        
        t_loss_sum += loss.item()
        _, preds = torch.max(cls_out, 1)
        t_correct += (preds == labels).sum().item()
        t_total += labels.size(0)
        
        pbar.set_postfix({"t_loss": f"{loss.item():.4f}"})

    avg_t_loss = t_loss_sum / len(train_loader)
    t_acc = t_correct / t_total if t_total > 0 else 0
    
    # --- Validate ---
    model.eval()
    v_loss_sum, v_correct, v_total = 0.0, 0, 0
    vis_count_epoch = 0
    
    with torch.no_grad():
        for feats, labels, bboxes, paths in val_loader:
            feats, labels, bboxes = feats.to(device), labels.to(device), bboxes.to(device)
            
            cls_out, bbox_out = model(feats)
            loss_cls = criterion_class(cls_out, labels)
            mask = (labels == 1)
            loss_bbox = criterion_bbox(bbox_out[mask], bboxes[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=device)
            loss = loss_cls + loss_bbox
            
            v_loss_sum += loss.item()
            _, preds = torch.max(cls_out, 1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)
            
            # 可视化：每个 epoch 只画前 3 个异常样本
            if vis_count_epoch < 3 and vis_count_total < 20:
                for i in range(len(preds)):
                    if vis_count_epoch >= 3: break
                    if preds[i] == 1 or labels[i] == 1:
                        if paths[i]:
                            fname = f"ep{epoch+1}_{os.path.basename(paths[i])}"
                            if draw_bboxes_on_slice(paths[i], bbox_out[i].cpu().numpy(), bboxes[i].cpu().numpy(), os.path.join(annotated_dir, fname)):
                                vis_count_epoch += 1
                                vis_count_total += 1

    avg_v_loss = v_loss_sum / len(val_loader)
    v_acc = v_correct / v_total if v_total > 0 else 0
    
    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]['lr']
    
    # 更新历史记录
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(avg_t_loss)
    history["train_acc"].append(t_acc)
    history["val_loss"].append(avg_v_loss)
    history["val_acc"].append(v_acc)
    history["lr"].append(current_lr)
    history["time_per_epoch"].append(epoch_time)
    
    # 打印进度
    print(f"{epoch+1:<6} | {avg_t_loss:<12.4f} | {t_acc:<10.4f} | {avg_v_loss:<12.4f} | {v_acc:<10.4f} | {epoch_time:<8.1f}s | {current_lr:<10.2e}")
    
    # 调度器步长
    scheduler.step(avg_v_loss)
    
    # 保存最佳模型
    if avg_v_loss < best_val_loss:
        best_val_loss = avg_v_loss
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_v_loss,
            'config': config_info
        }, os.path.join(log_dir, "best_model.pth"))

# ================= 5. 保存日志与绘图 =================
# 保存 CSV
df_history = pd.DataFrame(history)
df_history.to_csv(log_csv_path, index=False)
print(f"\n📊 Training Log saved to: {log_csv_path}")

# 绘制曲线
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss Curve
ax1.plot(history["epoch"], history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
ax1.plot(history["epoch"], history["val_loss"], 'r-s', label='Val Loss', linewidth=2)
ax1.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Accuracy Curve
ax2.plot(history["epoch"], history["train_acc"], 'b-o', label='Train Acc', linewidth=2)
ax2.plot(history["epoch"], history["val_acc"], 'r-s', label='Val Acc', linewidth=2)
ax2.set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(curve_img_path, dpi=300)
plt.close()
print(f"📈 Training Curves saved to: {curve_img_path}")

# ================= 6. 测试集评估 =================
print("\n🧪 Evaluating Best Model on Test Set...")
checkpoint = torch.load(os.path.join(log_dir, "best_model.pth"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for feats, labels, _, _ in test_loader:
        feats = feats.to(device)
        cls_out, _ = model(feats)
        probs = torch.softmax(cls_out, 1)[:, 1].cpu().numpy()
        preds = torch.argmax(cls_out, 1).cpu().numpy()
        
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

all_preds, all_probs, all_labels = np.array(all_preds), np.array(all_probs), np.array(all_labels)

test_metrics = {
    "Accuracy": float(accuracy_score(all_labels, all_preds)),
    "Precision": float(precision_score(all_labels, all_preds, zero_division=0)),
    "Recall": float(recall_score(all_labels, all_preds, zero_division=0)),
    "F1_Score": float(f1_score(all_labels, all_preds, zero_division=0)),
    "AUC": float(roc_auc_score(all_labels, all_probs))
}

# ================= 6.1 额外分析图表 =================
def plot_additional_metrics(all_labels, all_probs, all_preds, features, log_dir):
    """绘制额外的分析图表"""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. 混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Nodule', 'Nodule'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("  ✅ confusion_matrix.png")
    
    # 2. ROC 曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    print("  ✅ roc_curve.png")
    
    # 3. Precision-Recall 曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    ax.plot(recall, precision, 'r-', linewidth=2, label=f'PR (AP = {ap:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'pr_curve.png'), dpi=150)
    plt.close()
    print("  ✅ pr_curve.png")
    
    # 4. 预测概率分布直方图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_probs[all_labels == 0], bins=50, alpha=0.6, label='Non-Nodule', color='blue')
    ax.hist(all_probs[all_labels == 1], bins=50, alpha=0.6, label='Nodule', color='red')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'probability_distribution.png'), dpi=150)
    plt.close()
    print("  ✅ probability_distribution.png")
    
    # 5. t-SNE 特征可视化 (采样以加快速度)
    if len(features) > 1000:
        # 采样
        np.random.seed(42)
        sample_idx = np.random.choice(len(features), 1000, replace=False)
        features_sample = features[sample_idx]
        labels_sample = all_labels[sample_idx]
    else:
        features_sample = features
        labels_sample = all_labels
    
    print("  ⏳ Computing t-SNE (may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sample)-1))
    features_2d = tsne.fit_transform(features_sample)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=labels_sample, cmap='coolwarm', alpha=0.6, s=20)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Label (0: Non-Nodule, 1: Nodule)', fontsize=10)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'tsne_visualization.png'), dpi=150)
    plt.close()
    print("  ✅ tsne_visualization.png")
    
    # 6. 类别分布统计
    fig, ax = plt.subplots(figsize=(6, 4))
    unique, counts = np.unique(all_labels, return_counts=True)
    bars = ax.bar(['Non-Nodule', 'Nodule'], counts, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
               str(count), ha='center', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    print("  ✅ class_distribution.png")

# 调用函数生成额外图表
print("\n📊 Generating additional analysis plots...")
plot_additional_metrics(all_labels, all_probs, all_preds, test_ds.features, log_dir)

total_training_time = time.time() - start_time_total

# ================= 7. 经典模型对比 =================
print("\n🤖 Running Classical Classifiers for Comparison...")
clf_results = {}
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

for name, clf in classifiers.items():
    t0 = time.time()
    clf.fit(train_ds.features, train_ds.labels)
    preds = clf.predict(test_ds.features)
    probs = clf.predict_proba(test_ds.features)[:, 1] if hasattr(clf, 'predict_proba') else preds
    t1 = time.time()
    
    clf_results[name] = {
        "Accuracy": float(accuracy_score(test_ds.labels, preds)),
        "Precision": float(precision_score(test_ds.labels, preds, zero_division=0)),
        "Recall": float(recall_score(test_ds.labels, preds, zero_division=0)),
        "F1_Score": float(f1_score(test_ds.labels, preds, zero_division=0)),
        "AUC": float(roc_auc_score(test_ds.labels, probs)),
        "Training_Time": round(t1-t0, 2)
    }

# ================= 8. 生成最终报告 =================
final_report = {
    "summary": {
        "total_training_time_seconds": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "visualizations_generated": vis_count_total
    },
    "pytorch_model_test_metrics": test_metrics,
    "classical_models_comparison": clf_results,
    "hyperparameters_used": config_info["hyperparameters"]
}

with open(report_json_path, 'w') as f:
    json.dump(final_report, f, indent=4)

# 打印最终表格
print("\n" + "="*70)
print("🏆 FINAL RESULTS REPORT")
print("="*70)
print(f"Total Training Time: {total_training_time:.2f}s | Best Epoch: {best_epoch}")
print(f"Test Metrics (PyTorch): Acc={test_metrics['Accuracy']:.4f}, F1={test_metrics['F1_Score']:.4f}, AUC={test_metrics['AUC']:.4f}")
print("\nComparison Table:")
print(f"| {'Model':<15} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8} | {'F1':>8} | {'AUC':>8} | {'Time(s)':>8} |")
print("-" * 85)
print(f"| {'DinoDetector':<15} | {test_metrics['Accuracy']:>8.4f} | {test_metrics['Precision']:>8.4f} | {test_metrics['Recall']:>8.4f} | {test_metrics['F1_Score']:>8.4f} | {test_metrics['AUC']:>8.4f} | {'N/A':>8} |")
for name, res in clf_results.items():
    print(f"| {name:<15} | {res['Accuracy']:>8.4f} | {res['Precision']:>8.4f} | {res['Recall']:>8.4f} | {res['F1_Score']:>8.4f} | {res['AUC']:>8.4f} | {res['Training_Time']:>8.2f} |")

print(f"\n✅ All artifacts saved in: {log_dir}")
print("   - training_log.csv (Detailed epoch-by-epoch metrics)")
print("   - training_config.json (Hyperparameters)")
print("   - final_report.json (Final results & comparison)")
print("   - training_curves.png (Visual plots)")
print("   - best_model.pth (Weights)")
print("="*70)