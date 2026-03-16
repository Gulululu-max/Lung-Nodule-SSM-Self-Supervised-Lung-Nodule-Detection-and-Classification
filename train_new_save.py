import os
import json
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示 (防止乱码，根据系统环境可能需要调整字体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

# 导入配置和模型定义
from model import DinoDetector
from config import device, batch_size, learning_rate, weight_decay, num_epochs, checkpoint_dir, features_dir

# 确保输出目录存在
os.makedirs(checkpoint_dir, exist_ok=True)

# 初始化日志字典
training_log = {
    "start_time": datetime.datetime.now().isoformat(),
    "config": {
        "device": str(device),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "feature_dim": None, # 稍后填充
        "model_architecture": "DinoDetector"
    },
    "epoch_history": [],
    "best_model_info": {},
    "pytorch_test_results": {},
    "classical_models_results": {},
    "end_time": None,
    "total_training_duration_seconds": None
}

print("="*50)
print("🚀 Starting Classifier Training on Pre-extracted Features")
print("="*50)

# 1. 加载预提取的特征 (.npy)
print("\n1. Loading pre-extracted features...")
try:
    train_features = np.load(os.path.join(features_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(features_dir, "train_labels.npy"))
    
    val_features = np.load(os.path.join(features_dir, "val_features.npy"))
    val_labels = np.load(os.path.join(features_dir, "val_labels.npy"))
    
    test_features = np.load(os.path.join(features_dir, "test_features.npy"))
    test_labels = np.load(os.path.join(features_dir, "test_labels.npy"))
    
    print(f"✅ Train: {train_features.shape}, Labels: {train_labels.shape}")
    print(f"✅ Val:   {val_features.shape}, Labels: {val_labels.shape}")
    print(f"✅ Test:  {test_features.shape}, Labels: {test_labels.shape}")
    
except FileNotFoundError as e:
    print(f"❌ Error: Feature files not found! Please run the feature extraction script first.\nDetails: {e}")
    exit()

# 2. 数据预处理
train_X = torch.tensor(train_features, dtype=torch.float32).to(device)
train_y = torch.tensor(train_labels, dtype=torch.long).to(device)

val_X = torch.tensor(val_features, dtype=torch.float32).to(device)
val_y = torch.tensor(val_labels, dtype=torch.long).to(device)

test_X = torch.tensor(test_features, dtype=torch.float32).to(device)
test_y = torch.tensor(test_labels, dtype=torch.long).to(device)

train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 获取特征维度
feature_dim = train_features.shape[1]
training_log["config"]["feature_dim"] = int(feature_dim)
print(f"\n2. Feature dimension detected: {feature_dim}")

# 4. 初始化 PyTorch 模型
print("\n3. Initializing PyTorch Classifier (DinoDetector)...")
model = DinoDetector(feature_dim).to(device)

criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 5. 训练循环
print(f"\n4. Training for {num_epochs} epochs...")
best_val_acc = 0.0
train_losses = []
val_losses = [] # 新增：记录验证集损失
train_accuracies = []
val_accuracies = []

# 记录训练开始时间
start_time = time.time()

for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in pbar:
        optimizer.zero_grad()
        
        class_output, _ = model(inputs)
        
        loss = criterion_class(class_output, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        _, predicted = torch.max(class_output, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = train_correct / train_total
    
    # --- Validate ---
    model.eval()
    total_val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            class_output, _ = model(inputs)
            loss = criterion_class(class_output, labels)
            
            total_val_loss += loss.item()
            _, predicted = torch.max(class_output, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = val_correct / val_total
    
    # 记录历史数据用于绘图和日志
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # 构建当前 epoch 的日志条目
    epoch_log = {
        "epoch": epoch + 1,
        "train_loss": float(avg_train_loss),
        "train_accuracy": float(train_acc),
        "val_loss": float(avg_val_loss),
        "val_accuracy": float(val_acc),
        "current_lr": optimizer.param_groups[0]['lr']
    }
    training_log["epoch_history"].append(epoch_log)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 学习率调整
    scheduler.step(avg_val_loss)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(checkpoint_dir, "dino_detector_best.pth")
        torch.save(model.state_dict(), best_model_path)
        
        # 更新最佳模型日志
        training_log["best_model_info"] = {
            "epoch": epoch + 1,
            "val_accuracy": float(best_val_acc),
            "val_loss": float(avg_val_loss),
            "model_path": best_model_path
        }
        print(f"💾 New best model saved with Val Acc: {best_val_acc:.4f}")

# 计算训练总耗时
end_time = time.time()
total_duration = end_time - start_time
training_log["end_time"] = datetime.datetime.now().isoformat()
training_log["total_training_duration_seconds"] = round(total_duration, 2)

print("\n✅ PyTorch Model Training Finished!")
print(f"⏱️ Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")

# 6. 在测试集上评估 PyTorch 模型
print("\n5. Evaluating Best PyTorch Model on Test Set...")
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "dino_detector_best.pth"), map_location=device))
model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        class_output, _ = model(inputs)
        probs = torch.softmax(class_output, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(class_output, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

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

# 7. 对比经典机器学习模型
print("\n6. Training & Evaluating Classical Classifiers for Comparison...")
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results = {}
for name, clf in classifiers.items():
    print(f"   Training {name}...")
    clf.fit(train_features, train_labels)
    
    test_preds = clf.predict(test_features)
    if hasattr(clf, "predict_proba"):
        test_probs = clf.predict_proba(test_features)[:, 1]
    else:
        test_probs = test_preds
    
    acc = float(accuracy_score(test_labels, test_preds))
    prec = float(precision_score(test_labels, test_preds, zero_division=0))
    rec = float(recall_score(test_labels, test_preds, zero_division=0))
    f1 = float(f1_score(test_labels, test_preds, zero_division=0))
    auc = float(roc_auc_score(test_labels, test_probs))
    
    results[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}
    print(f"   ✅ {name} - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

training_log["classical_models_results"] = results

# 8. 打印最终对比表格
print("\n" + "="*80)
print("🏆 FINAL COMPARISON ON TEST SET (Patient-Level Split)")
print("="*80)
print(f"| {'Model':<15} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'AUC':>10} |")
print("-" * 80)

row_pt = f"| {'DinoDetector':<15} | {pt_metrics['Accuracy']:>10.4f} | {pt_metrics['Precision']:>10.4f} | {pt_metrics['Recall']:>10.4f} | {pt_metrics['F1-Score']:>10.4f} | {pt_metrics['AUC']:>10.4f} |"
print(row_pt)

for name, metrics in results.items():
    row = f"| {name:<15} | {metrics['acc']:>10.4f} | {metrics['prec']:>10.4f} | {metrics['rec']:>10.4f} | {metrics['f1']:>10.4f} | {metrics['auc']:>10.4f} |"
    print(row)

print("="*80)
print("🎉 Training and Evaluation Complete!")

# ==========================================
# 9. 生成训练曲线图
# ==========================================
print("\n7. Generating Training Curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图 1: Loss 曲线
axes[0].plot(train_losses, label='Train Loss', color='blue', marker='o')
axes[0].plot(val_losses, label='Val Loss', color='orange', marker='s')
axes[0].set_title('Training & Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# 子图 2: Accuracy 曲线
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
plt.close() # 释放内存

# ==========================================
# 10. 保存完整训练日志 (JSON)
# ==========================================
log_save_path = os.path.join(checkpoint_dir, "training_log.json")
with open(log_save_path, 'w', encoding='utf-8') as f:
    json.dump(training_log, f, indent=4, ensure_ascii=False)

print(f"📝 Detailed training log saved to: {log_save_path}")
print(f"Best PyTorch model saved at: {os.path.join(checkpoint_dir, 'dino_detector_best.pth')}")