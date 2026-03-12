import os
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

# 导入配置和模型定义
from model import DinoDetector  # 假设你的 DinoDetector 定义在这里
from config import device, batch_size, learning_rate, weight_decay, num_epochs, checkpoint_dir, features_dir

print("="*50)
print("🚀 Starting Classifier Training on Pre-extracted Features")
print("="*50)

# 1. 加载预提取的特征 (.npy)
print("\n1. Loading pre-extracted features...")
try:
    train_features = np.load(os.path.join(features_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(features_dir, "train_labels.npy"))
    # bboxes 如果需要训练 bbox 回归头可以加载，如果只训练分类器可以暂时不用，或者作为辅助
    # train_bboxes = np.load(os.path.join(features_dir, "train_bboxes.npy")) 
    
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

# 2. 数据预处理 (转换为 PyTorch Tensor)
# DINOv2 输出的特征通常已经是 float32，不需要额外归一化，但可以转为 Tensor
train_X = torch.tensor(train_features, dtype=torch.float32).to(device)
train_y = torch.tensor(train_labels, dtype=torch.long).to(device)

val_X = torch.tensor(val_features, dtype=torch.float32).to(device)
val_y = torch.tensor(val_labels, dtype=torch.long).to(device)

test_X = torch.tensor(test_features, dtype=torch.float32).to(device)
test_y = torch.tensor(test_labels, dtype=torch.long).to(device)

# 创建 TensorDataset 和 DataLoader (现在速度会极快，因为只是内存拷贝)
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 获取特征维度
feature_dim = train_features.shape[1]
print(f"\n2. Feature dimension detected: {feature_dim}")

# 4. 初始化 PyTorch 模型 (DinoDetector)
print("\n3. Initializing PyTorch Classifier (DinoDetector)...")
model = DinoDetector(feature_dim).to(device)

# 损失函数和优化器
# 注意：如果你之前提取特征时没有包含有效的 BBox (全是0)，建议先只训练分类头
# 这里假设你只想做分类任务，或者 BBox 任务在特征提取阶段已经通过某种方式处理了
# 如果必须训练 BBox，请确保 train_bboxes 已加载且有效
criterion_class = nn.CrossEntropyLoss()
# criterion_bbox = nn.MSELoss() # 暂时注释掉，除非你确定要联合训练且 BBox 数据有效

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 5. 训练循环 (仅基于特征)
print(f"\n4. Training for {num_epochs} epochs...")
best_val_acc = 0.0
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in pbar:
        # inputs 已经是特征向量了，不需要再过 DINO 模型！
        optimizer.zero_grad()
        
        class_output, _ = model(inputs) # 假设 model 返回 (class_out, bbox_out)
        
        loss = criterion_class(class_output, labels)
        # 如果有 BBox 任务且数据有效，可以在这里加上 loss_bbox
        
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
    
    train_losses.append(avg_train_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 学习率调整
    scheduler.step(avg_val_loss)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, "dino_detector_best.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 New best model saved with Val Acc: {best_val_acc:.4f}")

print("\n✅ PyTorch Model Training Finished!")

# 6. 在测试集上评估 PyTorch 模型
print("\n5. Evaluating Best PyTorch Model on Test Set...")
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "dino_detector_best.pth")))
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
    "Accuracy": accuracy_score(all_labels, all_preds),
    "Precision": precision_score(all_labels, all_preds, zero_division=0),
    "Recall": recall_score(all_labels, all_preds, zero_division=0),
    "F1-Score": f1_score(all_labels, all_preds, zero_division=0),
    "AUC": roc_auc_score(all_labels, all_probs)
}

print(f"PyTorch DinoDetector Test Results:")
for k, v in pt_metrics.items():
    print(f"  {k}: {v:.4f}")

# 7. 对比经典机器学习模型 (RandomForest, etc.)
print("\n6. Training & Evaluating Classical Classifiers for Comparison...")
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results = {}
for name, clf in classifiers.items():
    print(f"   Training {name}...")
    # 经典模型通常在 CPU 上训练更快，所以用 numpy 数据
    clf.fit(train_features, train_labels)
    
    # Predict
    test_preds = clf.predict(test_features)
    if hasattr(clf, "predict_proba"):
        test_probs = clf.predict_proba(test_features)[:, 1]
    else:
        test_probs = test_preds
    
    # Metrics
    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    auc = roc_auc_score(test_labels, test_probs)
    
    results[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}
    print(f"   ✅ {name} - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# 8. 打印最终对比表格
print("\n" + "="*80)
print("🏆 FINAL COMPARISON ON TEST SET (Patient-Level Split)")
print("="*80)
print(f"| {'Model':<15} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'AUC':>10} |")
print("-" * 80)

# PyTorch Model
row_pt = f"| {'DinoDetector':<15} | {pt_metrics['Accuracy']:>10.4f} | {pt_metrics['Precision']:>10.4f} | {pt_metrics['Recall']:>10.4f} | {pt_metrics['F1-Score']:>10.4f} | {pt_metrics['AUC']:>10.4f} |"
print(row_pt)

# Classical Models
for name, metrics in results.items():
    row = f"| {name:<15} | {metrics['acc']:>10.4f} | {metrics['prec']:>10.4f} | {metrics['rec']:>10.4f} | {metrics['f1']:>10.4f} | {metrics['auc']:>10.4f} |"
    print(row)

print("="*80)
print("🎉 Training and Evaluation Complete!")
print(f"Best PyTorch model saved at: {os.path.join(checkpoint_dir, 'dino_detector_best.pth')}")