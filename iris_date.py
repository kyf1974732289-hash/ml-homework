import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. 数据加载与划分
# ----------------------------
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 2. 模型定义：多种配置
# ----------------------------
models = {}

# SVM with different kernels
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    models[f'SVM_{kernel}'] = SVC(kernel=kernel, random_state=42)

# MLP with different architectures
mlp_configs = {
    'MLP_(10)': (10,),
    'MLP_(20,10)': (20, 10),
    'MLP_(50)': (50,),
    'MLP_(100)': (100,)
}
for name, hidden in mlp_configs.items():
    models[name] = MLPClassifier(
        hidden_layer_sizes=hidden,
        max_iter=1000,
        random_state=42,
        solver='adam',
        early_stopping=True
    )

# GBDT models
models['GBDT_sklearn'] = GradientBoostingClassifier(random_state=42)
models['XGBoost'] = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# ----------------------------
# 3. 评估函数：交叉验证 + 测试集 + 时间
# ----------------------------
results = []

for name, model in models.items():
    # 决定是否使用标准化数据（树模型不需要）
    X_train_use = X_train_scaled if 'SVM' in name or 'MLP' in name else X_train
    X_test_use = X_test_scaled if 'SVM' in name or 'MLP' in name else X_test

    # 训练时间
    start = time.time()
    model.fit(X_train_use, y_train)
    train_time = time.time() - start

    # 预测时间 + 测试准确率
    start = time.time()
    y_pred = model.predict(X_test_use)
    pred_time = time.time() - start
    test_acc = accuracy_output = accuracy_score(y_test, y_pred)

    # 5折交叉验证（使用原始X, y，但注意：SVM/MLP需要标准化 → 用pipeline更严谨，此处简化）
    if 'SVM' in name or 'MLP' in name:
        # 对需要标准化的模型，手动做 CV（避免数据泄露）
        from sklearn.model_selection import StratifiedKFold
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            scaler_cv = StandardScaler()
            X_tr_scaled = scaler_cv.fit_transform(X_tr)
            X_val_scaled = scaler_cv.transform(X_val)
            model_cv = type(model)(**model.get_params())
            model_cv.fit(X_tr_scaled, y_tr)
            score = model_cv.score(X_val_scaled, y_val)
            cv_scores.append(score)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
    else:
        # 树模型无需标准化，直接 CV
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

    results.append({
        'Model': name,
        'CV_Acc_Mean': cv_mean,
        'CV_Acc_Std': cv_std,
        'Test_Acc': test_acc,
        'Train_Time(s)': train_time,
        'Pred_Time(s)': pred_time
    })

# ----------------------------
# 4. 结果汇总与排序
# ----------------------------
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='CV_Acc_Mean', ascending=False).reset_index(drop=True)

# 打印表格
print("=== 模型性能对比（5折交叉验证 + 测试集） ===")
print(df_results.to_string(index=False, float_format="%.4f"))

# ----------------------------
# 5. 最佳模型详细报告（选CV最高的）
# ----------------------------
best_model_name = df_results.iloc[0]['Model']
best_model = models[best_model_name]

X_use = X_train_scaled if 'SVM' in best_model_name or 'MLP' in best_model_name else X_train
X_test_use = X_test_scaled if 'SVM' in best_model_name or 'MLP' in best_model_name else X_test

best_model.fit(X_use, y_train)
y_pred_best = best_model.predict(X_test_use)

print(f"\n=== 最佳模型 {best_model_name} 的分类报告 ===")
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))