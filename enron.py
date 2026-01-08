import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import time

# ----------------------------
# 1. 加载单个 Enron 数据集
# ----------------------------
def load_enron_data(base_dir):
    texts, labels = [], []

    # Load ham
    ham_path = os.path.join(base_dir, "ham")
    for filename in os.listdir(ham_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(ham_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                    labels.append(0)  # ham
            except Exception as e:
                continue

    # Load spam
    spam_path = os.path.join(base_dir, "spam")
    for filename in os.listdir(spam_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(spam_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                    labels.append(1)  # spam
            except Exception as e:
                continue

    return texts, np.array(labels)

# ----------------------------
# 2. 主循环：遍历 enron1 到 enron6
# ----------------------------
def main():
    all_results = []
    enron_datasets = [f"enron{i}" for i in range(1, 7)]  # ['enron1', 'enron2', ..., 'enron6']

    for dataset_name in enron_datasets:
        print(f"\n{'='*60}")
        print(f"正在处理数据集: {dataset_name}")
        print(f"{'='*60}")

        # 检查文件夹是否存在
        if not os.path.exists(dataset_name):
            print(f"警告: 文件夹 '{dataset_name}' 不存在，跳过。")
            continue

        try:
            # 加载数据
            texts, labels = load_enron_data(dataset_name)
            print(f"加载成功: {len(texts)} 封邮件 ({np.sum(labels)} spam, {len(labels)-np.sum(labels)} ham)")

            # 划分训练/测试集
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # 构建模型 pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    stop_words='english',
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )),
                ('nb', MultinomialNB(alpha=0.1))
            ])

            # 训练
            start = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start

            # 测试集评估
            start = time.time()
            y_pred = pipeline.predict(X_test)
            pred_time = time.time() - start
            test_acc = accuracy_score(y_test, y_pred)

            # 交叉验证 (5-fold)
            cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring='accuracy')
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

            # 保存结果
            all_results.append({
                'Dataset': dataset_name,
                'Test_Accuracy': test_acc,
                'CV_Accuracy_Mean': cv_mean,
                'CV_Accuracy_Std': cv_std,
                'Train_Time(s)': round(train_time, 2),
                'Pred_Time(s)': round(pred_time, 4),
                'Total_Emails': len(texts)
            })

            # 打印简要结果
            print(f"测试准确率: {test_acc:.4f}")
            print(f"CV 准确率: {cv_mean:.4f} ± {cv_std:.4f}")

        except Exception as e:
            print(f"处理 {dataset_name} 时出错: {e}")
            continue

    # ----------------------------
    # 3. 汇总并打印所有结果
    # ----------------------------
    if all_results:
        df_results = pd.DataFrame(all_results)
        print(f"\n{'='*80}")
        print("所有 Enron 数据集性能汇总")
        print(f"{'='*80}")
        print(df_results.to_string(index=False, float_format="%.4f"))

        # （可选）保存到 CSV 文件
        df_results.to_csv("enron_spam_classification_results.csv", index=False)
        print("\n结果已保存至 'enron_spam_classification_results.csv'")
    else:
        print("未找到任何有效的 enron 数据集。")

# ----------------------------
# 4. 运行主程序
# ----------------------------
if __name__ == "__main__":
    main()