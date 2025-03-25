# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 初始设置
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import kagglehub

# 下载最新版本数据集
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)

# ==================== 数据加载与预处理 ====================
print("Step 1: 数据加载与预处理")
df = pd.read_csv(path + "/diabetes.csv")

# 设置随机种子并添加性别列
np.random.seed(42)
df['Gender'] = np.random.choice(['Male', 'Female'], size=len(df))

def advanced_feature_engineering(df):
    df = df.copy()

    # 处理原始缺失值（0值转NaN）
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    for col in zero_cols:
        # 先按Outcome分组填充
        df[col] = df.groupby("Outcome")[col].transform(
            lambda x: x.fillna(x.median() if not x.isnull().all() else np.nanmedian(df[col]))
        )
        # 全局填充兜底
        df[col].fillna(df[col].median(), inplace=True)

    # 创建新特征
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Insulin'] = df['Age'] * df['Insulin']

    # 处理新特征的缺失值
    new_features = ['Glucose_BMI', 'Age_Insulin']
    for feat in new_features:
        df[feat].fillna(df[feat].median(), inplace=True)

    return df

df = advanced_feature_engineering(df)

# ==================== 特征工程 ====================
print("\nStep 2: 特征工程")
df["BMI_Age"] = df["BMI"] * df["Age"]
df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)

# ==================== 数据准备 ====================
print("\nStep 3: 数据准备")
# 将性别列转换为数值型
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Male=0, Female=1

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 处理类别不平衡
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res,
    test_size=0.2,
    stratify=y_res,
    random_state=42
)

# ==================== 模型配置 ====================
print("\nStep 4: 模型配置")
model_configs = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", None]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss"),
        "params": {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42),
        "params": {
            "num_leaves": [31, 63],
            "learning_rate": [0.01, 0.1],
            "feature_fraction": [0.8, 1.0]
        }
    }
}

# ==================== 超参数调优 ====================
print("\nStep 5: 超参数调优")
best_models = {}

for name, config in model_configs.items():
    print(f"\n{'=' * 30} Tuning {name} {'=' * 30}")
    model = config["model"]
    param_grid = config["params"]

    # 网格搜索
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 保存最佳模型
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_:.4f}")

# ==================== 模型评估 ====================
print("\nStep 6: 模型评估")

def plot_learning_curve(estimator, title, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 5),
        scoring="roc_auc"
    )

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="r", label="Training Score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), "o-", color="g", label="CV Score")
    plt.fill_between(train_sizes,
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                     np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                     alpha=0.1, color="g")
    plt.legend(loc="best")
    plt.show()

for name, model in best_models.items():
    print(f"\n{'=' * 30} Evaluating {name} {'=' * 30}")

    # 学习曲线
    plot_learning_curve(model, f"{name} Learning Curve", X_train, y_train)

    # 预测评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# ==================== 模型集成 ====================
print("\nStep 7: 模型集成")
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting="soft"
)
voting_clf.fit(X_train, y_train)

# 评估集成模型
print("\n" + "=" * 30 + " Evaluating Ensemble Model " + "=" * 30)
y_pred_ensemble = voting_clf.predict(X_test)
y_proba_ensemble = voting_clf.predict_proba(X_test)[:, 1]

print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
print(f"Ensemble ROC AUC: {roc_auc_score(y_test, y_proba_ensemble):.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

# ==================== 保存最佳模型 ====================
print("\nStep 8: 保存模型")
model_performance = {}
for name, model in best_models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    model_performance[name] = roc_auc_score(y_test, y_proba)

# 加入集成模型
model_performance["Ensemble"] = roc_auc_score(y_test, y_proba_ensemble)

# 找出最佳模型
best_model_name = max(model_performance, key=model_performance.get)
best_model = voting_clf if best_model_name == "Ensemble" else best_models[best_model_name]

print(f"\n最佳模型: {best_model_name} (ROC AUC: {model_performance[best_model_name]:.4f})")
joblib.dump(best_model, "best_diabetes_model.pkl")
print("模型已保存为 best_diabetes_model.pkl")

# ==================== 特征重要性 ====================
print("\nStep 9: 特征重要性分析")
try:
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        raise AttributeError

    features = df.drop("Outcome", axis=1).columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"无法生成特征重要性: {str(e)}")