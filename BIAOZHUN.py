import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 示例数据
data = {
    'Pregnancies': [1, 0, 2, 4],
    'Glucose': [85, 78, 90, 88],
    'BloodPressure': [66, 70, 64, 72],
    'SkinThickness': [29, 0, 35, 30],
    'Insulin': [0, 0, 0, 120],
    'BMI': [26.6, 31.2, 24.3, 29.5],
    'DiabetesPedigreeFunction': [0.351, 0.245, 0.368, 0.450],
    'Age': [31, 25, 30, 28]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 创建性别衍生特征：怀孕次数大于0为女性（0），否则为男性（1）
df['Gender'] = df['Pregnancies'].apply(lambda x: 0 if x > 0 else 1)

# 创建其他衍生特征
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Age_Insulin'] = df['Age'] * df['Insulin']
df['BMI_Age'] = df['BMI'] * df['Age']
df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

# 特征选择
features = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                'Glucose_BMI', 'Age_Insulin', 'BMI_Age',
                'Glucose_Insulin_Ratio', 'Gender']]

# 创建 StandardScaler 实例
scaler = StandardScaler()

# 拟合并转换数据
X_scaled = scaler.fit_transform(features)

# 保存 scaler 和模型
joblib.dump(scaler, 'scaler.pkl')