from flask import Flask, request, render_template
import pandas as pd
import os

import joblib


UI = Flask(__name__)

print("Templates folder path:", os.path.abspath(UI.template_folder))

# 加载模型和标准化器
model = joblib.load("best_diabetes_model.pkl")  # 确保模型文件在正确位置
scaler = joblib.load("scaler.pkl")  # 加载之前保存的标准化器

@UI.route('/')
def home():
    return render_template('index.html')

@UI.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 获取用户输入的特征
        pregnancies = int(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        # 创建输入 DataFrame
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age]
        })

        # 创建性别衍生特征
        input_data['Gender'] = input_data['Pregnancies'].apply(lambda x: 0 if x > 0 else 1)

        # 创建其他衍生特征
        input_data['Glucose_BMI'] = input_data['Glucose'] * input_data['BMI']
        input_data['Age_Insulin'] = input_data['Age'] * input_data['Insulin']
        input_data['BMI_Age'] = input_data['BMI'] * input_data['Age']
        input_data['Glucose_Insulin_Ratio'] = input_data['Glucose'] / (input_data['Insulin'] + 1)

        # 特征选择（确保与训练时一致）
        X = input_data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                         'Glucose_BMI', 'Age_Insulin', 'BMI_Age',
                         'Glucose_Insulin_Ratio', 'Gender']]

        # 标准化
        X_scaled = scaler.transform(X)
        print(f"标准化后的特征: {X_scaled}")
        # 预测
        predictions_proba = model.predict_proba(X_scaled)
        diabetes_probability = predictions_proba[0][1]
        formatted_probability = "{:.2f}".format(diabetes_probability)
        print(f"患病概率: {formatted_probability}")

        return render_template('result.html', probability = formatted_probability)


if __name__ == '__main__':
    UI.run(debug=True)