from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)
CORS(app)

# Данные модели
X = np.array([[1400, 7.0, 3.5], [1200, 6.5, 3.2], [1600, 8.0, 3.8]])
y = [1, 0, 1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Загрузка данных о университетах
with open('universities.json', 'r') as f:
    universities = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(f"Received data: {data}")

    # Получение данных от пользователя
    sat = data.get('SAT')
    ielts = data.get('IELTS')
    gpa = data.get('GPA')

    if sat is None or ielts is None or gpa is None:
        return jsonify({'error': 'Введите SAT, IELTS и GPA'}), 400

    # Масштабирование входных данных
    input_data = np.array([[sat, ielts, gpa]])
    input_data_scaled = scaler.transform(input_data)

    # Предсказание вероятности поступления
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Подбор университетов
    recommended_universities = [
        uni['name'] for uni in universities
        if sat >= uni['min_sat'] and ielts >= uni['min_ielts'] and gpa >= uni['min_gpa']
    ]

    # Возврат результата
    return jsonify({
        'admission_chance': probability * 100,
        'recommended_universities': recommended_universities
    })

if __name__ == '__main__':
    app.run(debug=True)
