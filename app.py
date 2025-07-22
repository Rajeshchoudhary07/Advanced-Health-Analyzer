from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)

# Create or load model
MODEL_PATH = "health_model.pkl"
if not os.path.exists(MODEL_PATH):
    # Create synthetic data and train model
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 80, 1000),
        'Gender': np.random.randint(0, 2, 1000),
        'BMI': np.random.uniform(18, 40, 1000).round(1),
        'BloodPressure': np.random.randint(90, 180, 1000),
        'Cholesterol': np.random.randint(1, 4, 1000),
        'Glucose': np.random.randint(1, 4, 1000),
        'Smoking': np.random.randint(0, 2, 1000),
        'Alcohol': np.random.randint(0, 3, 1000),
        'PhysicalActivity': np.random.randint(0, 3, 1000)
    }
    df = pd.DataFrame(data)
    risk_factors = (
        (df['Age'] > 50).astype(int) + 
        (df['BMI'] > 30).astype(int) + 
        (df['BloodPressure'] > 140).astype(int) + 
        (df['Cholesterol'] > 2).astype(int) + 
        (df['Glucose'] > 2).astype(int) + 
        df['Smoking'] + 
        (df['Alcohol'] > 1).astype(int) - 
        (df['PhysicalActivity'] > 0).astype(int)
    )
    df['HealthRisk'] = np.where(risk_factors >= 4, 'High',
                               np.where(risk_factors >= 2, 'Medium', 'Low'))
    
    le = LabelEncoder()
    df['HealthRisk'] = le.fit_transform(df['HealthRisk'])
    
    X = df.drop('HealthRisk', axis=1)
    y = df['HealthRisk']
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    
    joblib.dump((model, le), MODEL_PATH)
else:
    model, le = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        # Prepare input data
        input_data = np.array([[
            int(data['age']),
            int(data['gender']),
            float(data['bmi']),
            int(data['bp']),
            int(data['cholesterol']),
            int(data['glucose']),
            int(data['smoking']),
            int(data['alcohol']),
            int(data['activity'])
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)
        risk_level = le.inverse_transform(prediction)[0]
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_level, data)
        
        return jsonify({
            'status': 'success',
            'risk_level': risk_level,
            'recommendations': recommendations,
            'visualization_data': generate_visualization_data(input_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def generate_recommendations(risk_level, data):
    recommendations = []
    
    # General recommendations based on risk level
    if risk_level == 'High':
        recommendations.append({
            'type': 'warning',
            'message': 'Immediate consultation with a healthcare professional is recommended'
        })
    elif risk_level == 'Medium':
        recommendations.append({
            'type': 'warning',
            'message': 'Consider lifestyle changes and regular check-ups'
        })
    else:
        recommendations.append({
            'type': 'success',
            'message': 'Maintain your healthy lifestyle'
        })
    
    # Specific recommendations based on inputs
    if float(data['bmi']) > 30:
        recommendations.append({
            'type': 'diet',
            'message': 'Consider weight management strategies'
        })
    
    if int(data['bp']) > 140:
        recommendations.append({
            'type': 'heart',
            'message': 'Monitor your blood pressure regularly'
        })
    
    if int(data['activity']) == 0:
        recommendations.append({
            'type': 'exercise',
            'message': 'Incorporate regular physical activity into your routine'
        })
    
    return recommendations

def generate_visualization_data(input_data):
    # This would be more sophisticated in a real application
    return {
        'risk_score': int(np.random.randint(0, 100, 1)[0]),
        'comparison': {
            'age_group': int(np.random.randint(60, 80, 1)[0]),
            'similar_profiles': int(np.random.randint(50, 90, 1)[0])
        },
        'factors': {
            'bmi_impact': float(input_data[0][2]) / 40 * 100,
            'bp_impact': float(input_data[0][3]) / 180 * 100,
            'activity_impact': (3 - input_data[0][8]) / 3 * 100
        }
    }

if __name__ == '__main__':
    app.run(debug=True)