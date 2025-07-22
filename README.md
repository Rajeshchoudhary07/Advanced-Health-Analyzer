# Advanced-Health-Analyzer
# Advanced-Health-Analyzer
# 🧠 Health Risk Predictor Web App

A *machine learning-powered web application* that evaluates a user's health risk level based on medical inputs and provides actionable recommendations. Designed with an interactive interface and built for accessibility across all devices.



## 📌 Project Overview

This project is a full-stack web application that predicts *health risk levels (Low / Medium / High)* using a *Decision Tree Classifier* trained on health parameters such as age, BMI, blood pressure, and more. It also offers:

- *Interactive Visualizations* of health factors  
- *Personalized Recommendations* to improve user well-being  
- *Real-time Predictions* using a clean and responsive UI  

---

## 🛠 Tech Stack

| Layer        | Technology                         |
|--------------|-------------------------------------|
| *Frontend* | HTML5, CSS3, JavaScript             |
| *Backend*  | Python Flask                        |
| *ML Model* | Decision Tree Classifier (scikit-learn) |
| *Styling*  | Custom CSS, Font Awesome Icons      |
| *Charts*   | Chart.js                            |

---

## 🌟 Key Features

- ✅ *User-Friendly Interface* with responsive design  
- 📊 *Visual Health Reports* via doughnut and radar charts  
- 🧮 *Real-Time ML Predictions* using 9 key health parameters  
- 📱 *Mobile-First* responsive layout  
- 🔄 *AJAX-based Form Submission* for seamless experience  
- 📁 *Model Persistence* with joblib  

---

## 🧬 Health Parameters Used

The app processes the following 9 health metrics:

- Age  
- BMI (Body Mass Index)  
- Systolic & Diastolic Blood Pressure  
- Glucose Level  
- Smoking Status  
- Physical Activity  
- Alcohol Intake  
- Cholesterol Level  

---

## 📈 Data Visualization

- *Doughnut Charts*: Breakdown of individual risk factors  
- *Radar Charts*: Comparison of user data vs ideal metrics  
- *Color Indicators*: Visual cues for Low, Medium, and High risk  

---

## 🧪 How It Works

1. *User Input*: Health parameters are entered through an intuitive web form.  
2. *Backend Processing*: Flask server processes the input and passes it to the ML model.  
3. *Prediction & Visualization*:
   - Health Risk Level is predicted (Low/Medium/High)  
   - Results are visualized with charts and metrics  
   - Health recommendations are shown for improvement  

---

## ⚙ Technical Details

- *Model Training*: A synthetic dataset is used to train a Decision Tree Classifier  
- *Joblib Serialization*: Trained model is saved and reused for predictions  
- *AJAX*: Improves UX by submitting forms without reloading the page  
- *Chart.js*: Powers interactive and animated data visualizations  

---

## 📂 Project Structure
├── static/

│ ├── css/

│ ├── js/

│ └── images/

├── templates/

│ ├── index.html

│ └── result.html

├── model/

│ └── decision_tree_model.pkl

├── app.py

├── requirements.txt

└── README.md
