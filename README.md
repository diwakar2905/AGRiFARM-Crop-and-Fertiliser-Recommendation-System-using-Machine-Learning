🌾 AGRiFARM - Crop & Fertilizer Recommendation System using Machine Learning

An intelligent, data-driven system built with machine learning that empowers farmers and agricultural professionals to make smarter decisions about crop selection and fertilizer application based on real-time soil and environmental parameters.

![Screenshot 2025-05-17 011408](https://github.com/user-attachments/assets/4200ee9f-e8d7-488f-8292-6d2854c6cf90)

![Screenshot 2025-05-17 011431](https://github.com/user-attachments/assets/24c57e47-03b6-4b60-8f58-85382ca6fc03)

![Screenshot 2025-05-17 011445](https://github.com/user-attachments/assets/f57073a5-edcb-4452-b7c9-fd3059900050)

📑 Table of Contents
🌟 Overview

🚀 Features

📊 Datasets

🧠 Model Architectures

🏗️ System Architecture

📈 Results

💻 Installation

🛠️ Usage

🤝 Contributing

📝 License

🌟 Overview
Modern agriculture demands data-driven insights for improving yield, profitability, and sustainability. This project uses advanced machine learning models to analyze parameters like:

🌱 Nitrogen (N)

🌱 Phosphorus (P)

🌱 Potassium (K)

🌡️ Temperature

💧 Humidity

⚗️ pH Level

🌧️ Rainfall

…and recommends the best-suited crop and fertilizer tailored to those conditions.

🚀 Features
🌿 Crop Recommendation — Suggests the most suitable crops based on soil and climate parameters.

🧪 Fertilizer Recommendation — Recommends optimal fertilizers to enhance productivity and soil health.

🌐 Interactive Web Application — Clean, intuitive Flask-based interface.

📊 Data Visualization — Graphs and charts for clearer insights.

📈 High Accuracy — Utilizes multiple ML models with excellent performance.

📊 Datasets
📌 Crop Recommendation Dataset
Features: N, P, K, Temperature, Humidity, pH, Rainfall

Target: Crop Name

Records: 2,200

Source: Kaggle

📌 Fertilizer Recommendation Dataset
Features: Soil Type, Crop Type, Nutrient Levels

Target: Fertilizer Name

🧠 Model Architectures
📈 Crop Recommendation Models
Decision Tree

Gaussian Naive Bayes

Support Vector Machine (SVM)

Logistic Regression

Random Forest ✅ (Best: 98.55% accuracy)

XGBoost

K-Nearest Neighbors (KNN)

📈 Fertilizer Recommendation Model
Algorithm: Decision Tree

Accuracy: [100%]

🏗️ System Architecture
Two primary modules integrated into a single web app:

🌿 Crop Recommendation Module — Predicts suitable crops based on soil and environment.

🧪 Fertilizer Recommendation Module — Suggests appropriate fertilizers for the selected crop.

📈 Results
📌 Crop Recommendation
✅ Random Forest: 98.55% accuracy

XGBoost, GaussianNB, and others also performed well.

📌 Fertilizer Recommendation
✅ Descision Tree: 99.55% accuracy

For detailed confusion matrices and evaluation metrics, refer to the Jupyter Notebooks in this repository.

🌟 Acknowledgements
Kaggle Datasets

Flask Documentation

Scikit-learn

Community contributors ❤️

📌 Connect & Feedback
Have ideas or suggestions? Feel free to open an issue or reach out via LinkedIn or email.
