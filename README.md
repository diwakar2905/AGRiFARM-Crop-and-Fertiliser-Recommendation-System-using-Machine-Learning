# AGRiFARM-Crop-and-Fertiliser-Recommendation-System-using-Machine-Learning
🌾 Crop & Fertilizer Recommendation System using Machine Learning
An intelligent system that leverages machine learning to guide farmers and agricultural professionals in selecting the most suitable crops and fertilizers, based on real-time soil and environmental data.
![Screenshot 2025-05-17 011408](https://github.com/user-attachments/assets/9ebd29c5-f813-4727-9c24-660fed9717b1)
![Screenshot 2025-05-17 011445](https://github.com/user-attachments/assets/fdf1ea43-3ed0-4fac-88db-e7ae3ab28b4f)
![Screenshot 2025-05-17 011431](https://github.com/user-attachments/assets/3009f50d-59f1-435f-9ec1-1f746401ed90)




📑 Table of Contents
Overview

Features

Datasets

Model Architectures

System Architecture

Results

Installation

Usage

Contributing

License

🌟 Overview
Modern agriculture requires data-driven decisions for optimal yield and sustainability. This project utilizes advanced machine learning algorithms to provide actionable recommendations for both crop selection and fertilizer application, tailored to specific soil nutrients and environmental conditions. By analyzing parameters like nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall, the system empowers users to make informed choices that boost productivity and soil health.

🚀 Features
Crop Recommendation: Suggests the best crops for given soil and environmental parameters.

Fertilizer Recommendation: Recommends optimal fertilizers to maximize crop yield and maintain soil health.

User-Friendly Web App: Interactive and intuitive interface for seamless user experience.

Data Visualization: Graphical representation of predictions and input data.

High Accuracy: Employs multiple machine learning models for reliable results.

📊 Datasets
Crop Recommendation Dataset
Features: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall

Target: Crop label

Size: 2,200 records

Source: Kaggle - Crop Recommendation Dataset

Fertilizer Recommendation Dataset
Features: Soil Type, Crop Type, Nutrient Levels

Target: Fertilizer Name

🧠 Model Architectures
Crop Recommendation
Algorithms Used:

Decision Tree

Gaussian Naive Bayes

Support Vector Machine (SVM)

Logistic Regression

Random Forest

XGBoost

K-Nearest Neighbors (KNN)

Best Model: Random Forest (Accuracy: 99.55%)

Fertilizer Recommendation
Algorithm Used: [Specify, e.g., Decision Tree]

Accuracy: [Specify if available]

🏗️ System Architecture
The system is structured into two main modules:

Crop Recommendation Module: Processes soil and environmental parameters to predict the most suitable crop.

Fertilizer Recommendation Module: Suggests the optimal fertilizer based on selected crop and soil nutrients.

Both modules are integrated into a web application for a smooth user workflow.

📈 Results
Crop Recommendation:

Random Forest achieved the highest accuracy (99.55%).

Other models like XGBoost and Gaussian Naive Bayes also performed well.

Fertilizer Recommendation:

[Add specific results if available.]

For detailed metrics and confusion matrices, refer to the Jupyter notebooks in this repository.

💻 Installation
Clone the Repository

bash
git clone https://github.com/diwakar2905/Crop-and-Fertiliser-Recommendation-System-using-Machine-Learning.git
cd Crop-and-Fertiliser-Recommendation-System-using-Machine-Learning
Create a Virtual Environment

bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install Dependencies

bash
pip install -r requirements.txt
Run the Application

bash
python app.py
Access the Web Application
Open your browser and navigate to http://localhost:5000

🛠️ Usage
Crop Recommendation
Go to the Crop Recommendation section in the web app.

Enter soil parameters: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.

Click Predict to receive the recommended crop.

Fertilizer Recommendation
Go to the Fertilizer Recommendation section.

Enter the crop name and current soil nutrient levels.

Click Recommend to get the suggested fertilizer.

🤝 Contributing
Contributions are welcome! To contribute:

Fork this repository.

Create a new branch for your feature or bugfix:

bash
git checkout -b feature/YourFeatureName
Commit your changes:

bash
git commit -m "Add Your Feature"
Push to your branch:

bash
git push origin feature/YourFeatureName
Open a Pull Request and describe your changes.
