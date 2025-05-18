# 🌾 AGRiFARM - Crop & Fertilizer Recommendation System using Machine Learning

An intelligent, data-driven system built with machine learning that empowers farmers and agricultural professionals to make smarter decisions about crop selection and fertilizer application based on real-time soil and environmental parameters.
***
![Screenshot 2025-05-17 011408](https://github.com/user-attachments/assets/4200ee9f-e8d7-488f-8292-6d2854c6cf90)

![Screenshot 2025-05-17 011431](https://github.com/user-attachments/assets/24c57e47-03b6-4b60-8f58-85382ca6fc03)

![Screenshot 2025-05-17 011445](https://github.com/user-attachments/assets/f57073a5-edcb-4452-b7c9-fd3059900050)

***
🌟 Overview
Modern agriculture thrives on data-driven insights to enhance yield, profitability, and sustainability. AGRiFARM uses machine learning models trained on agricultural data to analyze:

🌱 Nitrogen (N)

🌱 Phosphorus (P)

🌱 Potassium (K)

🌡️ Temperature

💧 Humidity

⚗️ pH Level

🌧️ Rainfall

…and recommends the best crop and fertilizer combination suited to your local conditions.
***
🚀 Features
🌿 Crop Recommendation – Suggests the ideal crops for the given soil and weather conditions.

🧪 Fertilizer Recommendation – Recommends suitable fertilizers to balance nutrient levels.

🌐 Interactive Web Interface – Built with Flask and deployed for accessibility.

📊 Visualization – Charts and graphs for intuitive understanding.

🤖 High Accuracy Models – Ensemble and classical models trained and evaluated.
***
📊 Datasets
📌 Crop Recommendation Dataset
Features: N, P, K, Temperature, Humidity, pH, Rainfall

Target: Crop Name

Records: ~2,200

Source: Kaggle

📌 Fertilizer Recommendation Dataset
Features: Soil Type, Crop Type, Nutrient Deficiencies

Target: Fertilizer Name

Source: Custom/Preprocessed
***
🧠 Model Architectures
🌿 Crop Prediction Models
Decision Tree

Random Forest ✅ Best: 98.55% Accuracy

Logistic Regression

SVM

XGBoost

Gaussian Naive Bayes

K-Nearest Neighbors

🧪 Fertilizer Recommendation Model
Decision Tree
✅ Achieved 99.55% Accuracy
***
🏗️ System Architecture

Two modules working together:

🌿 Crop Recommendation

🧪 Fertilizer Suggestion
***
📈 Results
✅ Crop Prediction:
Random Forest: 98.55%

SVM, Logistic Regression also performed well

✅ Fertilizer Prediction:
Decision Tree: 99.55%

For detailed accuracy, confusion matrix, and precision-recall metrics, check the Jupyter notebooks in the repository.
***
💻 Installation
Clone the repository and install dependencies:

git clone https://github.com/diwakar2905/AGRiFARM-Crop-and-Fertiliser-Recommendation-System-using-Machine-Learning.git
cd AGRiFARM-Crop-and-Fertiliser-Recommendation-System-using-Machine-Learning
pip install -r requirements.txt

🛠️ Usage
To run the Flask web application locally:

python app.py
Visit http://localhost:5000 in your browser.
***
🤝 Contributing
Contributions, suggestions, and forks are welcome!
Feel free to submit a pull request or open an issue.
***
📝 License
This project is open-sourced under the MIT License.
***
🙌 Acknowledgements

🌐 Kaggle Datasets

📘 Flask Documentation

📊 Scikit-learn

❤️ All open-source contributors

Scikit-learn

Community contributors ❤️

