# 🩺 Breast Cancer Detection System using Machine Learning

A Machine Learning-powered web application that predicts whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** based on medical diagnostic measurements. The project is built using Python, Scikit-learn, and Streamlit, providing fast and accurate predictions through an interactive web interface.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide, and early diagnosis plays a crucial role in successful treatment.

This project uses Machine Learning to analyze diagnostic features extracted from breast cell samples and predict whether a tumor is benign or malignant.

The entire Machine Learning workflow—from data preprocessing and model training to deployment—has been implemented in this project.

---

## 🚀 Features

- 🔍 Predicts Breast Cancer Diagnosis
- 📊 Uses 30 diagnostic input features
- 🤖 Machine Learning-based prediction
- 📈 Displays prediction confidence
- 🌐 Interactive Streamlit Web Application
- 💾 Saved ML Pipeline using Joblib
- 📱 Clean and user-friendly interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset** containing **30 numerical features** extracted from digitized images of breast cell nuclei.

Examples of features include:

- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Smoothness Mean
- Compactness Mean
- Concavity Mean
- Symmetry Mean
- Fractal Dimension Mean

Target Classes:

- **Benign (0)**
- **Malignant (1)**

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Loading

- Loaded dataset using Pandas
- Inspected dataset structure
- Checked missing values and data types

### 2️⃣ Data Preprocessing

- Removed unnecessary columns (if any)
- Split data into training and testing sets

### 3️⃣ Model Training

Multiple Machine Learning models were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### 4️⃣ Model Evaluation

Models were evaluated using:

- Accuracy Score
- ROC-AUC Score
- Classification Report
- Confusion Matrix

---

## 🏆 Best Model

**Random Forest Classifier**

### Performance

| Metric | Score |
|---------|-------|
| Accuracy | **97.37%** |
| ROC-AUC Score | **99.29%** |

---

## 💾 Model Saving

The trained pipeline was saved using **Joblib**, allowing predictions without retraining.

```python
joblib.dump(best_model, "breast_cancer_pipeline.pkl")
```

---

## 🌐 Streamlit Web Application

The web application allows users to:

- Enter diagnostic measurements
- Predict whether the tumor is Benign or Malignant
- View prediction confidence instantly

---

## 📷 Screenshots

### Home Page

> Add a screenshot here

### Prediction Result

> Add a screenshot here

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Breast-Cancer-Detection-Predictor.git
```

Move into the project folder:

```bash
cd Breast-Cancer-Detection-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Breast-Cancer-Detection-Predictor
│
├── app.py
├── Breast_Cancer_Detection.ipynb
├── breast_cancer_pipeline.pkl
├── breast_cancer.csv
├── requirements.txt
├── README.md
└── screenshots
```

---

## 🎯 Learning Outcomes

This project helped me gain practical experience in:

- Data Preprocessing
- Machine Learning Classification
- Model Evaluation
- Pipeline Creation
- Streamlit Deployment
- GitHub Project Management

---

## 👨‍💻 Author

**Lakshay Verma**

AI & Machine Learning Enthusiast

---

## 🙏 Acknowledgements

This project was developed as part of my **AI & Machine Learning Internship** at **InternPe**.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

---

## 📄 License

This project is created for educational purposes as part of an internship.
