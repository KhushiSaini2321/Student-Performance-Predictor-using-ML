# Student-Performance-Predictor-using-ML
# 🎓 Student Performance Predictor using Machine Learning

This project uses a Machine Learning model to predict whether a student will **Pass** or **Fail** based on their study habits and academic data.

---

## 📌 Objective

To build a simple machine learning model that predicts student performance using features like:

- Study hours
- Attendance percentage
- Previous grades
- Sleep hours

The output is a binary classification: **Pass or Fail**.

---

## 📂 Dataset

A custom CSV file (`student_data.csv`) with manually created records.

### Sample Data:

| StudyHours | Attendance | PreviousGrade | SleepHours | Performance |
|------------|------------|----------------|-------------|--------------|
| 3.0        | 85         | 75             | 6.0         | Pass         |
| 2.0        | 60         | 55             | 5.0         | Fail         |

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧠 Algorithms Used

- **Decision Tree Classifier** (primary model)
- Optional: Logistic Regression (comparison)

---

## 📈 What the Project Does

- Loads and processes CSV data
- Encodes categorical labels
- Splits data into training/testing sets
- Trains a Decision Tree model
- Evaluates model with:
  - Accuracy score
  - Confusion matrix
  - Classification report
- Visualizes:
  - Confusion Matrix
  - Correlation Heatmap

---

## ✅ Output Example

- **Accuracy:** 100% (on test data)
- **Prediction Output:** `[1, 0, 1]` → (Pass, Fail, Pass)
- **Confusion Matrix:**  
  ![confusion-matrix](screenshots/confusion_matrix.png)

- **Correlation Heatmap:**  
  ![heatmap](screenshots/heatmap.png)


