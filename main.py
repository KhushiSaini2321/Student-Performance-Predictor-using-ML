# main.py

#  DAY 1: DATA LOADING, CLEANING & EXPLORATION
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Load the dataset
df = pd.read_csv("student_data.csv")
print(" Dataset Preview:\n", df.head())

# Dataset Info
print("\n Dataset Info:")
print(df.info())

# Check missing values
print("\n Missing Values:\n", df.isnull().sum())

# (Optional) Fill missing values if any
df.fillna(method='ffill', inplace=True)

#  Describe numerical data
print("\n Statistical Summary:\n", df.describe())

#  Encode the 'Performance' column (Pass → 1, Fail → 0)
le = LabelEncoder()
df['Performance'] = le.fit_transform(df['Performance'])

#  Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

#  DAY 2: MODEL TRAINING, PREDICTION & EVALUATION

# Feature selection
X = df.drop("Performance", axis=1)
y = df["Performance"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("\n Predictions:", y_pred)

#  Evaluate the model
print("\n Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

#  Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:\n", cm)

#  Plot Confusion Matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
