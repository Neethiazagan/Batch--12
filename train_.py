import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('sample_patient_dataset.csv')

# Encode categorical features
le = LabelEncoder()
for col in ['Gender', 'Blood_Pressure', 'Smoking_Status', 'Family_History', 'Symptoms', 'Diagnosis']:
    df[col] = le.fit_transform(df[col])

# Drop Patient_ID
df = df.drop(columns=['Patient_ID'])

# Separate features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test…