# -*- coding: utf-8 -*-
"""Support Vector Machine(SVM).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1egO5OliA6fcoqbsSZvJ4nbpOWW0q3xNj
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Loading the dataset
cell_df = pd.read_csv('sample_data/cell_samples.csv')
print("✅ Dataset loaded successfully!")

# Check initial info
print("Shape:", cell_df.shape)
print("Class distribution:\n", cell_df['Class'].value_counts())

# Plot sample data
benign_df = cell_df[cell_df['Class'] == 2][:200]
malignant_df = cell_df[cell_df['Class'] == 4][:200]

axes = benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='Benign')
malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='Malignant', ax=axes)

plt.title("Cell Samples: Clump vs. Uniformity of Cell Size")
plt.xlabel("Clump Thickness")
plt.ylabel("Uniformity of Cell Size")
plt.grid(True)
plt.show()

# Clean data: convert BareNuc to numeric
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

# Feature selection
features = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
            'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']
X = cell_df[features].values
y = cell_df['Class'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Train SVM
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(X_train, y_train)

# Predict
y_predict = classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_predict)
print("\n📊 Classification Report:\n", classification_report(y_test, y_predict))
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(classifier, 'svm_cell_classifier.joblib')
print("💾 Model saved as 'svm_cell_classifier.joblib'")

