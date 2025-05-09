# -*- coding: utf-8 -*-
"""evaluation

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TXom_QGfXjdgBr79VOp7Tp9_Eo37JuE0
"""

# src/evaluation.py

from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy