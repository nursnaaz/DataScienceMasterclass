import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load preprocessed data
preprocessed_data = pd.read_csv("preprocessed_data.csv")

# Separate features and labels
X_train = preprocessed_data[['feature1', 'feature2']]
y_train = preprocessed_data['label']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "trained_model.joblib")
