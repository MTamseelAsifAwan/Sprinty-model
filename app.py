from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from google.generativeai import configure, GenerativeModel

# Flask setup
app = Flask(__name__)

# Load Gemini API Key
GEMINI_API_KEY = "AIzaSyB3KsQChR4Ng1zq3dorOcmdmf5U-w8pfHk"  # Replace with your Gemini API Key
configure(api_key=GEMINI_API_KEY)
model_gemini = GenerativeModel('gemini-2.0-flash')

# Load data (make sure 'Book1data.csv' is present in the same directory as app.py)
data = pd.read_csv('Book1data.csv', encoding='latin1')
data['Created '] = pd.to_datetime(data['Created'])
data['Resolved'] = pd.to_datetime(data['Resolved'])
data['Duration'] = (data['Resolved'] - data['Created ']).dt.days
data = data.drop(['Resolved'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Prepare features and target
X = data.drop(['Duration', 'Created '], axis=1)
y = data['Duration']
columns = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LSTM model
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
model_path = "lstm_task_duration_model.h5"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = Sequential([
        LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train_reshaped, y_train, epochs=20, batch_size=64, validation_data=(X_test_reshaped, y_test), callbacks=[early_stopping])
    model.save(model_path)

# Preprocess task input
def preprocess_task_input(task_name, priority, labels, columns):
    task_name_features = [1 if 'bug' in task_name.lower() else 0]
    priority_numeric = 1 if priority.lower() == 'high' else 0

    label_features = np.zeros(len(columns) - 2)
    for label in labels:
        label_lower = label.lower()
        if label_lower in columns:
            try:
                label_index = list(columns).index(label_lower) - 2
                if 0 <= label_index < len(label_features):
                    label_features[label_index] = 1
            except:
                pass

    features = task_name_features + [priority_numeric] + label_features.tolist()
    input_features = np.zeros(len(columns))
    input_features[0] = task_name_features[0]
    input_features[1] = priority_numeric
    for i, feature in enumerate(label_features):
        try:
            column_name = list(columns)[i + 2]
            if column_name in [l.lower() for l in labels]:
                input_features[i + 2] = 1
        except IndexError:
            pass

    return input_features

# LSTM Prediction function
def predict_with_lstm(lstm_input):
    reshaped = lstm_input.reshape(1, 1, -1)
    lstm_pred = model.predict(reshaped)[0][0]
    return float(lstm_pred)  # Convert to float to avoid type issues with timedelta

# Gemini Prediction function
def predict_with_gemini(task_name, priority, labels, created_date):
    prompt = f"""
Task Description: {task_name}
Priority: {priority}
Labels: {', '.join(labels)}
Created Date: {created_date.strftime('%Y-%m-%d')}

Estimate task duration in days as a single integer without explanations.
"""
    try:
        response = model_gemini.generate_content(prompt)
        prediction = response.text.strip()
        return int(prediction)  # Ensure it's an integer (not float32)
    except:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        task_name = data.get('task_name')
        priority = data.get('priority')
        labels = data.get('labels', [])
        created_date_str = data.get('created_date')
        
        if not task_name or not priority or not created_date_str:
            return jsonify({"error": "Missing required fields"}), 400
        
        created_date = datetime.strptime(created_date_str, '%Y-%m-%d')
        
        features = preprocess_task_input(task_name, priority, labels, columns)
        lstm_duration = predict_with_lstm(features)
        lstm_end_date = created_date + timedelta(days=lstm_duration)

        gemini_duration = predict_with_gemini(task_name, priority, labels, created_date)
        gemini_end_date = None
        if gemini_duration:
            gemini_end_date = created_date + timedelta(days=gemini_duration)

        average_duration = (lstm_duration + gemini_duration) / 2 if gemini_duration else None
        average_end_date = created_date + timedelta(days=average_duration) if average_duration else None

        response = {
            "lstm_duration": round(lstm_duration, 2),
            "lstm_end_date": lstm_end_date.strftime('%Y-%m-%d'),
            "gemini_duration": gemini_duration,
            "gemini_end_date": gemini_end_date.strftime('%Y-%m-%d') if gemini_end_date else None,
            "average_end_date": average_end_date.strftime('%Y-%m-%d') if average_end_date else None
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
