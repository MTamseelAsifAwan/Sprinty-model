import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from google.generativeai import configure, GenerativeModel
from flask import Flask, request, jsonify
import os

# Flask Setup
app = Flask(__name__)

# Load your Gemini API key (replace with your actual key)
GEMINI_API_KEY = "AIzaSyB3KsQChR4Ng1zq3dorOcmdmf5U-w8pfHk"
configure(api_key=GEMINI_API_KEY)
model_gemini = GenerativeModel('gemini-2.0-flash')

# Load Data
data = pd.read_csv('Book1data.csv', encoding='latin1')

# 1. Preprocessing
data['Created '] = pd.to_datetime(data['Created'])
data['Resolved'] = pd.to_datetime(data['Resolved'])
data['Duration'] = (data['Resolved'] - data['Created ']).dt.days
data = data.drop(['Resolved'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Separate features (X) and target variable (y)
X = data.drop(['Duration', 'Created '], axis=1)
y = data['Duration']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LSTM Model Setup
model_path = "lstm_task_duration_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1), activation='relu', return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model.fit(X_train_reshaped, y_train, epochs=20, batch_size=16, validation_data=(X_test_reshaped, y_test), callbacks=[early_stopping])
    model.save(model_path)

# Helper Functions for LSTM Prediction
# Helper Functions for LSTM Prediction
def predict_with_lstm(features):
    # Reshaping the input for LSTM
    features_reshaped = features.reshape(1, 1, features.shape[0])  # Reshape to (1, 1, num_features)
    prediction = model.predict(features_reshaped)[0][0]
    return float(prediction)

# Preprocess Task Input (adjusted)
def preprocess_task_input(task_name, priority, labels, columns):
    task_name_features = [1 if 'bug' in task_name.lower() else 0]
    priority_numeric = 1 if priority.lower() == 'high' else 0
    label_features = np.zeros(len(columns) - 2)
    for label in labels:
        if label.lower() in columns:
            try:
                label_index = list(columns).index(label.lower()) - 2
                if label_index >= 0 and label_index < len(label_features):
                    label_features[label_index] = 1
            except ValueError:
                pass
    input_features = np.zeros(len(columns))
    input_features[0] = task_name_features[0]
    input_features[1] = priority_numeric
    for i, feature in enumerate(label_features):
        try:
            column_name = list(columns)[i + 2]
            if column_name in [label.lower() for label in labels]:
                input_features[i + 2] = 1
        except IndexError:
            pass
    return input_features

# Gemini Prediction Function
def predict_with_gemini(task_name, priority, labels, created_date):
    prompt = f"""
    You are an expert project manager tasked with estimating task durations.

    Task Description: {task_name}
    Priority: {priority}
    Labels: {', '.join(labels)}
    Created Date: {created_date.strftime('%Y-%m-%d')}

    Based on your experience, please estimate the duration of this task in days. 
    Think step-by-step and consider the following factors:

    * Task Complexity: Analyze the task description and labels to assess its complexity.
    * Priority: High priority tasks generally require quicker completion.
    * Potential Roadblocks: Consider potential challenges or dependencies that might affect the duration.

    After careful consideration, provide the estimated duration in days as a single integer.
    Avoid any other text or explanations in your response.""" 

    try:
        response = model_gemini.generate_content(prompt)
        gemini_prediction_text = response.text.strip()
        print(f"Gemini Raw Output: '{gemini_prediction_text}'")  # Print raw output for debugging
        
        try:
            duration_gemini = int(gemini_prediction_text)
            predicted_end_date_gemini = created_date + timedelta(days=duration_gemini)
            return predicted_end_date_gemini, duration_gemini
        except ValueError:
            print(f"Warning: Could not convert Gemini output '{gemini_prediction_text}' to an integer.")
            return None, None

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None, None
# Flask Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Task Duration Prediction API using LSTM and Gemini!"}), 200

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Re-train or load the model (already done above)
        return jsonify({'message': 'Model trained successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['POST'])
def test_prediction():
    try:
        payload = request.get_json()
        task_name = payload['task_name']
        priority = payload['priority']
        labels = payload['labels']
        created_date = pd.to_datetime(payload['created_date'])
        
        # Process and predict using LSTM
        task_features = preprocess_task_input(task_name, priority, labels, X.columns)
        lstm_pred_days = predict_with_lstm(np.array(task_features))
        predicted_end_date_lstm = created_date + timedelta(days=lstm_pred_days)

        # Predict using Gemini
        predicted_end_date_gemini, duration_gemini = predict_with_gemini(task_name, priority, labels, created_date)

        # Calculate average predicted end date
        avg_duration = (lstm_pred_days + duration_gemini) / 2 if duration_gemini else lstm_pred_days
        average_predicted_end_date = created_date + timedelta(days=avg_duration)

        # Return predictions
        result = {
            "lstm_duration_days": round(lstm_pred_days, 2),
            "lstm_predicted_end_date": predicted_end_date_lstm.strftime('%Y-%m-%d'),
            "gemini_duration_days": duration_gemini if duration_gemini else "N/A",
            "gemini_predicted_end_date": predicted_end_date_gemini.strftime('%Y-%m-%d') if predicted_end_date_gemini else "N/A",
            "average_predicted_end_date": average_predicted_end_date.strftime('%Y-%m-%d')
        }

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
