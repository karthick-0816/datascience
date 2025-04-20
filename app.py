from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, encoder, and column lists
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('num_cols.pkl', 'rb') as f:
    num_cols = pickle.load(f)
with open('cat_cols.pkl', 'rb') as f:
    cat_cols = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {}
    for col in num_cols + cat_cols:
        if col == 'Year':
            year = int(request.form[col])
            input_data['Car_Age'] = 2025 - year
        elif col in num_cols:
            input_data[col] = float(request.form[col])
        else:
            input_data[col] = request.form[col]

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    cat_encoded = encoder.transform(input_df[cat_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
    input_df = pd.concat([input_df.drop(cat_cols, axis=1), cat_encoded_df], axis=1)

    # Ensure all columns match training data
    feature_cols = model.feature_names_in_
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:.2f}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)