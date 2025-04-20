from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import logging
import sys

# Set up logging to console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model, scaler, encoder, and column lists
try:
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
    logger.debug("Model, scaler, encoder, and column lists loaded successfully.")
except Exception as e:
    logger.error("Failed to load pickled files: %s", str(e))
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        logger.debug("Received form data: %s", form_data)

        input_data = {}
        for col in num_cols + cat_cols:
            if col == 'Manufacturing_year':
                year = request.form.get(col)
                input_data['Car_Age'] = 2025 - int(year) if year and year.isdigit() else 0
            elif col in num_cols:
                value = request.form.get(col)
                input_data[col] = float(value) if value and value.replace('.', '').isdigit() else 0.0
            else:
                input_data[col] = request.form.get(col, '')

        logger.debug("Processed input data: %s", input_data)

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        try:
            cat_encoded = encoder.transform(input_df[cat_cols])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
            input_df = pd.concat([input_df.drop(cat_cols, axis=1), cat_encoded_df], axis=1)
        except ValueError as e:
            logger.error("Encoding error: %s. Categories might not match training data.", str(e))
            return render_template('index.html', prediction_text='Error: Invalid category. Please use options from the dropdowns.')

        # Ensure all columns match training data
        feature_cols = model.feature_names_in_
        logger.debug("Model expected features: %s", feature_cols)
        logger.debug("Input DataFrame columns before alignment: %s", input_df.columns.tolist())
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]
        logger.debug("Input DataFrame columns after alignment: %s", input_df.columns.tolist())

        # Scale features
        input_scaled = scaler.transform(input_df)
        logger.debug("Input scaled successfully.")

        # Predict
        prediction = model.predict(input_scaled)[0]
        logger.debug("Prediction: %s", prediction)

        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:.2f}')

    except ValueError as e:
        logger.error("ValueError: %s", str(e))
        return render_template('index.html', prediction_text='Error: Invalid numeric input. Please enter valid numbers.')
    except KeyError as e:
        logger.error("KeyError: %s", str(e))
        return render_template('index.html', prediction_text='Error: Missing data. Please fill all fields.')
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        return render_template('index.html', prediction_text='Error: An unexpected issue occurred. Please try again.')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)