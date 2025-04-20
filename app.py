from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import logging
import sys
import re
from datetime import datetime

# Set up logging to console
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

# Function to clean model name (remove year prefix)
def clean_model_name(model_name):
    if model_name:
        return re.sub(r'^\d{4}\s+', '', model_name).strip()
    return model_name

@app.route('/')
def home():
    # Get valid model names from encoder
    model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        logger.debug("Received form data: %s", form_data)

        # Define required fields from form (exclude computed or optional fields)
        required_form_fields = [
            'Manufacturing_year', 'KM driven', 'Mileage', 'Engine capacity',
            'Power', 'Seats', 'Model Name', 'Fuel type', 'Transmission', 'Ownership'
        ]
        missing_fields = [field for field in required_form_fields if field not in form_data or not form_data[field].strip()]
        if missing_fields:
            logger.error("Missing required form fields: %s", missing_fields)
            model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
            return render_template('index.html', prediction_text=f'Error: Missing data for {", ".join(missing_fields)}.', model_names=model_names)

        input_data = {}
        # Handle numeric columns
        for col in num_cols:
            if col == 'Car_Age':
                continue  # Computed later
            value = form_data.get(col, '').strip()
            if not value or not value.replace('.', '').replace('-', '').isdigit():
                logger.error("Invalid numeric input for %s: %s", col, value)
                model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
                return render_template('index.html', prediction_text=f'Error: Invalid numeric input for {col}.', model_names=model_names)
            input_data[col] = float(value)

        # Handle categorical columns with defaults for optional fields
        for col in cat_cols:
            if col in form_data:
                value = form_data[col].strip()
                if col == 'Model Name':
                    value = clean_model_name(value)  # Clean model name
                input_data[col] = value if value else ''
            else:
                # Default values for optional categorical fields
                input_data[col] = ''  # Empty string for categorical fields like Spare key, Imperfections, Repainted Parts

        # Calculate Car_Age
        year = form_data.get('Manufacturing_year', '').strip()
        if not year or not year.isdigit():
            logger.error("Invalid Manufacturing_year: %s", year)
            model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
            return render_template('index.html', prediction_text='Error: Invalid Manufacturing Year.', model_names=model_names)
        car_age = datetime.now().year - int(year)
        if car_age < 0 or car_age > 100:
            logger.error("Invalid Car_Age: %s", car_age)
            model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
            return render_template('index.html', prediction_text='Error: Invalid Manufacturing Year.', model_names=model_names)
        input_data['Car_Age'] = car_age

        logger.debug("Processed input data: %s", input_data)

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        try:
            if hasattr(encoder, 'handle_unknown') and encoder.handle_unknown != 'ignore':
                logger.warning("Encoder does not have handle_unknown='ignore'. Consider updating encoder.")
            cat_encoded = encoder.transform(input_df[cat_cols])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
            input_df = pd.concat([input_df.drop(cat_cols, axis=1), cat_encoded_df], axis=1)
        except ValueError as e:
            logger.error("Encoding error: %s. Categories might not match training data.", str(e))
            model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
            return render_template('index.html', prediction_text='Error: Invalid Model Name. Please select a valid car model from the dropdown.', model_names=model_names)

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

        model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:.2f}', model_names=model_names)

    except ValueError as e:
        logger.error("ValueError: %s", str(e))
        model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
        return render_template('index.html', prediction_text='Error: Invalid numeric input. Please enter valid numbers.', model_names=model_names)
    except KeyError as e:
        logger.error("KeyError: %s", str(e))
        model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
        return render_template('index.html', prediction_text='Error: Missing data. Please fill all fields.', model_names=model_names)
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        model_names = encoder.categories_[cat_cols.index('Model Name')] if 'Model Name' in cat_cols else []
        return render_template('index.html', prediction_text='Error: An unexpected issue occurred. Please try again.', model_names=model_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)