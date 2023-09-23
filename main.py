from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'models\sc.sav')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form submission
        item_weight = float(request.form['item_weight'])
        item_fat_content = int(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])

        # Add more code to get other input features

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'item_weight': [item_weight],
            'item_fat_content': [item_fat_content],
            'item_visibility': [item_visibility]
            # Add other features here
        })

        # Make the prediction using the model
        prediction = model.predict(input_data)

        # Convert prediction to a human-readable format
        prediction_result = "Unknown"
        if prediction[0] == 0:
            prediction_result = "Low Sales"
        elif prediction[0] == 1:
            prediction_result = "Medium Sales"
        elif prediction[0] == 2:
            prediction_result = "High Sales"

        return jsonify({'prediction': prediction_result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
