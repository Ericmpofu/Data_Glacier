from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('linear_regression_model.joblib')

# Home route - render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction form - render prediction_form.html
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        age = float(request.form['age'])

        # Predict using the model
        prediction = model.predict([[age]])

        # Render prediction_result.html with the result
        return render_template('prediction_result.html', prediction=prediction[0])

    # If GET request, render prediction_form.html
    return render_template('prediction_form.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
