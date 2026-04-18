from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
with open('log_reg_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Safely extract and format inputs
        inputs = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'InternetService': request.form['InternetService'],
            'Contract': request.form['Contract'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }
        
        df_input = pd.DataFrame([inputs])
        
        # Apply the exact LabelEncoders saved during training
        for col, le in encoders.items():
            # If a new unknown category appears, fallback to first known class
            if df_input[col][0] in le.classes_:
                df_input[col] = le.transform(df_input[col])
            else:
                df_input[col] = 0

        # Predict churn (0 = No, 1 = Yes)
        pred = model.predict(df_input)[0]
        prediction = 'Churn' if pred == 1 else 'No Churn'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
