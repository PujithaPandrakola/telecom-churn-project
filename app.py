from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and data
try:
    model = joblib.load('churn_gb_model.pkl')
    data = pd.read_excel('P585 Churn.xlsx')
except Exception as e:
    print(f"Error loading files: {e}")
    raise

data.columns = [c.strip().lower().replace(' ', '.') for c in data.columns]
data.replace(['Nan', 'NA', '', None, pd.NA], np.nan, inplace=True)
states = sorted(data['state'].dropna().unique())
area_codes = sorted(data['area.code'].dropna().unique())

@app.route('/')
def home():
    return render_template('index.html', states=states, area_codes=area_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from form
        input_data = {
            'state': request.form['state'],
            'area.code': request.form['area_code'],
            'account.length': float(request.form['account_length']),
            'voice.plan': request.form['voice_plan'],
            'voice.messages': float(request.form['voice_messages']),
            'intl.plan': request.form['intl_plan'],
            'intl.mins': float(request.form['intl_mins']),
            'intl.calls': float(request.form['intl_calls']),
            'intl.charge': float(request.form['intl_charge']),
            'day.mins': float(request.form['day_mins']),
            'day.calls': float(request.form['day_calls']),
            'day.charge': float(request.form['day_charge']),
            'eve.mins': float(request.form['eve_mins']),
            'eve.calls': float(request.form['eve_calls']),
            'eve.charge': float(request.form['eve_charge']),
            'night.mins': float(request.form['night_mins']),
            'night.calls': float(request.form['night_calls']),
            'night.charge': float(request.form['night_charge']),
            'customer.calls': float(request.form['customer_calls'])
        }
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        churn_label = 'Yes' if prediction == 1 else 'No'

        return render_template('result.html', churn=churn_label, probability=f"{probability:.2%}", inputs=input_data)
    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
