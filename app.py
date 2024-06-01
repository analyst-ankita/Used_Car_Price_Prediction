from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('lr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    data = [
        int(request.form['Year']),
        float(request.form['Selling_Price']),
        int(request.form['Kms_Driven']),
        int(request.form['Fuel_Type']),
        int(request.form['Seller_Type']),
        int(request.form['Transmission'])
    ]
    
    final_features = np.array(data).reshape(1, -1)
    
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Estimated Present Price in lakhs: ₹{output}')

if __name__ == "__main__":
    app.run(debug=True)
