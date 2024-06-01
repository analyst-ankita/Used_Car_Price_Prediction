from flask import Flask, request, render_template
import numpy as np
import utils
import config

app = Flask(__name__)
model = utils.load_model()

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
    app.run(port=config.PORT_NUMBER, host= '0.0.0.0' , debug=True)
