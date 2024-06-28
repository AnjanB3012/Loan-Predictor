from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as layers
import numpy as np
print(tf.__version__)

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('dnn_model.keras')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        data = request.form
        example_input = [
            float(data['Gender']),
            float(data['Married']),
            float(data['Dependents'])/4,
            float(data['Education']),
            float(data['Self_Employed']),
            float(data['Credit_History']),
            float(data['Property_Area'])/2,
            float(data['ApplicantIncome'])/81000,
            float(data['CoapplicantIncome'])/33837,
            float(data['LoanAmount'])/600,
            float(data['Loan_Amount_Term'])/480
        ]
        example_input = np.array(example_input, dtype=np.float32)

        # Make predictions
        predictions = model.predict(example_input)
        prediction = predictions[0][0]
        print(prediction)
        if prediction<0.5:
            prediction = 'Reject'
        else:
            prediction = 'Approve'
        # Return the prediction result in the rendered HTML page
        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
