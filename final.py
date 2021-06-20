#Importing the libraries
import pickle
from flask import Flask,request,jsonify,render_template
import requests
import numpy as np

#Global variables
app = Flask(__name__)
loadedModel = pickle.load(open('diabetes.sav','rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template['diabetes.html']

#User defined functions
@app.route('/predict', methods=['POST'])
def predict():
    #Getting the input
    bmi = int(request.form['bmi'])
    age = int(request.form['age'])
    glucose = int(request.form['glucose'])

    print('Age:',age)
    print('BMI:',bmi)
    print('Glucose:',glucose)

    #Making predictions
    prediction = loadedModel.predict([[glucose,bmi,age]])[0]
    confidence = loadedModel.predict_proba([[glucose, bmi, age]])

    #Returning the predictions
    if prediction == 1:
        sendPrediction = 'Diabetic'
    else:
        sendPrediction = 'Not Diabetic'
        
    sendConfidence = str(round(np.amax(confidence[0]*100),2))

    print(sendPrediction)
    print(sendConfidence)

    return render_template('diabetes.html', diagnosis_output = sendPrediction, confidence_output = sendConfidence)

#Main function
if __name__ == '__main__':
    app.run(debug=True)