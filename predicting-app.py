import json
import requests

def predictDiabetes(BMI, Age, Glucose):
    url = 'http://127.0.0.1:5000/'
    data = {'BMI':BMI, 'Age':Age, 'Glucose':Glucose}
    data_json = json.dumps(data)
    headers = {'Content-Type':'application/json'}
    response = requests.post(url, data=data_json, headers=headers)
    result = json.loads(response.text)
    return result

if __name__ == '__main__':
    BMI = input("Enter your BMI: ")
    Age = input("Enter your Age: ")
    Glucose = input("Enter your Glucose: ")

    predictions = predictDiabetes(BMI, Age, Glucose)

    print("\nPrediction:")
    print("Diabetic" if predictions['prediction'] == 1 else "Not diabetic")
    print("Confidence", predictions['confidence'], "%")
    print("")