# Importing required modules.
import numpy as np
import pandas as pd
from os import write
from PIL import Image
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect

RandomForestClassifier = RandomForestClassifier()

app = Flask(__name__)

df = pd.read_csv('NewDS/diabetes.csv')
#df = pd.read_csv('NewDS/diabetes.csv')

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

user_data = {}

def get_user_input(user_data):
    features = pd.DataFrame(user_data, index=[0])
    return features

@app.route('/')
def default():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def form():
    input_pregnancies = request.form['enteredPregnancies']
    input_glucose = request.form['enteredGlucose']
    input_bloodPressure = request.form['enteredBP']
    input_skinThickness = request.form['enteredSkinThickness']
    input_insulin = request.form['enteredInsulin']
    input_bmi = request.form['enteredBMI']
    input_diabetesPedigreeFunction = request.form['enteredDiabetesPedigreeFunction']
    input_age = request.form['enteredAge']

    # print(input_pregnancies)
    # print(input_glucose)
    # print(input_bloodPressure)
    # print(input_skinThickness)
    # print(input_insulin)
    # print(input_bmi)
    # print(input_diabetesPedigreeFunction)
    # print(input_age)

    user_data = {'input_pregnancies':input_pregnancies,
                 'input_glucose':input_glucose,
                 'input_bloodPressure':input_bloodPressure,
                 'input_skinThickness':input_skinThickness,
                 'input_insulin':input_insulin,
                 'input_bmi':input_bmi,
                 'input_diabetesPedigreeFunction':input_diabetesPedigreeFunction,
                 'input_age':input_age}
    
    user_input = get_user_input(user_data)

    RandomForestClassifier.fit(X_train, Y_train)

    prediction = RandomForestClassifier.predict(user_input)

    # print(type(prediction))

    rf_predict_train = RandomForestClassifier.predict(X_train)
    rf_predict_test = RandomForestClassifier.predict(X_test)

    print("TRAINING Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_train, rf_predict_train)*100))
    print("TESTING Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, rf_predict_test)*100))

    return render_template('index.html', pred=prediction)

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/aboutProject')
# def aboutProject():
#     return render_template('aboutProject.html')

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000, threaded=True)