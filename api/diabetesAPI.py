from flask_restful import Resource
from flask import request,jsonify
from flask import make_response
import json
import tensorflow as tf
import sklearn
import numpy as np
import joblib

class DiabetesModel(Resource):
  def post(self):
    json_data = request.get_json()
    age = float(json_data.get('age'))
    print(age)
    bmi = float(json_data.get('bmi'))
    print(bmi)
    glucose = float(json_data.get('glucose'))
    print(glucose)
    bloodpress = float(json_data.get('bloodpress'))
    print(bloodpress)

    model = joblib.load("C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/DiabetesModel.pkl")
    arr = np.array([age,bmi,glucose,bloodpress])
    prediction = model.predict_proba(arr.reshape(1,-1))
    
    list_data = prediction.tolist()
    return jsonify(list_data)