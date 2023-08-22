from flask_restful import Resource
from flask import request,jsonify
from flask import make_response
import json
import tensorflow as tf
import sklearn
import numpy as np
import joblib
import pandas as pd

class CardiovascularModel(Resource):
  def post(self):
    json_data = request.get_json()
    gender = float(json_data.get('gender'))
    height = float(json_data.get('height'))
    weight = float(json_data.get('weight'))
    ap_hi = float(json_data.get('bloodpress_high'))
    ap_lo = float(json_data.get('bloodpress_low'))
    cholesterol = float(json_data.get('total_cholesterol'))
    gluc = float(json_data.get('glucose'))
    smoke = float(json_data.get('smoke'))
    alco = float(json_data.get('alco'))
    age_years = float(json_data.get('age'))


    model = joblib.load("C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/Cardiovascular.pkl")
    arr = np.array([gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,1,age_years])
    prediction = model.predict_proba(arr.reshape(1,-1))
    
    list_data = prediction.tolist()
    return jsonify(list_data)
    