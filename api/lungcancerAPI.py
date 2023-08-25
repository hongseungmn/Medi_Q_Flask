from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import pandas as pd
import joblib


# 폐암 예측 모델 로드
'''
                precision    recall  f1-score   support

           0       0.88      0.97      0.92        60
           1       0.96      0.86      0.91        59

    accuracy                           0.92       119
   macro avg       0.92      0.92      0.92       119
weighted avg       0.92      0.92      0.92       119
'''
#model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lung_cancer_LR_model.pkl')
model = joblib.load('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/lung_cancer_LR_model.pkl')

# 모델 훈련 시 사용한 특성명 목록
attributes = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", 
    "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING", 
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

# 'AGE' 스케일링을 위해 저장된 scaler 객체 로드
#scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'age_scaler.pkl')
scaler = joblib.load('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/age_scaler.pkl')

# 입력 데이터 전처리 함수
def preprocess_input_data(input_data):
    input_df = pd.DataFrame([input_data])
    
    # 'AGE' 열의 스케일링
    input_df['AGE'] = scaler.transform(input_df[['AGE']])
    
    return input_df

# 폐암 예측 모델 API 엔드포인트
class LungCancerModel(Resource):
    def post(self):
        # 클라이언트로부터 데이터 받기
        json_data = request.get_json()
        # 데이터 전처리
        preprocessed_data = preprocess_input_data(json_data)
        # 예측 확률 수행
        prediction_proba = model.predict_proba(preprocessed_data)
        # 예측 결과 반환
        print('prediction_proba', prediction_proba[0][1])
        return jsonify({'prediction_proba': prediction_proba[0][1]})
        


