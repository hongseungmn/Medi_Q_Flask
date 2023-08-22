from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# 데이터셋 불러오기
data = pd.read_csv('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/data/healthcare-dataset-stroke-data.csv')
# 전처리 객체들
mms = MinMaxScaler()
ss = StandardScaler()
label_encoders = {
    "gender": LabelEncoder(),
    "ever_married": LabelEncoder(),
    "work_type": LabelEncoder(),
    "Residence_type": LabelEncoder()
}
# 전처리 객체들 초기화 함수
def initialize_preprocessing_objects():
    # 라벨 인코더 학습
    for col, encoder in label_encoders.items():
        encoder.fit(data[col])
    # 스케일러 학습
    mms.fit(data[["age", "avg_glucose_level"]])
    data_for_ss = data[["gender", "ever_married", "work_type", "Residence_type"]].apply(LabelEncoder().fit_transform)
    ss.fit(data_for_ss)
# 전처리 객체들 초기화 실행
initialize_preprocessing_objects()
# 입력 데이터 전처리 함수
def preprocess_input_data(input_data):
    # 입력 데이터를 DataFrame으로 변환
    input_df = pd.DataFrame([input_data])
    # 라벨 인코딩
    for col, encoder in label_encoders.items():
        input_df[col] = encoder.transform(input_df[col])
    # 스케일링
    input_df[["age", "avg_glucose_level"]] = mms.transform(input_df[["age", "avg_glucose_level"]])
    input_df[["gender", "ever_married", "work_type", "Residence_type"]] = ss.transform(
        input_df[["gender", "ever_married", "work_type", "Residence_type"]])
    return input_df.iloc[0].to_dict()
# 모델 경로 및 로딩
#model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api', 'stroke_gold_model.xgb')
model = xgb.Booster()
model.load_model('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/stroke_gold_model.xgb')
# 뇌졸중 예측 모델 리소스 정의
class StrokeModel(Resource):
    def post(self):
        json_data = request.get_json()
        preprocessed_data = preprocess_input_data(json_data)
        print(preprocessed_data)
        dmatrix = xgb.DMatrix(np.array(list(preprocessed_data.values())).reshape(1, -1))
        prediction = model.predict(dmatrix)
        return jsonify({'prediction': prediction.tolist()})