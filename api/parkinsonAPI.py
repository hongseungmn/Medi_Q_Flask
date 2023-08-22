from flask_restful import Resource
from flask import request,jsonify
from flask import make_response
import json
import tensorflow as tf
import sklearn
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage import feature
import cv2
import base64
import pickle

class ParkinsonModel(Resource):
    def post(self):
        data = request.form["base64String"]
            # 파일 처리 로직 작성
            # 예를 들어, 파일 저장을 원한다면:
        print(data)
        #with open('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/parkinsons_spiral_model_Rf.pkl', 'rb') as f:
        #  model = pickle.load(f)
        model = joblib.load("C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/parkinsons_spiral_model_Rf.pkl")
        score = test_prediction(model,data).tolist()
        print("-------------------------------------------------")
        image = cv2.imread("C:/HSM/Workspace/pythonEnv/project_kosmo/file_name.png")

        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode()

        # JSON 응답으로 이미지 base64 데이터를 보냄
        #'score' : score
        return jsonify({'image_base64': image_base64})


def quantify_image(image):
    features = feature.hog(image, orientations=9,
                          pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                          transform_sqrt=True, block_norm="L1")
    return features

def test_prediction(model, base64_image_data):
    # get the list of images
    print(base64_image_data)
    # base64 디코드하여 이미지 데이터로 변환
    image_data = base64.b64decode(base64_image_data)
    nparr = np.frombuffer(image_data, np.uint8)
    output_images = []
    # pick 15 images at random
    # NumPy 배열을 이미지로 읽어들이기
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output = image.copy()
    cv2.imwrite("file_image.png", image)
    output = cv2.resize(output, (128, 128))
    # pre-process the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # quantify the image and make predictions based on the extracted features
    features = quantify_image(image)
    preds = model.predict([features])
    print(preds)
    predict_proba = model.predict_proba([features])
    print(predict_proba)
    label = "Parkinsons" if predict_proba[0][1]>0.6 else "Healthy"

    # draw the colored class label on the output image and add it to
    # the set of output images
    color = (0, 255, 0) if label == "Healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    output_images.append(output)
    cv2.imwrite("file_name.png", output)
    print("ok")
    return predict_proba