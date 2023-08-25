from flask_restful import Resource, Api
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

# 클래스 인덱스와 피부 병명 사이의 매핑 정의
class_names = {
    0: "akiec",  # "광선 각화증/보웬병(자외선 노출, 초기 피부암)",
    1: "bcc",    # "기저세포암(가장 흔한 피부암)",
    2: "bkl",    # "지루성 각화증(검버섯)",
    3: "df",     # "피부 섬유종(쥐젖)",
    4: "mel",    # "흑색종(피부암)",
    5: "nv",     # "갈색 세포모반(점)",
    6: "vasc"    # "혈관종"
}

# 모델의 그래디언트 업데이트 설정
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# 모델 초기화 함수
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "densenet":
        model_ft = models.densenet121(weights=None)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    return model_ft, input_size

# 현재 스크립트의 디렉토리 경로 획득
script_dir = os.path.dirname(os.path.abspath(__file__))

# 모델 파일의 절대 경로 생성
model_path = os.path.join(script_dir, 'best_skin_lesion_model.pth')

# 모델 로딩
model_weights = torch.load('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/best_skin_lesion_model (1).pth', map_location=torch.device('cpu'))

# 모델 초기화 및 가중치 로드
num_classes = 7
model_densenet, _ = initialize_model("densenet", num_classes, feature_extract=True, use_pretrained=False)
model_densenet.load_state_dict(model_weights)
model_densenet.eval()

app = Flask(__name__)
CORS(app)
api = Api(app)

# 이미지 전처리 함수
def preprocess_image(image_path):
    # 이미지 로드 및 전처리
    image = Image.open(image_path)
    
    # Jupyter Notebook에서 사용된 평균 및 표준편차 값
    norm_mean = [0.7630423088417134, 0.545648601460742, 0.5700468609021178]
    norm_std = [0.08914092883332351, 0.11945825343562663, 0.13270029760742103]
    
    # 전처리 변환 정의
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    return transformations(image).unsqueeze(0)  # 배치 차원 추가

class SkinLesionModel(Resource):
    def post(self):
        # 사용자로부터 이미지 파일 받기
        image_file = request.files['image']
        
        # 이미지 파일 임시 경로에 저장
        image_path = os.path.join(script_dir, 'uploaded_image.jpg')
        image_file.save(image_path)
        
        # 이미지 전처리
        tensor_image = preprocess_image(image_path)
        
        # 모델로 예측 수행
        with torch.no_grad():
            outputs = model_densenet(tensor_image)
            probs = F.softmax(outputs, dim=1)[0].tolist()
            predicted_class_index = torch.argmax(outputs, dim=1).item()
            
        # 결과 반환
        return jsonify({
            "predicted_disease": class_names[predicted_class_index],
            "confidence": '{:.10f}'.format(probs[predicted_class_index]),
            "all_probabilities": {class_names[i]: '{:.10f}'.format(prob) for i, prob in enumerate(probs)}
        })
