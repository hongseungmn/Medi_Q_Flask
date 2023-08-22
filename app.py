from flask import Flask, request, jsonify, render_template  # 서버 구현을 위한 Flask 객체 import
from flask_restful import Api
from flask_cors import CORS

from api.reviewAPI import Review
from api.diabetesAPI import DiabetesModel
from api.cardiovascularAPI import CardiovascularModel
from api.parkinsonAPI import ParkinsonModel
from api.strokeAPI import StrokeModel
from api.googleVisionAPI import OCR
from api.skinLesionAPI import SkinLesionModel

app = Flask(__name__)  

CORS(app)
api = Api(app)
api.add_resource(Review,"/review/<product_id>")
api.add_resource(DiabetesModel,"/diabetes")
api.add_resource(CardiovascularModel,"/cardiovascular")
api.add_resource(ParkinsonModel,"/parkinson")
api.add_resource(StrokeModel,'/stroke')
api.add_resource(OCR,"/ocr")
api.add_resource(SkinLesionModel, '/SkinLesionModel')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
