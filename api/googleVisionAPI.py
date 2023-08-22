import base64
from google.cloud import vision
from google.oauth2 import service_account
from flask_restful import Resource
from flask_restful import Resource,reqparse
from flask import make_response,jsonify,request
import re
import json
from io import BytesIO
from PIL import Image, ImageDraw



class OCR(Resource):
  def __init__(self):
    #credentials_path:프로젝트 ID,Private Key정보가 있는 .json파일의 경로
    self.credentials_path='./api/sonic-trail-391102-6632f36b2cd9.json'

  def post(self):
    #parser = reqparse.RequestParser()
    #parser.add_argument('base64')
    #args=parser.parse_args()
    data = request.form["base64"]
    #print(args['base64'])
    
    #print(texts)
    return self.detect_labels(data)

  def authenticate_service_account(self):
    '''
    서비스 계정 키를 로드하여 구글 Vision API에 인증하는 함수
    return: 인증 정보 객체
    '''
    #print('self.credentials_path:',self.credentials_path)
    credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    #print('scoped_credentials:', scoped_credentials)
    return scoped_credentials


  def detect_labels(self,image_base64):
    credentials = self.authenticate_service_account()
    image_content = base64.b64decode(image_base64)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    image = vision.Image(content=image_content)
    
    response = client.text_detection(image=image)
    full_text_annotation = response.full_text_annotation.pages
    bigestBox = 0
    titleText = ''
    imageDraw = Image.open(BytesIO(image_content))
    for page in full_text_annotation:  # 페이지 순회
      for block in page.blocks:
        for item in block.paragraphs:
          paragraph = ''
          for code in item.words:
            for text in code.symbols:
              for last in text.text:
                paragraph += last
            paragraph += ' '
          reg = r'[`~!@#$%^&*()_|+\-=?;:\'",.<>{}\[\]\\\//]'
          reg2 = r'[a-zA-Z0-9]'
          resultData = re.sub(reg, '', paragraph)
          resultData = re.sub(reg2, '', resultData)
          print("resultData", resultData)
          start_x = item.bounding_box.vertices[0].x
          start_y = item.bounding_box.vertices[0].y
          end_x = item.bounding_box.vertices[2].x - item.bounding_box.vertices[0].x
          end_y =  item.bounding_box.vertices[3].y - item.bounding_box.vertices[0].y
          boxSize = abs(end_x * end_y)
          if boxSize > bigestBox:
            bigestBox = boxSize
            titleText = resultData
            draw = ImageDraw.Draw(imageDraw)
            # 사각형 좌표와 색상 지정
            top_left = (start_x, start_y)
            bottom_right = (start_x+end_x, start_y+end_y)
            rectangle_color = (0, 255, 0)  # RGB 값 (빨강, 녹색, 파랑)
            # 사각형 그리기
            
            draw.rectangle([top_left, bottom_right], outline=rectangle_color,width=8)
            # 이미지 위에 사각형 그리기
          
          
    print("bigestBox :",bigestBox)
    print("titleText",titleText)
    
    
    output_path = './modified_image.jpg'
    #imageDraw.save(output_path)
    imageDraw = imageDraw.convert("RGB")
    modified_image_byte_io = BytesIO()
    imageDraw.save(modified_image_byte_io, format='JPEG')  # 포맷을 원하는 형식으로 선택
    modified_image_byte_io.seek(0)
    modified_image_bytes = modified_image_byte_io.getvalue()
    return jsonify({'base64':base64.b64encode(modified_image_bytes).decode('utf-8'),'titleText':titleText})
    