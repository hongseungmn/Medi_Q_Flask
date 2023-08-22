from flask_restful import Resource
from flask import make_response
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
import sklearn
import numpy as np
import joblib
import pandas as pd
import cx_Oracle
import os
import re
from konlpy.tag import Mecab
import pickle
import json
from api.reviewMethod.DeleteDuplicated import deleteDuplicate
import requests

class Review(Resource):
    def get(self,product_id):
      file_path = "C:/HSM/Workspace/pythonEnv/project_kosmo/resources/data/"+product_id+".json"
      with open(file_path, "rt", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
      return json_data



    def put(self,product_id):
      #self.sentiment_predict("별로인듯합니다...")
      connect = cx_Oracle.connect("PROJECT", "PROJECT", "localhost:1521/xe")
      df=pd.read_sql_query(
                  f""" 
                          SELECT content,R_PRODUCTNO,PRODUCTNAME 
                          FROM REVIEWTABLE RIGHT OUTER JOIN FOODTABLE 
                          ON REVIEWTABLE.R_PRODUCTNO =FOODTABLE.NO  
                          WHERE R_PRODUCTNO = {product_id} """ 
                          , con = connect)
      connect.close()
      try:
        product_name = df.loc[0, 'PRODUCTNAME']
      except:
        return '갱신할 리뷰 데이터가 없습니다'

      file_path = "C:/HSM/Workspace/pythonEnv/project_kosmo/resources/data/"+product_id+".json"
      json_data = {
        "nodes": [
          
        ],
        "links": [
          
        ]
      }
      with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(json_data, outfile, indent=2,ensure_ascii=False)
      print('title : ',product_name)
      for index, row in df.iterrows():
          content = row['CONTENT']
          if(content == None):continue
          self.sentiment_predict(content,product_name,product_id)
      
      deleteDuplicate(file_path)
      return '완료'
    
    def sentiment_predict(self,new_sentence,product_name,product_id):
      mecab = Mecab('C:/mecab/mecab-ko-dic')
      max_len = 80
      stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게','합' ,'아요','니다']
      origin_sentence = new_sentence
      new_sentence = re.sub(r'[^가-힣 ]','', new_sentence)
      new_sentence = mecab.morphs(new_sentence)
      new_sentence = [word for word in new_sentence if not word in stopwords]
      loaded_model = load_model('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/Review_Model.h5')
      # Tokenizer를 불러옴
      with open('C:/HSM/Workspace/pythonEnv/project_kosmo/resources/model/Review_Model_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
      encoded = tokenizer.texts_to_sequences([new_sentence])
      pad_new = pad_sequences(encoded, maxlen = max_len)
      print(new_sentence)
      score = float(loaded_model.predict(pad_new))
      if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
        load_data(product_name,3,new_sentence,origin_sentence,product_id)
      else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
        load_data(product_name,1,new_sentence,origin_sentence,product_id)



def load_data(productName,group,new_sentence,origin_sentence,product_id):
    file_path = "C:/HSM/Workspace/pythonEnv/project_kosmo/resources/data/"+product_id+".json"
    json_data = {
      "nodes": [
        
      ],
      "links": [
        
      ]
    }
    with open(file_path, "rt", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    
    for index,word in enumerate(new_sentence):
      json_data['nodes'].append({
          "id" : word,
          "group" : group,
          "val" : 1
      })
      json_data['links'].append({
          "source" : word,
          "target" : productName,
          "value" : 1
      })
      if index == len(new_sentence)-1:
          json_data['nodes'].append({
            "id" : origin_sentence,
            "group" : group,
            "val" : 1
          })
          json_data['links'].append({
            "source" : origin_sentence,
            "target" : new_sentence[0],
            "value" : 1
          })
    json_data['nodes'].append({
          "id" : productName,
          "group" : 2,
          "val" : 1
      })
    with open(file_path, 'w', encoding='utf-8') as outfile:
      json.dump(json_data, outfile, indent=2,ensure_ascii=False)
    
    
