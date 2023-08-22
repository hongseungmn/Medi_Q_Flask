import json
def deleteDuplicate(file_path):
  json_data = {}
  with open(file_path, "r+", encoding="utf-8") as json_file:
      json_data = json.load(json_file)
      
  key_to_check = 'id'
  # 중복 값을 제거하기 위한 임시 딕셔너리를 생성합니다.
  temp_dict = {}

  # 중복 값을 제거합니다.
  unique_data = {}
  unique_set = set()

  # 중복 값을 제거하고 "val" 값을 증가시킵니다.
  for node in json_data["nodes"]:
      key = node["id"]
      if key in unique_set:
          unique_data[key]["val"] += 1
      else:
          unique_data[key] = node
          unique_set.add(key)

  # 결과를 리스트로 변환합니다.
  result = list(unique_data.values())
  print(result)
  json_data['nodes'] = result


  # 중복 값을 제거할 기준으로 사용할 속성을 지정합니다.
  keys_to_check = ["source", "target"]

  # 중복 값을 제거하기 위한 임시 집합(set)을 생성합니다.
  temp_set = set()

  # 중복 값을 제거한 결과를 저장할 리스트를 생성합니다.
  result_links = []

  # 중복 값을 제거합니다.
  for d in json_data['links']:
      key_values = tuple(d[key] for key in keys_to_check)
      if key_values not in temp_set:
          temp_set.add(key_values)
          result_links.append(d)
  json_data['links'] = result_links
  print(result_links)

  with open(file_path, 'w', encoding='utf-8') as outfile:
    json.dump(json_data, outfile, indent=2,ensure_ascii=False)
