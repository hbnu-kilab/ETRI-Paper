import os
import json
from PIL import Image

API_KEYS_PATH = "api_keys.json"
RAW_JSON_DIR = "/home/cwhjpaper/data/json/raw"
IMAGE_DIR = "/home/cwhjpaper/data/images"
CROPPED_IMAGE_DIR = "/home/cwhjpaper/data/cropped_images"
IMAGE_DATA_JSON  = "/home/cwhjpaper/data/json/processed/image_metadata.json"
EDGE_CASE_JSON = "/home/cwhjpaper/data/json/processed/edge_case.json"
ERROR_LOG_JSON = "/home/cwhjpaper/data/json/processed/select_errors_log.json"

'''
edge case의 이미지 crop 사이즈를 키워서 저장하기 위한 파일

edge_cases 정보: EDGE_CASE_JSON
처리되지 않은 edge case 정보: ERROR_LOG_JSON
edge_cases 개수: 87 
=> 사이즈 키우고 다시 돌리니 29개의 edge case 처리 완료함
미처리된 edge_cases 개수: 58

edge case 형태: {
    "core_1738.json": [
        {
            "region_id": "2333448_0",
            "response": "I'm unable to view the image, so I can't determine the main object or select the most accurate caption."
        }
    ]
}
'''
with open(ERROR_LOG_JSON, 'r', encoding='utf-8') as file:
    error_cases = json.load(file)
    print("Number of currunt error data: ", len(error_cases))
    
with open(EDGE_CASE_JSON, 'r', encoding='utf-8') as file:
    edge_cases = json.load(file)
for json_path in list(edge_cases.keys()):
    edge_json_path = os.path.join(RAW_JSON_DIR, json_path)
    region_id = edge_cases[json_path][0]["region_id"]
    try:
        with open(edge_json_path, 'r', encoding='utf-8') as file:
            edge_json = json.load(file)
        vgid = list(edge_json.keys())[0]
        target_region = next((target for target in edge_json[vgid]["regions"] if target["id"] == region_id), None)
        full_image_path = os.path.join(IMAGE_DIR, f"{vgid}.jpg")
        
        if not os.path.exists(full_image_path):
            print("파일이 존재하지 않습니다:", full_image_path)
        elif target_region:
            with Image.open(full_image_path) as img:
                print(f"이미지 처리 중: {vgid}")
                # 원복 코드
                # x = target_region['x'] 
                # y = target_region['y']
                # width = target_region['width']
                # height = target_region['height']
                
                # 임의로 20px씩 늘림
                x = target_region['x']-10 if target_region['x'] > 10 else 0 
                y = target_region['y']-10 if target_region['y'] > 10 else 0
                width = target_region['width']+20
                height = target_region['height']+20
                
                cropped_img = img.crop((x,y,x+width,y+height))

                cropped_path = os.path.join(CROPPED_IMAGE_DIR, f"{vgid}/{region_id}.jpg")
                cropped_img.save(cropped_path)
                print(f"저장 완료: {cropped_path}")
    except FileNotFoundError:
        print("file not found")
        