from process_image_data import (
    load_gpt_api_key,
    load_data,
    ask_gpt,
    process_json_file
)
import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import openai


# 경로 설정
API_KEYS_PATH = "api_keys.json"
RAW_JSON_DIR = "/home/cwhjpaper/data/json/raw"
CROPPED_IMAGE_DIR = "/home/cwhjpaper/data/cropped_images"
PROCESSED_JSON = "/home/cwhjpaper/data/json/processed/copy_image_metadata.json"
ERROR_LOG_PATH = "/home/cwhjpaper/data/json/processed/select_errors_log_caption_test.json"

def select_candidate_captions(json_file):
    # json_file에서 image_id를 읽어옴
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        image_id = list(data.keys())[0]  # 첫 번째 image_id를 가져옴

    # copy_image_metadata.json에서 image_id를 찾음
    with open(PROCESSED_JSON, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    if image_id not in processed_data:
        return {"file": json_file, "region_id": None, "status": "error", "error": "Image ID not found in processed data"}

    # 각 region_id의 caption과 counterfactual_caption을 가져옴
    for region_id, region_data in data[image_id]["regions"].items():
        caption = region_data.get("caption", "")
        counterfactual_caption = find_error_caption(caption, region_data["captions"], region_data["counterfactual_captions"])
        
        # processed_data에 category와 counterfactual_caption 추가
        if region_id in processed_data[image_id]:
            processed_data[image_id][region_id]["counterfactual_caption"] = counterfactual_caption

    # 변경된 데이터를 다시 저장
    with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    return {"file": json_file, "region_id": None, "status": "success"}

def find_error_caption(caption, captions_list, counterfactual_list):
    for i, region_captions in enumerate(captions_list):
        for j, original_caption in enumerate(region_captions):
            if original_caption == caption:
                return counterfactual_list[i][j]
    return "No counterfactual caption found."

def main():
    openai.api__key = load_gpt_api_key(API_KEYS_PATH)
    json_files = [f for f in os.listdir(RAW_JSON_DIR) if f.endswith('.json')]
    errors = {}
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(select_candidate_captions, json_files), total=len(json_files)))

    for result in results:
        if result["status"] == "error":
            file = result["file"]
            if file not in errors:
                errors[file] = []
            errors[file].append({
                "region_id": result["region_id"],
                "response": result["response"]
            })

    if errors:
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as error_file:
            json.dump(errors, error_file, ensure_ascii=False, indent=4)
        print(f"Errors saved to {ERROR_LOG_PATH}")

        print("Success (•̀ᴗ•́)و ̑̑") # 귀욥다 ,,,, 
        

if __name__ == "__main__":
    main()