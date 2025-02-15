from process_image_data import (
    load_gpt_api_key,
    load_data,
    ask_gpt,
    process_json_file
)
import json
import os
import openai
from tqdm import tqdm

# 경로 설정
API_KEYS_PATH = "./api_key.json"
RAW_JSON_DIR = "./../data/json/raw"
CROPPED_IMAGE_DIR = "/home/cwhjpaper/data/cropped_images"
PROCESSED_JSON = "./../data/json/processed/copy_image_metadata.json"
ERROR_LOG_PATH = "./../data/json/processed/select_errors_log_caption_test.json"


'''

1. json 파일에서 image_id 찾음
2. image_metadata.json에서 image_id를 찾음
3. image_metadata.json에서 찾은 image_id에 있는 각 region_id의 caption을 가져옴
4. json 파일에서 caption을 찾아서 그와 쌍을 이루는 counterfactual_caption을 가져옴
5. caption으로 카테고리 프롬프트를 돌려서 카테고리를 찾음

'''

def select_candidate_captions(json_file):
    try:
        # json_file에서 image_id를 읽어옴
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            image_id = list(data.keys())[0]  # 첫 번째로 있는 image_id를 가져옴

        # copy_image_metadata.json 열기
        with open(PROCESSED_JSON, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)

        if image_id not in processed_data:
            return {"file": json_file, "region_id": None, "status": "error", "error": "Image ID not found in processed data"}

        # processed_data 에서 해당 image_id를 찾았는지 확인
        print(f"processed_data 에서 {image_id} 찾음")
        
        # 각 region_id에 대해 correct_caption을 가져오고, json 파일에서 counterfactual_caption을 찾음
        for region_id, region_data in processed_data[image_id].items():
            try:
                print(f"region_id: {region_id}")
                correct_caption = region_data["caption"]
                counterfactual_caption = None
                
                # json 파일의 regions에서 correct_caption과 매칭되는 counterfactual_caption 찾기
                for region in data[image_id]["regions"]:
                    for caption_data in region["captions"]:
                        if caption_data["caption"] == correct_caption:
                            counterfactual_caption = caption_data["counterfactual_caption"]
                            break
                    if counterfactual_caption:
                        break
                    
                # copy_image_metadata.json 파일에 counterfactual_caption 추가
                if counterfactual_caption:
                    processed_data[image_id][region_id]["counterfactual_caption"] = counterfactual_caption
                else:
                    processed_data[image_id][region_id]["counterfactual_caption"] = ""
                
                # 변경된 데이터를 파일에 저장
                with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=4)

            except Exception as e:
                return {"file": json_file, "region_id": region_id, "status": "error", "error": str(e)}

        return {"file": json_file, "region_id": None, "status": "success", "response": "Processing completed"}

    except Exception as e:
        return {"file": json_file, "region_id": None, "status": "error", "error": str(e)}


def main():
    try:
        openai.api__key = load_gpt_api_key(API_KEYS_PATH)
        json_files = [os.path.join(RAW_JSON_DIR, f) for f in os.listdir(RAW_JSON_DIR) if f.endswith('.json')]
        errors = {}
        
        # tqdm으로 진행상황 표시하면서 순차 처리
        results = []
        for json_file in tqdm(json_files):
            result = select_candidate_captions(json_file)
            results.append(result)

        for result in results:
            if result["status"] == "error":
                file = result["file"]
                if file not in errors:
                    errors[file] = []
                errors[file].append({
                    "region_id": result["region_id"],
                    "error": result.get("error", "Unknown error")
                })
                
        if errors:
            with open(ERROR_LOG_PATH, "w", encoding="utf-8") as error_file:
                json.dump(errors, error_file, ensure_ascii=False, indent=4)
            print(f"오류가 발생했습니다. 자세한 내용은 {ERROR_LOG_PATH}를 확인해주세요.")
        
        print("작업이 완료되었습니다 (•̀ᴗ•́)و ̑̑")

    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    main()