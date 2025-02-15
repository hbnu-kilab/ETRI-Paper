import json
import base64
import os
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from filelock import FileLock
import openai

from generate_image_categories import generate_categories_prompt



# 경로 설정
API_KEYS_PATH = "./api_key.json"
RAW_JSON_DIR = "./../data/json/raw"
CROPPED_IMAGE_DIR = "./../data/cropped_images"
PROCESSED_JSON = "./../data/json/processed/copy_image_metadata.json"
ERROR_LOG_PATH = "./../data/json/processed/select_errors_log_caption_test.json"


'''

1. json 파일에서 image_id 가져옴
2. image_metadata.json에서 해당 image_id를 찾음
3. image_metadata.json에서 찾은 image_id에 있는 각 region_id의 caption을 가져와서 카테고리 프롬프트를 돌림
4. json 파일에서 caption을 찾아서 그와 쌍을 이루는 counterfactual_caption을 가져옴
5. caption으로 카테고리 프롬프트를 돌려서 카테고리를 찾음

'''

def load_gpt_api_key(API_KEYS_PATH):
    try:
        with open(API_KEYS_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data["openai"]["api_key"]
    except (FileNotFoundError, KeyError):
        print("Error: API key file not found.")
        exit(1)


def load_data(json_file):
    try:
        file_path = os.path.join(RAW_JSON_DIR, json_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        vgid = list(data.keys())[0]
        regions = data[vgid]['regions']
        region_ids = [region["id"] for region in regions]
        captions = [[caption["caption"] for caption in region["captions"]] for region in regions]
        counterfactual_captions = [[counterfactual_caption["counterfactual_caption"] for counterfactual_caption in region["captions"]] for region in regions]
        
        base64_images = []
        for region_id in region_ids:
            try:
                image_path = os.path.join(CROPPED_IMAGE_DIR, f"{vgid}/{region_id}.jpg")
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                base64_images.append(base64_image)
            except FileNotFoundError:
                base64_images.append("")
    except:
        print(f"Error loading JSON data")
        return None, None, None, None, None 

    return vgid, region_ids, captions, counterfactual_captions, base64_images


# GPT API 호출
def ask_gpt(prompt, base64_image, max_tokens, model="gpt-4o", temperature=0):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", 
                 "content": """You are a concise and accurate AI assistant.
                    Do not include phrases like "Sure," "Certainly," "Of course," "Absolutely," "Let me provide that," 
                    or "Here's the information" in your responses."""},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }}
                ]}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"OpenAI API call error: {e}")
        return None


def select_candidate_captions_category(json_file):
    try:
        # json_file에서 image_id를 읽어옴
        vgid, region_ids, captions, counterfactual_captions, base64_images = load_data(json_file)
        if vgid is None:
            return {"file": json_file, "region_id": None, "status": "error", "error": "Failed to load data"}

        # copy_image_metadata.json 열기
        with open(PROCESSED_JSON, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)

        if vgid not in processed_data:
            return {"file": json_file, "region_id": None, "status": "error", "error": "Image ID not found in processed data"}

        # processed_data 에서 해당 image_id를 찾았는지 확인
        print(f"processed_data 에서 {vgid} 찾음")
        
        # 각 region_id에 대해 correct_caption을 가져오고, json 파일에서 counterfactual_caption을 찾음
        for idx, region_id in enumerate(region_ids):
                correct_caption = processed_data[vgid][region_id]["caption"]
                
                generate_category_prompt = generate_categories_prompt(correct_caption)
                response = ask_gpt(generate_category_prompt, base64_images[idx], max_tokens=1000)
                
                if re.search(r'[^:]*$', response):
                    cleaned_response = re.sub(r'\D', '', re.search(r'[^:]*$', response).group(0).strip())
                else:
                    cleaned_response = re.sub(r'\D', '', response.strip())
                
                if cleaned_response:
                    processed_data[vgid][region_id]["category"] = cleaned_response
                else:
                    processed_data[vgid][region_id]["category"] = ""
                            
                    # counterfactual_caption 찾기
                    counterfactual_caption = None
                    
                    # json 파일의 regions에서 correct_caption과 매칭되는 counterfactual_caption 찾기
                    for idx, region_data in enumerate(captions):
                        for caption_data in region_data:
                            if caption_data == correct_caption:
                                counterfactual_caption = region_data[idx]
                                break
                        if counterfactual_caption:
                            break
                    
                    # copy_image_metadata.json 파일에 counterfactual_caption 추가
                    if counterfactual_caption:
                        processed_data[vgid][region_id]["counterfactual_caption"] = counterfactual_caption
                    else:
                        processed_data[vgid][region_id]["counterfactual_caption"] = ""
                    
                    # 변경된 데이터를 파일에 저장
                    with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, ensure_ascii=False, indent=4)

        return {"file": json_file, "region_id": None, "status": "success", "response": "Processing completed"}

    except Exception as e:
        return {"file": json_file, "region_id": None, "status": "error", "error": str(e)}


def main():
    openai.api_key = load_gpt_api_key(API_KEYS_PATH)
    json_files = [f for f in os.listdir(RAW_JSON_DIR) if f.endswith('.json')]

    errors = {}

    # 순차적으로 파일 처리
    for json_file in tqdm(json_files):
        result = select_candidate_captions_category(json_file)
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

if __name__ == "__main__":
    main()