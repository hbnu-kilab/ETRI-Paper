import json
import base64
import os
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from filelock import FileLock
import openai

from select_image_caption import select_image_caption_prompt

# 경로 설정
API_KEYS_PATH = "api_keys.json"
RAW_JSON_DIR = "/home/cwhjpaper/data/json/raw"
CROPPED_IMAGE_DIR = "/home/cwhjpaper/data/cropped_images"
PROCESSED_JSON = "/home/cwhjpaper/data/json/processed/image_metadata.json"
ERROR_LOG_PATH = "/home/cwhjpaper/data/json/processed/select_errors_log.json"

# API 키 로드
def load_gpt_api_key(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data["openai"]["api_key"]
    except (FileNotFoundError, KeyError):
        print("Error: API key file not found.")
        exit(1)

# JSON 데이터 로드
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
                    or "Here’s the information" in your responses."""},
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

# JSON 파일 처리
def process_json_file(json_file):
    vgid, region_ids, captions, counterfactual_captions, base64_images = load_data(json_file)
    if vgid is None:
        return {"file": json_file, "region_id": None, "status": "error", "error": "Failed to load data"}

    try:
        lock = FileLock(PROCESSED_JSON + ".lock")
        with lock:
            if not os.path.exists(PROCESSED_JSON):
                save_data = {}
            else:
                with open(PROCESSED_JSON, 'r', encoding='utf-8') as file:
                    save_data = json.load(file)

            if vgid not in save_data:
                save_data[vgid] = {}

            for idx, region_id in enumerate(region_ids):
                if region_id not in save_data[vgid]:
                    save_data[vgid][region_id] = {}
                    if len(captions[idx]) > 1:
                        select_caption_prompt = select_image_caption_prompt(captions[idx])
                        response = ask_gpt(select_caption_prompt, base64_images[idx], max_tokens=1000)
                        if response is None:
                            return {"file": json_file, "region_id": region_id, "status": "error", "error": "GPT response is None"}

                        if re.search(r'[^:]*$', response):
                            cleaned_response = re.sub(r'\D', '', re.search(r'[^:]*$', response).group(0).strip())
                        else:
                            cleaned_response = re.sub(r'\D', '', response.strip())
                    else:
                        print("has only 1 caption", json_file, region_id, captions[idx])
                        cleaned_response = "0"

                    try:
                         save_data[vgid][region_id]['category'] = ""
                    except Exception:
                        return {"file": json_file, 
                                "region_id": region_id, 
                                "status": "error", 
                                "response": response}

            # 저장하기 전에 모든 작업 완료 확인
            with open(PROCESSED_JSON, 'w', encoding='utf-8') as file:
                json.dump(save_data, file, ensure_ascii=False, indent='\t')

    except Exception:
        return {"file": json_file, 
                "region_id": None, 
                "status": "error", 
                "response": None}

    return {"file": json_file, 
            "region_id": None, 
            "status": "success", 
            "error": None}

# 메인 함수
def main():
    openai.api_key = load_gpt_api_key(API_KEYS_PATH)
    json_files = [f for f in os.listdir(RAW_JSON_DIR) if f.endswith('.json')]

    errors = {}

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_json_file, json_files), total=len(json_files)))

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
        
# 코드 잘 보고 갑니다 ^__^ * 

if __name__ == "__main__":
    main()
