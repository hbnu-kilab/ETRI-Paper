import openai
import json
import base64
from typing import Optional, List, Dict, Any

class GPTHandler:
    def load_api_key(json_path: str) -> str:
        """API 키를 JSON 파일에서 로드합니다.

        Args:
            json_path (str): API 키가 저장된 JSON 파일 경로

        Returns:
            str: OpenAI API 키

        Raises:
            FileNotFoundError: JSON 파일을 찾을 수 없는 경우
            KeyError: JSON 파일에서 API 키를 찾을 수 없는 경우
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data["openai"]["api_key"]
        except (FileNotFoundError, KeyError):
            print("Error: API key file not found.")
            raise

    def __init__(self, json_path: str):
        """GPT 핸들러 초기화
        
        Args:
            json_path (str): OpenAI API 키가 저장된 JSON 파일 경로
        """
        self.api_key = self.load_api_key(json_path)
        openai.api_key = self.api_key

    def ask_gpt(self, 
                prompt: str, 
                base64_image: str, 
                max_tokens: int = 1000, 
                model: str = "gpt-4o", 
                temperature: float = 0) -> Optional[str]:
        """GPT API를 호출하여 응답을 받아옵니다.

        Args:
            prompt (str): GPT에게 전달할 프롬프트
            base64_image (str): Base64로 인코딩된 이미지
            max_tokens (int, optional): 최대 토큰 수. Defaults to 1000.
            model (str, optional): 사용할 GPT 모델. Defaults to "gpt-4o".
            temperature (float, optional): 응답의 다양성 조절. Defaults to 0.

        Returns:
            Optional[str]: GPT의 응답 또는 에러 발생 시 None
        """
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
        except openai.error.OpenAIError as e:
            print(f"OpenAI error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return None
