def select_image_caption_prompt(captions):
    """
    @description
    주어진 입력 데이터에 기반하여 GPT 모델을 통해 caption을 select 하기 위한 prompt를 작성하는 함수
    """

    # 프롬프트 생성
    # 이미지의 오브젝트간 연관성을 기준으로 큰 범위의 바운딩박스를 생성 
    formatted_captions = ""
    for idx, caption in enumerate(captions):
        formatted_captions = f"{formatted_captions}{idx}. {caption}\n\t"
    
    prompt = (
        f"""
        Select the most accurate and specific caption that best describes the main object featured in the image. Indicate your choice by selecting the corresponding number. It is better if the characteristics or movements of the main object are described specifically.
        Even if it is hard to recognize, You need to identify the main object in a given image and select the best match from the provided options.
        If you are unable to identify or determine the main object in the image or unable to view the image, explain why you are unable to.
        
        [Caption selection rules]

        1. The selected caption must not include any details that cannot be verified from the image.
        2. The selected caption can contain grammatical errors.

        [Captions]
        {formatted_captions}

        [Response format]
        The format of the answer is as follows:
        Answer: 1
        """
    )
    
    return prompt

