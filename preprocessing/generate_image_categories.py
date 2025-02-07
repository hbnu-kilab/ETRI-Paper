import openai


def generate_categories_prompt(caption, file_num, model="gpt-4o", temperature=0):
    """
    @description
    주어진 입력 데이터에 기반하여 GPT 모델을 통해 caption을 select 하기 위한 prompt를 작성하는 함수
    """

    # 프롬프트 생성
    # 이미지의 오브젝트간 연관성을 기준으로 큰 범위의 바운딩박스를 생성 
    
    prompt = (
        f"""
        This is a most accurate and specific caption that describes a image. Select a most appropriate category. Indicate your choice by selecting the corresponding number.
        Even if it is hard to select, you need to select the best match category from the given categories then, explain why you cannot select a category.


        [Category selection rules]
        1. You must select a category for the main object of the image.
        2. The given caption can contain grammatical errors.


        [Caption] 
        {caption}
        

        [Categories]
        1. person
        2. object
        3. artifact
        4. location
        5. substance
        6. group
        7. plant
        8. animal
        9. body
        10. phenomenon 
        11. food
        12. time
        13. event
        14. shape


        [Answer format]
        The format of the answer is as follows:
        Answer: 1
        """
    )
    
    return prompt

