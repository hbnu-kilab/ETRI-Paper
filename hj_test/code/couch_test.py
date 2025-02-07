import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "/home/cwhjpaper/hj_test/images/couch.png" 
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)


captions = [
    
]

text = clip.tokenize(captions).to(device)


with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # similarity = (image_features @ text_features.T).squeeze(0)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# # 유사도를 출력
# for i, caption in enumerate(captions):
#     print(f"Caption: {caption} | Similarity: {similarity[i].item():.4f}")

np.set_printoptions(precision=10, suppress=True)
print("Label probs:", probs) 