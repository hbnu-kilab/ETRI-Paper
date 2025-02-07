import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "/home/cwhjpaper/preprovessing/" 
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

captions = [
   
]

text = clip.tokenize(captions).to(device)


with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

np.set_printoptions(precision=10, suppress=True)
print("Label probs:", probs)