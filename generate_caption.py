from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
import torch
from PIL import Image

mod=VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature=ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer=AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

app=torch.device("cuda" if torch.cuda.is_available() else "cpu")
mod=mod.to(app)

max_length=32
num_beams=4
gen_kwargs={"max_length":max_length, "num_beams":num_beams}

def generate_caption(paths):
    image=[]
    for path in paths:
        img=Image.open(path)
        if img.mode!="RGB":
            img=img.convert("RGB")  
        image.append(img)
    
    pixel_values=feature(images=image, return_tensors="pt").pixel_values
    pixel_values=pixel_values.to(app)

    output=mod.generate(pixel_values, **gen_kwargs)

    predictions=tokenizer.batch_decode(output, skip_special_tokens=True)
    predictions=[prediction.strip() for prediction in predictions]
    print("Image Caption:", predictions)
    return predictions

generate_caption(["mosaic.jpg"])