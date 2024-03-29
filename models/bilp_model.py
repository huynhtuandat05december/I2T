from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
import torch
from utils.helper import resize_long_edge

class ImageCaptioning:
    def __init__(self, device, captioner_base_model='blip2'):
        self.device = device
        self.captioner_base_model = captioner_base_model
        self.processor, self.model = self.initialize_model()

    def initialize_model(self):
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        if self.captioner_base_model == 'blip2':
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",device_map={"": 0}, torch_dtype=self.data_type
            )
        # for gpu with small memory
        elif self.captioner_base_model == 'blip':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", device_map={"": 0}, torch_dtype=self.data_type)
        else:
            raise ValueError('arch not supported')
        model.to(self.device)
        return processor, model

    def image_caption(self, image_src):
        image = Image.open(image_src)
        print(image.size)
        image = resize_long_edge(image, 384)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)
        return generated_text