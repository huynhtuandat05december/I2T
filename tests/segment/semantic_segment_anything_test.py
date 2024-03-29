from models.segment.semantic_segment_anything_model import SemanticSegementAnything
from models.bilp_model import ImageCaptioning
from utils.helper import get_device

if __name__ == '__main__':
    device = get_device()
    image_caption_model = ImageCaptioning(device=device)
    model = SemanticSegementAnything( image_caption_model=image_caption_model)
    result = model.generate_mask("examples/3.jpg")
    print(result)