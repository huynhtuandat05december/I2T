from models.segment.segment_anything_model import SegmentAnything
from models.bilp_model import ImageCaptioning
from utils.helper import get_device

if __name__ == '__main__':
    device = get_device()
    image_caption_model = ImageCaptioning(device=device)
    model = SegmentAnything(device=device, image_caption_model=image_caption_model)
    result = model.generate_mask("examples/3.jpg")
    print(result)