from models.segment.semantic_segment_anything_model import SemanticSegementAnything
from models.region_semantic import RegionSemantic
from models.bilp_model import ImageCaptioning
from utils.helper import get_device

if __name__ == '__main__':
    device = get_device()
    image_caption_model = ImageCaptioning(device=device)
    model = RegionSemantic(device=device,image_caption_model=image_caption_model)
    result = model.region_semantic("examples/3.jpg")
    print(result)