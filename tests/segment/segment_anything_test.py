from models.segment.segment_anything_model import SegmentAnything
from utils.helper import get_device

if __name__ == '__main__':
    device = get_device()
    model = SegmentAnything(device=device)
    result = model.generate_mask("examples/3.jpg")
    print(result)