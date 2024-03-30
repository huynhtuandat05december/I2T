from models.blip_model import ImageCaptioning
from utils.helper import get_device

if __name__ == '__main__':
    device = get_device()
    model = ImageCaptioning(device=device)
    result = model.image_caption("examples/3.jpg")
    print(result)