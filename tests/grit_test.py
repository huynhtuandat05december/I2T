from models.grit_model import DenseCaptioning
from utils.helper import get_device

if __name__ == '__main__':
    device = get_device()
    model = DenseCaptioning(device=device)
    result = model.image_dense_caption("examples/3.jpg")
    print(result)