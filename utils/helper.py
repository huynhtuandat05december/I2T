import cv2
import torch

def resize_long_edge_cv2(image, target_size=384):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    else:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def resize_long_edge(image, target_size=384):
    width, height = image.size
    aspect_ratio = float(width) / float(height)

    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
