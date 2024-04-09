import cv2
from PIL import Image
from fastai.vision.core import PILImage

def get_fast_ai_image(image_object=None, image_path=None):
    if image_path is not None:
        image_object = cv2.imread(image_path)
    if image_object is None:
        raise Exception("Either image_path or image_object must be provided")
    image_rgb = cv2.cvtColor(image_object, cv2.COLOR_BGR2RGB)
    # Convert the RGB image to a PIL image
    image_pil = Image.fromarray(image_rgb)
    # Convert the PIL image to a Fastai Image object
    return PILImage.create(image_pil)