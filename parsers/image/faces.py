import cv2
from deepface import DeepFace
from PIL import Image
from fastai.vision.core import PILImage

def extract_faces_deepface(image):
    ''' Detect faces in an image using the deepface library 

    Parameters:
        image (numpy.ndarray or image filename): The input image, if numpy.ndarray, it should be in BGR format, like cv2.imread() output
    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (np.ndarray): The detected face as a NumPy array.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values

        - "confidence" (float): The confidence score associated with the detected face.
    '''

    # TODO: figure out how to ensure cache of model deepface uses for extracting faces, it will download model
    # if not found in cache
    return DeepFace.extract_faces(image, detector_backend='mtcnn')

def get_faceai_image(image_object=None, image_path=None):
    if image_path is not None:
        image_object = cv2.imread(image_path)
    if image_object is None:
        raise Exception("Either image_path or image_object must be provided")
    image_rgb = cv2.cvtColor(image_object, cv2.COLOR_BGR2RGB)
    # Convert the RGB image to a PIL image
    image_pil = Image.fromarray(image_rgb)
    # Convert the PIL image to a Fastai Image object
    return PILImage.create(image_pil)