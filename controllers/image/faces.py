import cv2
from deepface import DeepFace
from PIL import Image
from fastai.vision.core import PILImage


from constants import deepface_confidence_key
from controllers.image.parse_image import (
    get_part_of_image,
    get_image_n_parts_vertical,
)
from entities.facial_image import FacialImage


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


def get_image_from_facial_image_object(facial_image_object, padding=0):
    return get_part_of_image(facial_image_object.facial_area, facial_image_object.frame, padding)


def get_lips_from_image_of_face(face_image):
    # assume lips to be in the bottom third of the face
    return get_image_n_parts_vertical(image_in_memory_copy=face_image, n=3)[-1]

def extract_faces_as_face_objects(frame, face_recognition_threshold=None):
    deepface_faces = extract_faces_deepface(frame)
    face_objects = []
    for face in deepface_faces:
        if face_recognition_threshold is not None and face[deepface_confidence_key] < face_recognition_threshold:
            continue
        face_objects.append(FacialImage(face, frame))
    return face_objects