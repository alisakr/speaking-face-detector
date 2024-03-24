from constants import (
    deepface_embedding_key, 
    deepface_confidence_key,
    deepface_facial_area_key,
)

class FacialImage:
    def __init__(self, deepface_result, frame):
        # 224*224*3 tensor for the detected face
        self.facial_embedding = deepface_result[deepface_embedding_key]
        self.confidence = deepface_result[deepface_confidence_key]
        self.facial_area = deepface_result[deepface_facial_area_key]
        self.frame = frame