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
    def get_facial_area_size(self):
        return self.facial_area['w'] * self.facial_area['h']
    def compare(self, facial_image):
        if self.confidence > facial_image.confidence:
            return 1.0
        if self.confidence < facial_image.confidence:
            return -1.0
        if self.num_eyes() > facial_image.num_eyes():
            return 1.0
        if self.num_eyes() < facial_image.num_eyes():
            return -1.0
        if self.get_facial_area_size() > facial_image.get_facial_area_size():
            return 1.0
        if self.get_facial_area_size() < facial_image.get_facial_area_size():
            return -1.0
        return 0.0
    def num_eyes(self):
        num_eyes = 0
        if 'left_eye' in self.facial_area and self.facial_area['left_eye'] is not None:
            num_eyes += 1
        if 'right_eye' in self.facial_area and self.facial_area['right_eye'] is not None:
            num_eyes += 1
        return num_eyes