from fastai.vision.all import PILImage, Transform

# Define a grayscale transformation
class GrayscaleTransform(Transform):
    def __init__(self):
        pass
    
    def encodes(self, x: PILImage):
        return x.convert('L')