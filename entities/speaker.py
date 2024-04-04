from entities.facial_image import FacialImage

class Speaker:
    def __init__(self, name, facial_images: list[FacialImage]):
        self.name = name
        self._facial_images = facial_images
        # index of the best facial image in the set of images
        self.best_image_index = None
        self.find_best_image()
        self.num_matched_words = 0
        self.sum_matched_words_probability = 0

    def num_images(self):
        return len(self._facial_images)
    
    def add_image(self, image: FacialImage):
        self._facial_images.append(image)
        if self.best_image_index is None:
            self.best_image_index = len(self._facial_images) - 1
        elif image.compare(self._facial_images[self.best_image_index]) > 0:
            self.best_image_index = len(self._facial_images) - 1
    
    def get_best_image(self):
        if self.best_image_index is None:
            return None
        return self._facial_images[self.best_image_index]
    
    def find_best_image(self):
        '''
        Finds the best facial image for the speaker.
        '''
        if self.num_images() == 0:
            return None
        best_image_index = 0
        for i, image in enumerate(self._facial_images):
            if i == 0:
                continue
            if image.compare(self._facial_images[best_image_index]) > 0:
                best_image_index = i
        self.best_image_index = best_image_index
        return self._facial_images[best_image_index]
    
    def keep_best_image_only(self):
        '''
        Removes all images except the best one.
        '''
        if self.best_image_index is None:
            return None
        self._facial_images = [self._facial_images[self.best_image_index]]
        self.best_image_index = 0
        return self._facial_images[0]
    
    def get_facial_images(self):
        return self._facial_images
    
    def add_matched_word(self, probability):
        self.num_matched_words += 1
        self.sum_matched_words_probability += probability
