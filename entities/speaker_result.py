class SpeakerResult:
    def __init__(self, image_file, score, score_sum):
        self.image_file = image_file
        self.score = score
        self.score_sum = score_sum
    def __lt__(self, other):
        return self.score < other.score