import copy

from controllers.video.predict_words import SpeakerWordPredictor
from controllers.video.video_predictor import VideoPredictor

class SectionPredictor():
    def __init__(self, word_predictor: SpeakerWordPredictor, chunk_size=10):
        self.word_predictor = word_predictor
        self.chunk_size = 10

    def predict(self,  video: VideoPredictor, output_folder=None, output_json=None):
        result_by_word = self.word_predictor.predict_speaker_words(video, output_folder, output_json)
        if video.transcript is None or video.transcript.json is None or "segments" not in video.transcript.json:
            raise Exception("Transcript json is not available")
        transcript_sections = video.transcript.json["segments"]
        # TODO: result should be somehow tied to the original transcript json
        result_transcript = copy.deepcopy(video.transcript.json)
        results = []
        current_index = 0
        for i, section in enumerate(transcript_sections):
            # use either self.chunk_size word chunks or the entire section if it is less than 2*chunk_size words, 
            # every chunk should be the minimum of 10-19 words
            # or the entire section
            num_read = 0
            section_length = len(section["wdlist"])
            if section_length < 2*self.chunk_size:
                num_chunks = 1
            else:
                num_chunks = section_length // self.chunk_size
            current_chunk = []
            current_chunk_number = 0
            for word in result_by_word:
                if num_read >= section_length:
                    result_by_word = result_by_word[num_read:]
                    current_index += num_read
                    break
                current_chunk.append(word)
                if len(current_chunk) >= self.chunk_size and current_chunk_number+1 < num_chunks:
                    current_chunk_number += 1
                    results.append(self.get_prediction_for_chunk(current_chunk))
                    current_chunk = []
                num_read += 1
            if len(current_chunk) > 0:
                results.append(self.get_prediction_for_chunk(current_chunk))
        return results

    def get_prediction_for_chunk(self, chunk):
        #TODO: Implement this method
        return None

                


        
