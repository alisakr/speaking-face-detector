from functools import partial
import json
import pickle
import time

import cv2
from deepface import DeepFace
from fastai.vision.learner import load_learner
import numpy as np
import pandas as pd

from constants import (
    default_random_forest_model,
    default_speaking_model,
    face_recognition_threshold,
    features,
    max_speaker_prob_formula_field,
    max_frame_prob_formula_field,
    minimum_speaker_lip_change_frame_by_frame,
    num_frames_to_read,
    num_input_frames_to_model,
)
from controllers.transforms.transforms import GrayscaleTransform
from controllers.video.video_predictor import VideoPredictor

from controllers.image import (
    combine_images_horizontally,
    diff_image_structures,
    extract_faces_as_face_objects,
    get_image_from_facial_image_object,
    get_lips_from_image_of_face,
)
from entities.speaker import Speaker
from entities.speaker_result import SpeakerResult
from utils import (
    max_of_selected_columns,
    two_columns_equal,
)



class SpeakerWordPredictor:
    def __init__(self, image_model=None, random_forest_model=None, image_model_only=False):
        if image_model is None:
            self.image_model = load_learner(default_speaking_model)
        elif isinstance(image_model, str):
            self.image_model = load_learner(image_model)
        else:
            self.image_model = image_model
        if not image_model_only:
            if random_forest_model is None:
                with open(default_random_forest_model, 'rb') as file:
                    self.random_forest_model = pickle.load(file)
            elif isinstance(random_forest_model, str):
                with open(random_forest_model, 'rb') as file:
                    self.random_forest_model = pickle.load(file)
            else:
                self.random_forest_model = random_forest_model
        else:
            self.random_forest_model = None
        if self.random_forest_model is not None:
            self.features_to_index = {feature: i for i, feature in enumerate(features)}
            # TODO: refactor partials and change strings to constants
            # some of the fields are calculated based on the max of other fields
            max_speaker_prob_partial = partial(
                max_of_selected_columns, 
                columns=[
                    self.features_to_index['speaker_prob_start'], 
                    self.features_to_index['speaker_prob_middle'], 
                    self.features_to_index['speaker_prob_end']],
                    )
            max_frame_prob_partial = partial(
                max_of_selected_columns, 
                columns=[
                    self.features_to_index['frame_max_speaker_prob_start'], 
                    self.features_to_index['frame_max_speaker_prob_middle'], 
                    self.features_to_index['frame_max_speaker_prob_end']],
                    )
            has_max_speaker_partial = partial(
                two_columns_equal, 
                column1=self.features_to_index[max_frame_prob_formula_field], 
                column2=self.features_to_index[max_speaker_prob_formula_field],
                )
            has_max_speaker_start_partial = partial(
                two_columns_equal, 
                column1=self.features_to_index['speaker_prob_start'], 
                column2=self.features_to_index['frame_max_speaker_prob_start'],
                )
            has_max_speaker_middle_partial = partial(
                two_columns_equal, 
                column1=self.features_to_index['speaker_prob_middle'], 
                column2=self.features_to_index['frame_max_speaker_prob_middle'],
                )
            has_max_speaker_end_partial = partial(two_columns_equal, 
                column1=self.features_to_index['speaker_prob_end'], 
                column2=self.features_to_index['frame_max_speaker_prob_end'],
                )
            self.partials = [
                max_speaker_prob_partial, 
                max_frame_prob_partial, 
                has_max_speaker_partial, 
                has_max_speaker_start_partial, 
                has_max_speaker_middle_partial, 
                has_max_speaker_end_partial,
                ]

        
        
    def predict_speaker_words(self, video: VideoPredictor, output_folder=None, output_json=None):
        video.open_video()
        speaker_score_by_segment = []
        segment_results = []
        speaker_image_map = {}
        min_word_num = None
        max_word_num = None
        for i, segment in enumerate(video.transcript.segments):
            
            start_time_seconds = segment[0]
            end_time_seconds = segment[1]
            speaker_image_map[i] = []
            word = segment[3]
            segment_num = segment[4]
            word_in_segment_num = segment[5]
            if video.transcript.json is None:
                word_in_input_transcript = None
            else:
                word_in_input_transcript = video.transcript.json["segments"][segment_num]["wdlist"][word_in_segment_num]

            if min_word_num is None or segment_num < min_word_num:
                min_word_num = segment_num
            if max_word_num is None or segment_num > max_word_num:
                max_word_num = segment_num
            print(f"{i} Predicting for word {word}")
            speakers, scores, speaker_and_frame_attributes = self.get_predictions_for_word(
                video, start_time_seconds, end_time_seconds)
            if len(speakers) == 0:
                print(f"Warning: No speakers found for segment {i} word {word}")
                speaker_score_by_segment.append(None)
                continue
            if output_folder is not None:
                pd_df = pd.DataFrame(scores)
                pd_df.to_csv(f"{output_folder}/segment_{i}_{word}.csv", mode='w', header=False, index=True)
            winning_speaker_index = np.argmax(scores)
            start_formatted = format(start_time_seconds, ".1f")
            word_in_input_transcript_result = []
            score_sum = np.sum(scores)
            for j, speaker in enumerate(speakers):
                #face_outputs = [get_lips_from_image_of_face(get_image_from_facial_image_object(face_image_object)) for face_image_object in speaker.get_facial_images()]
                #face_output = combine_images_horizontally(output_filename=None, images_in_memory_copy=face_outputs)
                face_output = get_image_from_facial_image_object(speaker.get_best_image(), padding=25)
                if self.random_forest_model is not None:
                    max_speaker_prob_index = self.features_to_index[max_speaker_prob_formula_field]
                    speaker_prob = speaker_and_frame_attributes[j][max_speaker_prob_index]
                else:
                    speaker_prob = scores[j]
                if output_folder is not None:
                    if j == winning_speaker_index and speaker_prob >= 0.5:
                        speaker_file = f"{output_folder}/segment_{i}_word_{word}_{start_formatted}_{j}_winner.png"
                    elif j == winning_speaker_index:
                        speaker_file = f"{output_folder}/segment_{i}_word_{word}_{start_formatted}_{j}_best_guess.png"
                    else:
                        speaker_file = f"{output_folder}/segment_{i}_word_{word}_{start_formatted}_{j}.png"
                    cv2.imwrite(speaker_file, face_output)
                    speaker_image_map[i].append(speaker_file)
                    speaker_filename = speaker_file.split('/')[-1]
                    speaker_result = SpeakerResult(speaker_filename, speaker_prob, score_sum)
                    word_in_input_transcript_result.append(speaker_result)
            if output_folder is not None:
                pd_df = pd.DataFrame(speaker_and_frame_attributes)
                pd_df.to_csv(f"{output_folder}/segment_{i}_speaker_and_frame_attributes.csv", mode='w', header=False, index=True)
            speaker_score_by_segment.append((speakers, scores, word_in_input_transcript))
            segment_results.append((word_in_input_transcript, word_in_input_transcript_result))
              

        video.close_video()
        if output_json is not None:
            if output_folder is None:
                raise ValueError("Output folder must be provided to produce output json")
            with open(f'{output_folder}/{output_json}', 'w') as f:
                if video.transcript.json is None:
                    raise ValueError("Transcript json must be provided to produce output json")
                segment_num = 0
                for word_in_input_transcript, word_in_input_transcript_result in segment_results:
                    if word_in_input_transcript is None:
                        continue
                    word_in_input_transcript_result = sorted(word_in_input_transcript_result, reverse=True)
                    result_as_json = map(lambda obj: {
                        "speaker_image": obj.image_file, 
                        "score": obj.score, 
                        "score_sum": obj.score_sum,
                        }, 
                        word_in_input_transcript_result,
                        )
                    word_in_input_transcript["result"] = list(result_as_json)
                f.write(json.dumps(video.transcript.json))

                
        # return the speaker and scores for each word
        return speaker_score_by_segment


    def get_predictions_for_word(self, video, start_time_seconds, end_time_seconds):
        '''
        returns the speakers and their scores for a word ordered by most likely speaker
        '''
        middle_time = (start_time_seconds + end_time_seconds) / 2
        middle_time_ms = middle_time * 1000
        if self.random_forest_model is not None:
            frames = get_video_frames(video, num_frames_to_read, middle_time_ms)
        else:
            frames = get_video_frames(video, num_input_frames_to_model, middle_time_ms)
        potential_speakers = get_potential_speakers(frames)
        speaker_attributes = self.get_image_model_results(potential_speakers)
        if speaker_attributes is None or len(speaker_attributes) == 0:
            return [], [], []
        if self.random_forest_model is None:
            print("shape of speaker attributes", speaker_attributes.shape)
            print(speaker_attributes)
            for i in range(len(speaker_attributes)):
                if speaker_attributes[i][1] < minimum_speaker_lip_change_frame_by_frame:
                    speaker_attributes[i][0] = 0.0
            return potential_speakers, speaker_attributes.T[0], speaker_attributes
        max_vals_by_column = np.max(speaker_attributes, axis=0)
        min_vals_by_column = np.min(speaker_attributes, axis=0)
        mean_vals_by_column = np.mean(speaker_attributes, axis=0)
        max_tiled = np.tile(max_vals_by_column, (len(speaker_attributes), 1))
        min_tiled = np.tile(min_vals_by_column, (len(speaker_attributes), 1))
        mean_tiled = np.tile(mean_vals_by_column, (len(speaker_attributes), 1))
        
        speaker_and_frame_attributes = np.concatenate(
            (speaker_attributes, max_tiled, min_tiled, mean_tiled), axis=1)
        for partial_func in self.partials:
            partial_result = np.apply_along_axis(partial_func, axis=1, arr=speaker_and_frame_attributes)
            speaker_and_frame_attributes = np.concatenate(
                (speaker_and_frame_attributes, partial_result.reshape(len(partial_result), 1)),
                axis=1,
                )
        predictions = self.random_forest_model.predict(speaker_and_frame_attributes)
        final_speakers = []
        final_scores = []
        for i, speaker in enumerate(potential_speakers):
            final_speakers.append(speaker)
            final_scores.append(predictions[i])

        return final_speakers, final_scores, speaker_and_frame_attributes

    def get_image_model_results(self, potential_speakers):
        if len(potential_speakers) == 0:
            return []
        #TODO: potentially modularize the premodel transforms
        # TODO: make the image size configurable, perhaps move resize
        resize_partial = lambda image: cv2.resize(image, (100, 100))
        pre_model_transforms = [get_image_from_facial_image_object, get_lips_from_image_of_face, resize_partial]
        speaker_data = []

        for speaker in potential_speakers:
            speaker_attributes = self.get_image_model_results_and_attributes_for_speaker(speaker, pre_model_transforms)
            speaker_data.append(speaker_attributes)
        return np.array(speaker_data)
            
    def get_image_model_results_and_attributes_for_speaker(self, speaker, pre_model_transforms):
        image_parts = get_image_parts_for_speaker(speaker, pre_model_transforms)
        speaker_attributes = []
        lip_change_frame_by_frame = 0.0
        confidence = 0.0
        facial_area_size = 0.0
        num_faces = len(speaker.get_facial_images())
        # TODO: make start_frames, middle_frames and end_frames configurable
        start_frames = (0, 4)
        middle_frames = (2, 6)
        end_frames = (4, 8)
        if self.random_forest_model is None:
            frame_ranges = [start_frames]
            #return self.get_image_model_results_for_speaker(image_parts, [middle_frames])
        else:
            frame_ranges = [start_frames, middle_frames, end_frames]
        # lip movements for each 5 frame set, (start, middle, and end)
        lip_movement_totals_by_part = [0.0, 0.0, 0.0]
        lip_frame_by_frame_change_by_part = [0.0, 0.0, 0.0]

        for i, facial_image in enumerate(speaker.get_facial_images()):
            confidence += facial_image.confidence
            facial_area_size += facial_image.get_facial_area_size()/(facial_image.frame.shape[0]*facial_image.frame.shape[1])
            if i == 0:
                continue
            frame_change = diff_image_structures(image_parts[i-1], image_parts[i])
            lip_change_frame_by_frame += frame_change
            for j, frame_range in enumerate(frame_ranges):
                if should_include_delta(i, frame_range):
                    lip_frame_by_frame_change_by_part[j] += frame_change
                    if i == frame_range[1]:
                        lip_movement_totals_by_part[j] = diff_image_structures(image_parts[frame_range[0]], image_parts[frame_range[1]])
        facial_area_size /= num_faces
        confidence /= num_faces

        lip_movement_start_to_end = lip_movement_totals_by_part[0] + lip_movement_totals_by_part[2]
        speaker_attributes.extend(self.get_image_model_results_for_speaker(image_parts, frame_ranges))
        speaker_attributes.extend([
            lip_change_frame_by_frame,
            lip_movement_start_to_end])
        for i in range(len(frame_ranges)):
            speaker_attributes.extend([
                lip_frame_by_frame_change_by_part[i],
                lip_movement_totals_by_part[i],
                ])
        speaker_attributes.extend([confidence, facial_area_size])
        return speaker_attributes

    def get_image_model_results_for_speaker(self, image_parts, frame_ranges):
        probabilities = []
        i = 0
        for frame_range in frame_ranges:
            input_image_to_model = combine_images_horizontally(
                output_filename=None, 
                images_in_memory_copy=image_parts[frame_range[0]:frame_range[1] + 1], 
                # since we resize in pre-model transforms, second resize should be unnecessary
                resize_images=False,
                )
            # example output of model.predict: ('speaking', tensor(1), tensor([0.0787, 0.9213]))
            # another example output of model.predict: ('silent', tensor(0), tensor([0.9909, 0.0091]))
            # therefore speaking probability is always at index 1 of third item in tuple
            prediction = self.image_model.predict(input_image_to_model)
            prediction = prediction[2][1].item()
            probabilities.append(prediction)
            i += 1
        return probabilities    

                
def get_video_frames(video, num_frames, middle_time_ms):
    '''
    we want to read 9 frames with the 5th read frame in the middle
    '''
    num_frames_before_middle = num_frames // 2
    set_time_ms = middle_time_ms - (num_frames_before_middle * 1000 / video.video_fps)
    video.set_video(set_time_ms)
    frames = []
    for _ in range(num_frames):
        ret, frame = video.cap.read()
        if not ret:
            raise Exception("Error: Could not read frame from video")
        frames.append(frame)
    return frames

def get_potential_speakers(frames):
    potential_speakers = []
    for i, frame in enumerate(frames):
        faces = []
        try:
            faces = extract_faces_as_face_objects(frame, face_recognition_threshold)
        except Exception:
            #TODO: raise exception or return failure
            pass
        if faces is None or len(faces) == 0:
            # TODO: perhaps raise exception here
            return []
        if i == 0:
            for face in faces:
                # WARNING: we don't store i/which frame here because we only accept n consecutive frames
                # if that changes then we need to store the frame index, not just the face
                if face.confidence < face_recognition_threshold:
                    continue
                potential_speaker = Speaker("", [face])
                potential_speakers.append(potential_speaker)
        else:
            matched_speakers = []
            for speaker in potential_speakers:
                best_match_index = find_best_match_to_speaker(speaker, faces)
                if best_match_index is not None:
                    matched_speakers.append(speaker)
                    speaker.add_image(faces[best_match_index])
                    faces.pop(best_match_index)
            potential_speakers = matched_speakers
    return potential_speakers


def find_best_match_to_speaker(speaker, faces):
    '''
    find the best match to the speaker from the faces
    parameters:
        speaker: the speaker object
        faces: the list of facial images
    returns:
        best_match_index: index of the best match to the speaker
    '''
    best_match_index = None
    min_distance = 1.0
    for i, face in enumerate(faces):
        '''
                        example deepface.verify output
                        {
                            'verified': True,
                            'distance': 0.5235775859749816,
                            'threshold': 0.68,
                            'model': 'VGG-Face',
                            'detector_backend': 'opencv',
                            'similarity_metric': 'cosine',
                            'facial_areas': {'img1': {'x': 304,
                            'y': 182,
                            'w': 180,
                            'h': 180,
                            'left_eye': (56, 72),
                            'right_eye': (60, 70)},
                            'img2': {'x': 259,
                            'y': 203,
                            'w': 184,
                            'h': 184,
                            'left_eye': None,
                            'right_eye': None}},
                            'time': 1.26,
                        }
        '''
        speaker_area = speaker.get_best_image().get_facial_area_size()
        new_face_area = face.get_facial_area_size()
        # if the new face is too small or too large, compared to speaker, skip it
        if new_face_area < 0.25 * speaker_area or new_face_area > 4 * speaker_area:
            continue
        comparison = DeepFace.verify(
            face.facial_embedding, 
            speaker.get_best_image().facial_embedding, 
            enforce_detection=False,
            )
        if comparison['verified'] and comparison['distance'] < min_distance:
            best_match_index = i
            min_distance = comparison['distance']
    return best_match_index


def get_image_parts_for_speaker(speaker, transforms):
    image_parts = []
    for facial_image in speaker.get_facial_images():
        image_part = facial_image
        for transform in transforms:
            image_part = transform(image_part)
        image_parts.append(image_part)
    return image_parts

def should_include_delta(frame_index, frame_range):
    return frame_index > frame_range[0] and frame_index <= frame_range[1]

                