import os
import time

import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd


from controllers.image import (
    combine_images_horizontally,
    diff_image_structures, 
    extract_faces_deepface,
    get_faceai_image,
    get_image_from_facial_image_object,
    get_image_n_parts_vertical,
    get_lips_from_image_of_face,
    get_part_of_image,
    save_image_into_n_parts_horizontal,
)
from constants import (
    deepface_embedding_key,
    deepface_confidence_key,
    face_recognition_threshold,
    speaker_probability_columns,
)
from entities.facial_image import FacialImage
from entities.speaker import Speaker

def parse_video_for_speakers(video_path, transcript_segments, output_folder, ordered_speakers=None):
    cap = cv2.VideoCapture(video_path)
    ordered_speakers = [speaker.replace(' ', '_').lower() for speaker in ordered_speakers] if ordered_speakers else None
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    num_frames = 0
    # Get the total number of frames in the video
    num_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total number of frames in the video: {num_frames_input}")
    print(f"Number of frames per second: {fps}")
    print(f"Total number of transcript segments in the video: {len(transcript_segments)}")
    num_speaking_frames = 0
    for segment_info in transcript_segments:
        start_time_ms = segment_info[0] * 1000
        end_time_ms = segment_info[1] * 1000
        speaker = segment_info[2]
        # Set the video capture to the middle of the segment
        current_time_ms = (start_time_ms + end_time_ms) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
        _, frame = cap.read()
        if frame is None:
            continue      
        # If ordered_speakers is provided, split the frame into n parts and save each part as a separate image
        if ordered_speakers:
            split_filenames = []
            for possible_speaker in ordered_speakers:
                if possible_speaker == speaker:
                    split_filenames.append(f"{output_folder}/speaking/{possible_speaker}_{current_time_ms}.png")
                else:
                    split_filenames.append(f"{output_folder}/silent/{possible_speaker}_{current_time_ms}.png")    
            save_image_into_n_parts_horizontal(None, len(ordered_speakers), output_filenames=split_filenames, image_in_memory_copy=frame)
        else:
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # Save the frame to the output folder
            output_file = os.path.join(output_folder, f"{speaker}_{current_time_ms}.png")
            cv2.imwrite(output_file, frame)
        if speaker is not None:
            num_speaking_frames += 1
        num_frames += 1
    print(f"Total number of frames written to the output folder: {num_frames}")
    print(f"Total number of speaking frames: {num_speaking_frames}")
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

def get_times_and_speakers_all_frames(video_path, transcript_segments, output_folder, ordered_speakers=None):
    cap = cv2.VideoCapture(video_path)
    ordered_speakers = [speaker.replace(' ', '_').lower() for speaker in ordered_speakers] if ordered_speakers else None
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    num_frames = 0
    # Get the total number of frames in the video
    num_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total number of frames in the video: {num_frames_input}")
    print(f"Number of frames per second: {fps}")
    print(f"Total number of transcript segments in the video: {len(transcript_segments)}")
    num_speaking_frames = 0
    for segment_info in transcript_segments:
        start_time_ms = segment_info[0] * 1000
        end_time_ms = segment_info[1] * 1000
        speaker = segment_info[2]
        # Set the video capture to the middle of the segment
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)      
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        while current_time_ms < end_time_ms:
            # Read the next frame from the video
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            _, frame = cap.read()
            if frame is None:
                continue      
            # If ordered_speakers is provided, split the frame into n parts and save each part as a separate image
            if ordered_speakers:
                split_filenames = []
                for possible_speaker in ordered_speakers:
                    if possible_speaker == speaker:
                        split_filenames.append(f"{output_folder}/speaking/{possible_speaker}_{current_time_ms}.png")
                    else:
                        split_filenames.append(f"{output_folder}/silent/{possible_speaker}_{current_time_ms}.png")    
                save_image_into_n_parts_horizontal(None, len(ordered_speakers), output_filenames=split_filenames, image_in_memory_copy=frame)
            else:
                current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Save the frame to the output folder
                output_file = os.path.join(output_folder, f"{speaker}_{current_time_ms}.png")
                cv2.imwrite(output_file, frame)
            if speaker is not None:
                num_speaking_frames += 1
            num_frames += 1
    print(f"Total number of frames written to the output folder: {num_frames}")
    print(f"Total number of speaking frames: {num_speaking_frames}")
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

def parse_video_for_speakers_middle_n_frames(video_path, transcript_segments, output_folder, speaker_image_files=None, n=5, use_lips_only=True):
    '''
    plan is simple - get the faces in the middle n frames of each segment, concatenate them horizontally and save them
    
    '''

    cap = cv2.VideoCapture(video_path)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    # speaker name to speaker embedding map
    speaker_embeddings = get_speaker_embeddings(speaker_image_files)
    # Get the total number of frames in the video
    num_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame_ms = 1000 / fps
    print(f"Total number of frames in the video: {num_frames_input}")
    print(f"Number of frames per second: {fps}")
    num_printed = 0
    # using fps and n get the middle time frames
    for segment_info in transcript_segments:
        start_time_ms = segment_info[0] * 1000
        end_time_ms = segment_info[1] * 1000
        if segment_info[2] is not None:
            current_speaker = segment_info[2].replace(' ', '_').lower()
        else:
            current_speaker = 'unknown'
        # Set the video capture to get n/2 segments prior to the middle, the middle segment and (n/2)-1 after the middle
        num_pre_middle_frames = n // 2
        current_time_ms = (start_time_ms + end_time_ms) / 2
        current_time_ms -= num_pre_middle_frames * time_per_frame_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
        faces_by_person = []
        frames = []
        for i in range(n):
            # Read the next frame from the video
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            _, frame = cap.read()
            if frame is None:
                continue
            frames.append(frame)
            faces = None
            try:
                faces = extract_faces_deepface(frame)
            except Exception as e:
                print(f"Error: {e}")
                continue
            faces_matched_to = set()
            for face in faces:
                if face[deepface_confidence_key] < face_recognition_threshold:
                    continue
                if len(faces_by_person) == 0:
                    faces_by_person.append([(i, face)])
                    faces_matched_to.add(0)
                else:
                    matched = False
                    for j, faces_for_person in enumerate(faces_by_person):
                        if j in faces_matched_to:
                            # assume we have already found this person in the frame
                            continue
                        if num_printed < 1:
                            print(faces_for_person[0])
                            num_printed += 1
                        if DeepFace.verify(face[deepface_embedding_key], faces_for_person[0][1][deepface_embedding_key], 
                                           enforce_detection=False)["verified"]:
                            faces_by_person[j].append((i, face))
                            matched = True
                            faces_matched_to.add(j)
                            break
                    if not matched:
                        faces_by_person.append([(i, face)])
                        faces_matched_to.add(len(faces_by_person) - 1)
        # now we have the faces for each person in the middle n frames
        # we now want ...
        # 1. determine which speaker based on speaker embeddings
        # 2. concatenate the frames horizontally of each set of faces
        # 3. determine output filename for each of the concatenated frames
        # 4. save them, based on the transcript speaker in silent or speaking directory
        # save them, based on the speaker name in silent or speaking directory
                        
        # TODO: make this loop mapping function or list comprehension
        def get_facial_area_adjusted(facial_area, frame, padding=0):
            face_part_of_frame = get_part_of_image(facial_area, frame, padding)
            if use_lips_only:
                # assume lips to be in the bottom third of the face
                return get_image_n_parts_vertical(image_in_memory_copy=face_part_of_frame, n=3)[-1]
            return face_part_of_frame
        
        for face_finds in faces_by_person:
            if len(face_finds) < n:
                continue
            output_dir = ''
            matching_speaker = None
            for frame_face in face_finds:
                face = frame_face[1]
                
                for speaker, speaker_embedding in speaker_embeddings.items():
                    compare_to_speaker = DeepFace.verify(
                        face[deepface_embedding_key], speaker_embedding, enforce_detection=False)   
                    if compare_to_speaker['verified']:
                        matching_speaker = speaker
                        break
                if matching_speaker == current_speaker and matching_speaker is not None:
                    output_dir = 'speaking'
                    break
                elif matching_speaker is not None:
                    output_dir = 'silent'
                    break
            if matching_speaker is None:
                print(f"Could not find a matching speaker for one of the faces in the middle {n} frames")
                continue
            # now we concatenate the frames horizontally, we should test in kaggle how best to do that
            # do we need to resize the frames to be the same size?
            image_parts = [get_facial_area_adjusted(frame_face[1]['facial_area'], frames[frame_face[0]]) for frame_face in face_finds]
            output_filename = f"{output_folder}/{output_dir}/{matching_speaker}_{current_time_ms}.png"
            combine_images_horizontally(output_filename=output_filename, images_in_memory_copy=image_parts)


            
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

def parse_video_for_classifying_speakers(
        video_path, 
        transcript_segments, 
        model, 
        target, 
        output_folder, 
        n=5, 
        num_segments_to_try=None,
        debug_mode=False,
        debug_mode_output_folder=None,
        use_lips_only=True,
        target_probability_threshold=0.5,
        ):
    '''
    testing out how to find the speaker
    parameters:
        video_path (str): path to the video file
        transcript_segments (list of tuples): each tuple contains the start time, end time and speaker name
        model: the model or list of models to use for filtering to the desired faces, each model 
        target: the target label or list of targets to use for filtering to the desired faces, each target is a string or list of strings
        output_folder (str): the output folder to save the frames

    '''
    print("segment processing begins")
    cap = cv2.VideoCapture(video_path)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    unique_speakers = set([segment[2] for segment in transcript_segments])
    print(f"Unique speakers: {len(unique_speakers)}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame_ms = 1000 / fps
    processed_segments = 0
    speaker_faces = {}
    speaker_num_matched_words = {}
    speaker_num_unmatched_words = {}
    #TODO: remove or set configurable cap on num words per speaker
    per_speaker_cap = 100
    for segment_info in transcript_segments:
        # Step 1: get speaker, time.
        start_time_ms = segment_info[0] * 1000
        end_time_ms = segment_info[1] * 1000
        current_speaker = segment_info[2]
        if current_speaker not in speaker_num_matched_words:
            speaker_num_matched_words[current_speaker] = 0
        if current_speaker not in speaker_num_unmatched_words:
            speaker_num_unmatched_words[current_speaker] = 0
        if speaker_num_matched_words[current_speaker] > per_speaker_cap:
            continue
        # Set the video capture to get n/2 segments prior to the middle, the middle segment and (n/2)-1 after the middle
        # since we skip frames, should be 2*n - 1 // 2
        num_pre_middle_frames = (2 * n - 1) // 2
        current_time_ms = (start_time_ms + end_time_ms) / 2
        current_time_ms -= num_pre_middle_frames * time_per_frame_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
        # example output of model.predict: ('speaking', tensor(1), tensor([0.0787, 0.9213]))
        # another example output of model.predict: ('silent', tensor(0), tensor([0.9909, 0.0091]))
        # therefore speaking probability is alwasy at index 1 / second item
        target_index = 1
        speakers_by_probability = get_speakers_by_probability(
            cap, 
            model, 
            target_index, 
            use_lips_only=use_lips_only,
        )
        if len(speakers_by_probability) == 0 or speakers_by_probability[0][1][1] < target_probability_threshold:
            speaker_num_unmatched_words[current_speaker] += 1
            if debug_mode:
                print(f"Could not find a speaker for segment of {current_speaker} at time {current_time_ms}")
            continue
        speaker = speakers_by_probability[0][0]
        speaker_prob = speakers_by_probability[0][1][1]
        speaker_num_matched_words[current_speaker] += 1
        speaker.name = current_speaker
        if debug_mode:
            c = 0
            for any_speaker, any_speaker_prob in speakers_by_probability:
                speaker_prob_middle = any_speaker_prob[1]

                print(f"Speaker: {any_speaker.name}, Probability: {speaker_prob_middle}")
                image_parts = [get_image_from_facial_image_object(facial_image) for facial_image in any_speaker.get_facial_images()]
                raw_image = combine_images_horizontally(output_filename=None, images_in_memory_copy=image_parts)
                raw_image = get_image_n_parts_vertical(image_in_memory_copy=raw_image, n=3)[-1]
                label = target
                filename = f"{output_folder}/{current_speaker}_{label}_{current_time_ms}_{c}_{any_speaker_prob[0]}_{any_speaker_prob[1]}_{any_speaker_prob[2]}_{any_speaker_prob[3]}"
                filename += f"_{any_speaker_prob[4]}_{any_speaker_prob[5]}_{any_speaker_prob[6]}_raw.png"
                print(f"Saving image to {filename}")
                cv2.imwrite(filename, raw_image)
                c += 1
            print(f"current_time_ms: {current_time_ms}")
            print(f"speaker: {speaker}")
            cv2.imwrite(f"{debug_mode_output_folder}/{current_speaker}_{current_time_ms}_frame.png", speaker.get_best_image().frame)
       
        
        if current_speaker not in speaker_faces:
            speaker.add_matched_word(speaker_prob)
            speaker_faces[current_speaker] = [speaker]
            speaker.keep_best_image_only()
        else:
            best_match_index = find_best_match_to_speaker(
                speaker, [existing_speaker.get_best_image() for existing_speaker in speaker_faces[current_speaker]])
            if best_match_index is not None:
                speaker_faces[current_speaker][best_match_index].add_matched_word(speaker_prob)
                speaker_faces[current_speaker][best_match_index].add_image(speaker.get_best_image())
                speaker_faces[current_speaker][best_match_index].keep_best_image_only()
            else:
                speaker.add_matched_word(speaker_prob)
                speaker_faces[current_speaker].append(speaker)
                speaker.keep_best_image_only()
        
        processed_segments += 1
        if num_segments_to_try is not None and processed_segments >= num_segments_to_try:
            break

    # now we have the speaker faces, we want to determine most likely speaker and save the best image for each speaker
    for speaker_name, speakers in speaker_faces.items():
        max_matched_words = 0
        best_speaker_on_word_count = None
        max_probability = 0
        best_speaker_on_probability = None
        for speaker in speakers:
            if speaker.num_matched_words >= max_matched_words:
                max_matched_words = speaker.num_matched_words
                best_speaker_on_word_count = speaker
            if speaker.sum_matched_words_probability >= max_probability:
                max_probability = speaker.sum_matched_words_probability
                best_speaker_on_probability = speaker  
        if best_speaker_on_word_count is not None and best_speaker_on_probability is not None:
            if best_speaker_on_word_count != best_speaker_on_probability:
                print(f"Warning: Speaker {speaker_name} has different best speakers for word count and probability")
                continue
            output_filename = f"{output_folder}/word_count_{speaker_name}_{max_matched_words}_{max_probability}.png"
            face_output = get_image_from_facial_image_object(best_speaker_on_word_count.get_best_image(), padding=25)
            cv2.imwrite(output_filename, face_output)
              
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

def parse_video_for_matrix_data(
        video_path, 
        transcript_segments, 
        model, 
        speaker_image_files,
        result_output_csv,
        n=5, 
        use_lips_only=True,
        only_use_words_with_known_speakers=True,
        output_image_folder=None,
        ):
    '''
    testing out how to find the speaker
    parameters:
        video_path (str): path to the video file
        transcript_segments (list of tuples): each tuple contains the start time, end time and speaker name
        model: the model or list of models to use for filtering to the desired faces, each model 
        target: the target label or list of targets to use for filtering to the desired faces, each target is a string or list of strings
        output_folder (str): the output folder to save the frames

    '''
    print("segment processing begins")
    cap = cv2.VideoCapture(video_path)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    unique_speakers = set([segment[2] for segment in transcript_segments])
    print(f"Unique speakers: {len(unique_speakers)}")
    start_video_time = round(time.time() * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame_ms = 1000 / fps
    # speaker name to speaker embedding map
    speaker_embeddings = get_speaker_embeddings(speaker_image_files)
    result_matrix = []
    chunk_size = 20
    num_written_to_csv = 0
    num_words_missed = 0
    num_words_processed = 0
    columns = speaker_probability_columns + ['speaker_match']

    for i in range(len(speaker_probability_columns)):
        columns.append("frame_max_" + columns[i])
    for i in range(len(speaker_probability_columns)):
        columns.append("frame_min_" + columns[i])
    for i in range(len(speaker_probability_columns)):
        columns.append("frame_mean_" + columns[i])
    
    for segment_info in transcript_segments:
        # Step 1: get speaker, time.
        start_time_ms = segment_info[0] * 1000
        end_time_ms = segment_info[1] * 1000
        current_speaker = segment_info[2]
        if only_use_words_with_known_speakers and current_speaker is None:
            continue
        # Set the video capture to get n/2 segments prior to the middle, the middle segment and (n/2)-1 after the middle
        # since we skip frames, should be 2*n - 1 // 2
        num_pre_middle_frames = (2 * n - 1) // 2
        current_time_ms = (start_time_ms + end_time_ms) / 2
        current_time_ms -= num_pre_middle_frames * time_per_frame_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
        # example output of model.predict: ('speaking', tensor(1), tensor([0.0787, 0.9213]))
        # another example output of model.predict: ('silent', tensor(0), tensor([0.9909, 0.0091]))
        # therefore speaking probability is alwasy at index 1 / second item
        target_index = 1
        speakers_by_probability = get_speakers_by_probability(
            cap, 
            model, 
            target_index, 
            use_lips_only=use_lips_only,
        )
        found_speaker = False
        result_temp = []
        skip_term = False
        if speakers_by_probability is None:
            print(f"Warning: NO faces for the word {segment_info[3]}_{segment_info[2]}_{segment_info[0]}_{segment_info[1]}")
            num_words_missed += 1
            continue
        for speaker_for_word, speaker_attributes in speakers_by_probability:
            # check if speaker is the speaker for the word
            # if so, append 1, else append 0
            # then append speaker_attributes to the matrix
            speaker_for_word_image = speaker_for_word.get_best_image()
            expected_speaker_embedding = speaker_embeddings[current_speaker]
            compare_to_speaker = DeepFace.verify(
                        speaker_for_word_image.facial_embedding, 
                        expected_speaker_embedding, 
                        enforce_detection=False,
                        )
            if compare_to_speaker['verified']:
                speaker_match = 1
                if found_speaker:
                    print(f"Warning: Found multiple speakers for the same word {segment_info[3]}_{segment_info[2]}_{segment_info[0]}_{segment_info[1]}")
                    skip_term = True
                    break
                found_speaker = True
            else:
                speaker_match = 0
            result_temp.append(speaker_attributes + [speaker_match])
            if output_image_folder is not None:
                filename = f"{output_image_folder}/{start_video_time}_{len(result_matrix)+len(result_temp)-1}.png"
                cv2.imwrite(filename, get_image_from_facial_image_object(speaker_for_word_image, padding=25))
        if not found_speaker:
            print(f"Warning: Could not find a speaker for the word {segment_info[3]}_{segment_info[2]}_{segment_info[0]}_{segment_info[1]}")
            num_words_missed += 1
            continue
        if skip_term:
            num_words_missed += 1
            continue
        numpy_temp = np.array(result_temp)
        max_vals_by_column = np.max(numpy_temp, axis=0)[:len(speaker_probability_columns)]
        max_tiled = np.tile(max_vals_by_column, (numpy_temp.shape[0], 1))
        min_vals_by_column = np.min(numpy_temp, axis=0)[:len(speaker_probability_columns)]
        min_tiled = np.tile(min_vals_by_column, (numpy_temp.shape[0], 1))
        mean_vals_by_column = np.mean(numpy_temp, axis=0)[:len(speaker_probability_columns)]
        mean_tiled = np.tile(mean_vals_by_column, (numpy_temp.shape[0], 1))
        numpy_temp = np.concatenate((numpy_temp, max_tiled, min_tiled, mean_tiled), axis=1)
                
        num_words_processed += 1
        if len(result_matrix) == 0:
            result_matrix = numpy_temp
        else:
            result_matrix = np.concatenate((result_matrix, numpy_temp), axis=0)
        if result_output_csv is not None and len(result_matrix)-num_written_to_csv >= chunk_size:
            print(f"Num written to csv: {num_written_to_csv}")
            df = pd.DataFrame(result_matrix[num_written_to_csv:], columns=columns)
            if num_written_to_csv == 0:
                header = True
            else:
                header = False
            df.to_csv("temp_" + result_output_csv, mode='a', header=header, index=False)
            num_written_to_csv = len(result_matrix)

    if result_output_csv is not None and len(result_matrix) > num_written_to_csv:
        df = pd.DataFrame(result_matrix[num_written_to_csv:], columns=columns)
        if num_written_to_csv == 0:
            header = True
        else:
            header = False
        df.to_csv("temp_" + result_output_csv, mode='a', header=header, index=False)

    print(f"Number of words processed: {num_words_processed}")
    print(f"Number of words missed: {num_words_missed}")
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    df = pd.DataFrame(result_matrix, columns=columns)
    df.to_csv(result_output_csv, mode='w', header=True, index=False)
    return result_matrix
       
    


def get_speakers_by_probability(cap, model, target_index, use_lips_only=True):
    '''
    read num_frames frames and returns potenial speakers ordered by descending speaker probability
    parameters:
        cap: the video capture object, set to the first frame to read
        model: the model to use for speaker classification
        target_index: the index in the predictiosns tensor of the target label to maximize for classification
        num_frames: the number of frames to save
        use_lips_only: whether to use only the lips for the speaker classification (default: True)
        skip_frames: the number of frames to skip in between each read frame (default: 0)
    returns:
        structured numpy 2d array of potential speaker and all attributes for predicting their liklelihood of being the speaker.
    '''
    potential_speakers = []
    frames = []
    num_frames = 9
    num_input_frames_to_model = 5
    for i in range(num_frames):
        # Read the next frame from the video
        _, frame = cap.read()
        if frame is None:
            return None

        
        frames.append(frame)
        faces = None
        try:
            faces = extract_faces_deepface(frame)
            face_objects = []
            for face in faces:
                if face[deepface_confidence_key] < face_recognition_threshold:
                    continue
                face_objects.append(FacialImage(face, frame))
            faces = face_objects
        except Exception as e:
            print(f"Error: {e}")
            continue
        if len(faces) == 0:
            return None
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
    
    # pre_model_transforms takes facial_image_object as input and then each function takes as input the result of the previous function
    # last function should return image
    pre_model_transforms = [get_image_from_facial_image_object]
    if use_lips_only:
        pre_model_transforms.append(get_lips_from_image_of_face)
    speakers_ordered_by_prob = []
    
    for speaker in potential_speakers:
        image_parts = []
        speaker_attributes = []
        lip_change_frame_by_frame = 0.0
        confidence = 0.0
        facial_area_size = 0.0
        num_faces = len(speaker.get_facial_images())
        start_frames = (0, 4)
        middle_frames = (2, 6)
        end_frames = (4, 8)
        frame_ranges = [start_frames, middle_frames, end_frames]
        # lip movements for each 5 frame set, (start, middle, and end)
        lip_movement_totals_by_part = [0.0, 0.0, 0.0]
        lip_frame_by_frame_change_by_part = [0.0, 0.0, 0.0]
        for i, facial_image in enumerate(speaker.get_facial_images()):
            image_part = facial_image
            for transform in pre_model_transforms:
                image_part = transform(image_part)
            image_parts.append(cv2.resize(image_part, (100, 100)))
            confidence += facial_image.confidence
            facial_area_size += facial_image.get_facial_area_size()/(facial_image.frame.shape[0]*facial_image.frame.shape[1])
            if len(image_parts) == 1:
                continue
            frame_change = diff_image_structures(image_parts[i-1], image_parts[i])
            for j, frame_range in enumerate(frame_ranges):
                if i > frame_range[0] and i <= frame_range[1]:
                    lip_frame_by_frame_change_by_part[j] += frame_change
                if i == frame_range[1]:
                    lip_movement_totals_by_part[j] = diff_image_structures(image_parts[frame_range[0]], image_parts[frame_range[1]])
            lip_change_frame_by_frame += frame_change
            
            
        facial_area_size /= num_faces
        confidence /= num_faces
        
        lip_movement_start_to_end = lip_movement_totals_by_part[0] + lip_movement_totals_by_part[2]
        start_input_image_to_model = combine_images_horizontally(output_filename=None, images_in_memory_copy=image_parts[:num_input_frames_to_model])
        # get frames in the middle, [2:7]
        middle_input_image_to_model = combine_images_horizontally(output_filename=None, images_in_memory_copy=image_parts[2:2+num_input_frames_to_model])
        end_input_image_to_model = combine_images_horizontally(output_filename=None, images_in_memory_copy=image_parts[-num_input_frames_to_model:])
        # example output of model.predict: ('speaking', tensor(1), tensor([0.0787, 0.9213]))
        # another example output of model.predict: ('silent', tensor(0), tensor([0.9909, 0.0091]))
        # therefore speaking probability is alwasy at index 1 / second item
        predictions_start = model.predict(get_faceai_image(start_input_image_to_model))
        speaker_attributes.append(predictions_start[2][target_index].item())
        predictions_middle = model.predict(get_faceai_image(middle_input_image_to_model))
        speaker_attributes.append(predictions_middle[2][target_index].item())
        predictions_end = model.predict(get_faceai_image(end_input_image_to_model))
        speaker_attributes.append(predictions_end[2][target_index].item())
        speaker_attributes = [
            predictions_start[2][1].item(),
            predictions_middle[2][1].item(),
            predictions_end[2][1].item(),
            lip_change_frame_by_frame,
            lip_movement_start_to_end,
            lip_frame_by_frame_change_by_part[0],
            lip_movement_totals_by_part[0],
            lip_frame_by_frame_change_by_part[1],
            lip_movement_totals_by_part[1],
            lip_frame_by_frame_change_by_part[2],
            lip_movement_totals_by_part[2],
            confidence,
            facial_area_size,
        ]
        speakers_ordered_by_prob.append((speaker, speaker_attributes))
    speakers_ordered_by_prob = sorted(speakers_ordered_by_prob, key=lambda x: x[1][1], reverse=True)
    return speakers_ordered_by_prob



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
        if face.confidence < face_recognition_threshold:
            continue
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
        comparison = DeepFace.verify(
            face.facial_embedding, 
            speaker.get_best_image().facial_embedding, 
            enforce_detection=False,
            )
        if comparison['verified'] and comparison['distance'] < min_distance:
            best_match_index = i
            min_distance = comparison['distance']
    return best_match_index

def get_speaker_embeddings(speaker_image_files):
    '''
    get the speaker embeddings from the speaker image files
    parameters:
        speaker_image_files: the list of speaker image files'''
    speaker_embeddings = {}
    for image_file in speaker_image_files:
        filename_parts = image_file.split('/')
        speaker = filename_parts[-1].split('.')[0]
        # each speaker face image should have exactly one face
        extracted_faces = extract_faces_deepface(image_file)
        extracted_face = None
        for face in extracted_faces:
            if face[deepface_confidence_key] < 0.8:
                print(f"Warning: Detected face with confidence less than 0.8 in {image_file}")
                continue
            if face['facial_area']['left_eye'] is None or face['facial_area']['right_eye'] is None:
                continue 
            extracted_face = face
            break
        if extracted_face is None:
            raise Exception(f"Error: Could not find a face in {image_file}")
        speaker_embeddings[speaker.replace(' ', '_').lower()] = extracted_face[deepface_embedding_key]
    return speaker_embeddings
