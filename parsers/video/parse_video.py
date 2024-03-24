import cv2
import os
from deepface import DeepFace
from parsers.image import (
    combine_images_horizontally, 
    extract_faces_deepface,
    get_faceai_image,
    get_image_n_parts_vertical,
    get_part_of_image,
    save_image_into_n_parts_horizontal,
)
from constants import (
    deepface_embedding_key,
    deepface_confidence_key,
    face_recognition_threshold,
)

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
    # speaker name 
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
        current_speaker = segment_info[2].replace(' ', '_').lower()
        if current_speaker not in speaker_embeddings:
            continue
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
        target_probability_threshold=0.5,
        use_lips_only=True,
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
    for segment_info in transcript_segments:
        # Step 1: get speaker, time.
        start_time_ms = segment_info[0] * 1000
        end_time_ms = segment_info[1] * 1000
        current_speaker = segment_info[2]
        # Set the video capture to get n/2 segments prior to the middle, the middle segment and (n/2)-1 after the middle
        num_pre_middle_frames = n // 2
        current_time_ms = (start_time_ms + end_time_ms) / 2
        current_time_ms -= num_pre_middle_frames * time_per_frame_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
        faces_by_person = []
        # we want to take the face with the strongest face image for each person
        # faces_by_person_index_to_best_face therefore is...
        # faces_by_person_index: (face_index_in_list, confidence)
        faces_by_person_index_to_best_face = []
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
            if len(faces) == 0:
                break
            
            if i == 0:
                for j in range(len(faces)):
                    # WARNING: we don't store i/which frame here because we only accept n consecutive frames
                    # if that changes then we need to store the frame index, not just the face
                    if faces[j][deepface_confidence_key] < face_recognition_threshold:
                        continue
                    faces_by_person.append([faces[j]])
                    faces_by_person_index_to_best_face.append((i, faces[j][deepface_confidence_key]))
            else:
                best_match_for_new_frame_faces = {}
                for j, new_face in enumerate(faces):
                    min_distance = 1.0
                    best_match = None
                    for k, faces_for_person in enumerate(faces_by_person):
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
                        if len(faces_for_person) < i:
                            continue
                        best_face_index_of_existing_person = faces_by_person_index_to_best_face[k][0]
                        face_test = DeepFace.verify(
                            new_face[deepface_embedding_key], 
                            faces_for_person[best_face_index_of_existing_person][deepface_embedding_key], enforce_detection=False)
                        if face_test["verified"]:
                            if face_test["distance"] < min_distance:
                                min_distance = face_test["distance"]
                                best_match = k
                    if best_match is not None:
                        if best_match not in best_match_for_new_frame_faces:
                            best_match_for_new_frame_faces[best_match] = (j, min_distance)
                        else:
                            if min_distance < best_match_for_new_frame_faces[best_match][1]:
                                best_match_for_new_frame_faces[best_match] = (j, min_distance)
                            
                for j, face_list in enumerate(faces_by_person):
                    # here we are looping through existing faces and checking if the new faces match
                    if j in best_match_for_new_frame_faces:
                        new_face = faces[best_match_for_new_frame_faces[j][0]]
                        face_list.append(new_face)
                        if new_face[deepface_confidence_key] > faces_by_person_index_to_best_face[j][1]:
                            faces_by_person_index_to_best_face[j] = (i, new_face[deepface_confidence_key])
        # now we have the up to n faces for each person
        # TODO: make this loop mapping function or list comprehension
        def get_facial_area_adjusted(facial_area, frame, padding=0):
            face_part_of_frame = get_part_of_image(facial_area, frame, padding)
            if use_lips_only:
                # assume lips to be in the bottom third of the face
                return get_image_n_parts_vertical(image_in_memory_copy=face_part_of_frame, n=3)[-1]
            return face_part_of_frame
        potential_speakers = []
        potential_speakers_raw = []
        for faces in faces_by_person:
            if len(faces) < n:
                continue
            image_parts = [get_facial_area_adjusted(face['facial_area'], frames[j]) for j, face in enumerate(faces)]
            image_parts_raw = [get_part_of_image(face['facial_area'], frames[j]) for j, face in enumerate(faces)]
            potential_speakers.append(image_parts)
            potential_speakers_raw.append(image_parts_raw)
        num_silent = 0
        num_speaking = 0
        max_speaker_prob = 0
        max_speaker_index = len(potential_speakers)
        min_speaker_prob = 1.0
        preds = []
        for i, potential_speaker in enumerate(potential_speakers):
            input_image_to_model = combine_images_horizontally(output_filename=None, images_in_memory_copy=potential_speaker)
            # TODO: apply model here
            predictions = model.predict(get_faceai_image(input_image_to_model))
            preds.append(predictions)
            speaker_prob = predictions[2].max()
            if predictions[0] == target:
                num_speaking += 1
                if speaker_prob > max_speaker_prob:
                    max_speaker_prob = speaker_prob
                    max_speaker_index = i
                if speaker_prob < min_speaker_prob:
                    min_speaker_prob = speaker_prob

            else:
                num_silent += 1 
        for i, potential_speaker in enumerate(potential_speakers):
            raw_image = combine_images_horizontally(output_filename=None, images_in_memory_copy=potential_speakers_raw[i])
            predictions = preds[i]
            print(predictions[2])
            if i == max_speaker_index:
                label = 'speaking'
                speaker_prob = predictions[2].max()
                cv2.imwrite(f"{output_folder}/{current_speaker}_{label}_{current_time_ms}_{speaker_prob}.png", raw_image)
            else:
                # pass
                label = 'silent'
                if predictions[0] == target:
                    speaker_prob = predictions[2].max()
                else:
                    speaker_prob = predictions[2].min()
                # cv2.imwrite(f"{output_folder}/{current_speaker}_{label}_{current_time_ms}_{speaker_prob}_{i}.png", input_image_to_model)
                cv2.imwrite(f"{output_folder}/{current_speaker}_{label}_{current_time_ms}_{speaker_prob}_{i}_raw.png", raw_image)
        cv2.imwrite(f"{output_folder}/{current_speaker}_{current_time_ms}_frame.png", frames[n//2])
        print(f"current_speaker: {current_speaker} num_silent: {num_silent} num_speaking: {num_speaking} max_speaker_prob: {max_speaker_prob} min_speaker_prob: {min_speaker_prob} max_speaker_index: {max_speaker_index}")
        
        processed_segments += 1
        if num_segments_to_try is not None and processed_segments >= num_segments_to_try:
            break




                


        

    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

def get_most_likely_speaker(cap, n=5, use_lips_only=True):
    '''
    read n frames and return the most likely speaker
    parameters:
        cap: the video capture object, set to the first frame to read
        n: the number of frames to read
        use_lips_only: whether to use only the lips for the speaker classification
    '''
    potential_speakers = []
    frames = []
    for i in range(n):
        # Read the next frame from the video
        _, frame = cap.read()
        if frame is None:
            return None
        frames.append(frame)
        faces = None
        try:
            faces = extract_faces_deepface(frame)
        except Exception as e:
            print(f"Error: {e}")
            continue
        if len(faces) == 0:
            return None
        if i == 0:
            for j in range(len(faces)):
                # WARNING: we don't store i/which frame here because we only accept n consecutive frames
                # if that changes then we need to store the frame index, not just the face
                # TODO: make this use the entities
                potential_speakers.append([faces[j]])
