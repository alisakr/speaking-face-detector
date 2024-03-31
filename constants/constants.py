deepface_embedding_key = 'face'
deepface_confidence_key = 'confidence'
deepface_facial_area_key = 'facial_area'
# threshold for confidence in the image being a face
face_recognition_threshold = 0.88

max_wait_for_transcript_seconds = 600
reduct_api_key = 'reduct_api_key'
reduct_organization_key = 'reduct_organization_id'
reduct_transcript_complete_status = "transcribed"
default_speaking_model = 'models/land_day_2_obstructed_lips_even_split_finetune_of_3.pkl'
speaker_prob_start = "speaker_prob_start"
speaker_prob_middle = "speaker_prob_middle"
speaker_prob_end = "speaker_prob_end"
lip_change_frame_by_frame = "lip_change_frame_by_frame"
lip_movement_start_to_end = "lip_movement_start_to_end"
facial_confidence = "facial_confidence"
facial_area_size = "facial_area_size"
speaker_probability_columns = [
    speaker_prob_start, 
    speaker_prob_middle, 
    speaker_prob_end, 
    lip_change_frame_by_frame, 
    lip_movement_start_to_end, 
    facial_confidence, 
    facial_area_size,
    ]