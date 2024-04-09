deepface_embedding_key = 'face'
deepface_confidence_key = 'confidence'
deepface_facial_area_key = 'facial_area'
# threshold for confidence in the image being a face
face_recognition_threshold = 0.88

max_wait_for_transcript_seconds = 600
reduct_api_key = 'reduct_api_key'
reduct_organization_key = 'reduct_organization_id'
reduct_transcript_complete_status = "transcribed"
default_random_forest_model = 'models/random_forest_all_data_regressor_500_forests_50_min_leaf_91_accuracy_44_of_48_at_threshold_of_50_percent_split_15.pkl'
#default_speaking_model = 'models/clean_briahna_2_krystal_kyle_eugene_emma_4_sunday_2_obstructed_lips_even_split_finetune_of_4.pkl'
default_speaking_model = 'models/lips_speaking_model_0_finetune_of_2_random_split.pkl'

# number of frames to read from video for every word in transcript
num_frames_to_read = 9
num_input_frames_to_model = 5

# random forest fields
speaker_prob_start = "speaker_prob_start"
speaker_prob_middle = "speaker_prob_middle"
speaker_prob_end = "speaker_prob_end"
lip_change_frame_by_frame = "lip_change_frame_by_frame"
lip_change_start_frame_by_frame = "lip_change_start_frame_by_frame"
lip_change_middle_frame_by_frame = "lip_change_middle_frame_by_frame"
lip_change_end_frame_by_frame = "lip_change_end_frame_by_frame"
lip_movement_total_in_start = "lip_movement_total_in_start"
lip_movement_total_in_middle = "lip_movement_total_in_middle"
lip_movement_total_in_end = "lip_movement_total_in_end"
lip_movement_start_to_end = "lip_movement_start_to_end"
facial_confidence = "facial_confidence"
facial_area_size = "facial_area_size"
speaker_probability_columns = [
    speaker_prob_start, 
    speaker_prob_middle, 
    speaker_prob_end, 
    lip_change_frame_by_frame,
    lip_movement_start_to_end,
    lip_change_start_frame_by_frame,
    lip_movement_total_in_start,
    lip_change_middle_frame_by_frame,
    lip_movement_total_in_middle,
    lip_change_end_frame_by_frame,
    lip_movement_total_in_end,    
    facial_confidence, 
    facial_area_size,
    ]

features = ['speaker_prob_start', 'speaker_prob_middle', 'speaker_prob_end',
       'lip_change_frame_by_frame', 'lip_movement_start_to_end',
       'lip_change_start_frame_by_frame', 'lip_movement_total_in_start',
       'lip_change_middle_frame_by_frame', 'lip_movement_total_in_middle',
       'lip_change_end_frame_by_frame', 'lip_movement_total_in_end',
       'facial_confidence', 'facial_area_size',
       'frame_max_speaker_prob_start', 'frame_max_speaker_prob_middle',
       'frame_max_speaker_prob_end', 'frame_max_lip_change_frame_by_frame',
       'frame_max_lip_movement_start_to_end',
       'frame_max_lip_change_start_frame_by_frame',
       'frame_max_lip_movement_total_in_start',
       'frame_max_lip_change_middle_frame_by_frame',
       'frame_max_lip_movement_total_in_middle',
       'frame_max_lip_change_end_frame_by_frame',
       'frame_max_lip_movement_total_in_end', 'frame_max_facial_confidence',
       'frame_max_facial_area_size', 'frame_min_speaker_prob_start',
       'frame_min_speaker_prob_middle', 'frame_min_speaker_prob_end',
       'frame_min_lip_change_frame_by_frame',
       'frame_min_lip_movement_start_to_end',
       'frame_min_lip_change_start_frame_by_frame',
       'frame_min_lip_movement_total_in_start',
       'frame_min_lip_change_middle_frame_by_frame',
       'frame_min_lip_movement_total_in_middle',
       'frame_min_lip_change_end_frame_by_frame',
       'frame_min_lip_movement_total_in_end', 'frame_min_facial_confidence',
       'frame_min_facial_area_size', 'frame_mean_speaker_prob_start',
       'frame_mean_speaker_prob_middle', 'frame_mean_speaker_prob_end',
       'frame_mean_lip_change_frame_by_frame',
       'frame_mean_lip_movement_start_to_end',
       'frame_mean_lip_change_start_frame_by_frame',
       'frame_mean_lip_movement_total_in_start',
       'frame_mean_lip_change_middle_frame_by_frame',
       'frame_mean_lip_movement_total_in_middle',
       'frame_mean_lip_change_end_frame_by_frame',
       'frame_mean_lip_movement_total_in_end', 'frame_mean_facial_confidence',
       'frame_mean_facial_area_size', 'max_speaker_prob', 'max_frame_prob',
       'has_max_speaker', 'has_max_speaker_start', 'has_max_speaker_middle', 'has_max_speaker_end']

max_speaker_prob_formula_field = 'max_speaker_prob'
max_frame_prob_formula_field = 'max_frame_prob'
has_max_speaker_formula_field = 'has_max_speaker'
has_max_speaker_start_formula_field = 'has_max_speaker_start'
has_max_speaker_middle_formula_field = 'has_max_speaker_middle'
has_max_speaker_end_formula_field = 'has_max_speaker_end'
