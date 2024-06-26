import argparse
import sys
import csv
sys.path.append('./')

from controllers.transcript.parse_transcript import *
from controllers.command_line_args_parser import get_or_create_transcript
from controllers.transforms.transforms import GrayscaleTransform
from controllers.video.predict_words import SpeakerWordPredictor
from controllers.video.video_predictor import VideoPredictor
from entities.transcript import Transcript
from utils import create_directory_if_not_exists

def make_predictions(image_model_only=False):
    '''
    Process input transcript + video, and generate training data for speaker classifier. Run from reduct_face_project dir
    
    Example usage:
    
    python ./scripts/get_predictions
    '''
    parser = argparse.ArgumentParser(
        description='Process input transcript + video, and generate training data for speaker classifier. Run from reduct_face_project dir')
    parser.add_argument('--input_transcript', default='', type=str, 
                        help='input transcript filename')
    parser.add_argument('--video_path', default='', type=str, help='path to video file')
    parser.add_argument('--end_time_seconds', default=-1, type=float, help='end time in seconds')
    parser.add_argument('--start_time_seconds', default=-1, type=float, help='start time in seconds')
    parser.add_argument("--output_directory", default=None, type=str, help="output directory of frames")
    parser.add_argument("--debug_output_directory", default=None, type=str, help="runs in debug mode when set and outputs to directory")
    parser.add_argument("--api_key_yaml", default=None, type=str, help="yaml file with api key")
    parser.add_argument("--doc_id", default="", type=str, help="doc id for reduct")
    parser.add_argument("--segments_csv", default=None, type=str, help="segment csv file")
    parser.add_argument("--output_json", default=None, type=str, help="output json file")

    args = parser.parse_args()
    
    if args.start_time_seconds < 0:
        args.start_time_seconds = None
    if args.end_time_seconds < 0:
        args.end_time_seconds = None
    predictor = SpeakerWordPredictor(image_model_only=image_model_only)
    transcript_json = get_or_create_transcript(args)

    times_and_speakers = get_times_and_speakers(
            None,
            transcript_json, 
            start_time_seconds=args.start_time_seconds, 
            end_time_seconds=args.end_time_seconds,
            )
    print(times_and_speakers)
    create_directory_if_not_exists(args.output_directory)
    
    transcript = Transcript(transcript_json, times_and_speakers)
    if args.segments_csv is not None:
        with open(args.segments_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "speaker", "word"])
            writer.writerows(transcript.segments)

    video = VideoPredictor(args.video_path, transcript)
    predictor.predict_speaker_words(video, args.output_directory, args.output_json)

