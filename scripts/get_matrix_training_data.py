import argparse
import os
import sys
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow log level to suppress unnecessary messages
warnings.filterwarnings("ignore")  # Ignore warnings
sys.path.append('./')
import logging
logging.disable(logging.CRITICAL)

from fastai.vision.learner import load_learner

from controllers.video.parse_video import parse_video_for_matrix_data
from controllers.transcript.parse_transcript import get_times_and_speakers
from controllers.command_line_args_parser import get_or_create_transcript
from constants import default_speaking_model
from utils import create_directory_if_not_exists

def main():
    
    parser = argparse.ArgumentParser(
        description='Process image and model to get single prediction')
    parser.add_argument('--model', default=default_speaking_model, type=str, help='model to use')
    parser.add_argument('--output_directory', default=None, type=str, help='output folder of frames for debugging')
    parser.add_argument('--video_path', default='', type=str, help='path to video file')
    parser.add_argument('--speaker_face_directory', default='', type=str, help='path to directory containing images of speakers')
    parser.add_argument('--end_time_seconds', default=-1, type=float, help='end time in seconds')
    parser.add_argument('--start_time_seconds', default=-1, type=float, help='start time in seconds')
    parser.add_argument("--api_key_yaml", default=None, type=str, help="yaml file with api key")
    parser.add_argument("--doc_id", default="", type=str, help="doc id for reduct")
    parser.add_argument("--output_csv", type=str, help="output csv file")
    parser.add_argument("--speaker_name", type=str, help="speaker name")

    args = parser.parse_args()
    transcript_json = get_or_create_transcript(args)
    if args.start_time_seconds < 0:
        args.start_time_seconds = None
    if args.end_time_seconds < 0:
        args.end_time_seconds = None
    times_and_speakers = get_times_and_speakers(
        None,
        transcript_json, 
        start_time_seconds=args.start_time_seconds, 
        end_time_seconds=args.end_time_seconds,
        image_for_each_segment=False,
        speaker_key='speaker_name',
        )
    if args.output_directory is not None:
        create_directory_if_not_exists(args.output_directory)
    
    for i, segment in enumerate(times_and_speakers):
        times_and_speakers[i] = (segment[0], segment[1], args.speaker_name, segment[3])
    
    print(times_and_speakers)
    model = load_learner(args.model)
    parse_video_for_matrix_data(
        args.video_path, 
        times_and_speakers,
        model,
       [args.speaker_face_directory + f"/{args.speaker_name}.png"], 
        args.output_csv, 
        output_image_folder=args.output_directory,
        )



main()