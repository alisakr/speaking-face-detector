import argparse
import sys
import time
sys.path.append('./')

from fastai.vision.learner import load_learner

from gateways.reduct import *
from constants import *
from controllers.video.parse_video import *
from controllers.transcript.parse_transcript import *
from controllers.command_line_args_parser import get_or_create_transcript
from utils import create_directory_if_not_exists


def main():
    '''
    Process input transcript + video, and generate training data for speaker classifier. Run from reduct_face_project dir
    
    Example usage:
    
    python ./scripts/get_predictions.py 
    --input_transcript electronic_intifada_transcript.json 
    --video_path "../../Downloads/gaza_port_4Ez1gebPyHs.mp4" 
    --model "models/lips_speaking_model_0_finetune_of_2_random_split.pkl"
    '''
    parser = argparse.ArgumentParser(
        description='Process input transcript + video, and generate training data for speaker classifier. Run from reduct_face_project dir')
    parser.add_argument('--input_transcript', default='', type=str, 
                        help='input transcript filename')
    parser.add_argument('--model', default=constants.default_speaking_model, type=str, help='model to use')
    parser.add_argument('--video_path', default='', type=str, help='path to video file')
    parser.add_argument('--target', default='speaking', type=str, help='target label class')
    parser.add_argument('--end_time_seconds', default=-1, type=float, help='end time in seconds')
    parser.add_argument('--start_time_seconds', default=-1, type=float, help='start time in seconds')
    parser.add_argument("--output_directory", default="out", type=str, help="output directory of frames")
    parser.add_argument("--debug_output_directory", default=None, type=str, help="runs in debug mode when set and outputs to directory")
    parser.add_argument("--max_segments", default=5000, type=int, help="number of segments to try")
    parser.add_argument("--api_key_yaml", default=None, type=str, help="yaml file with api key")
    parser.add_argument("--doc_id", default="", type=str, help="doc id for reduct")
    parser.add_argument("--video_url", default="", type=str, help="url for video, if not local...not yet implemented")
    args = parser.parse_args()
    
    if args.start_time_seconds < 0:
        args.start_time_seconds = None
    if args.end_time_seconds < 0:
        args.end_time_seconds = None
    model = load_learner(args.model)
    print("args parsed")
    start_time = round(time.time() * 1000)
    transcript_json = get_or_create_transcript(args)
    times_and_speakers = get_times_and_speakers(
            None,
            transcript_json, 
            start_time_seconds=args.start_time_seconds, 
            end_time_seconds=args.end_time_seconds,
            image_for_each_segment=True,
            )
    print("times_and_speakers parsed")
    print(int(args.max_segments))
    debug_output_directory = args.debug_output_directory
    debug_mode = debug_output_directory is not None
    if debug_mode:
        create_directory_if_not_exists(debug_output_directory)
    create_directory_if_not_exists(args.output_directory)
    parse_video_for_classifying_speakers(
        args.video_path, 
        times_and_speakers, 
        model, 
        args.target, 
        args.output_directory, 
        num_segments_to_try=int(args.max_segments),
        debug_mode=debug_mode,
        debug_mode_output_folder=debug_output_directory,
        )
    end_time = round(time.time() * 1000)
    print(f"Time taken: {end_time - start_time}ms")

main()