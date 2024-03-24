import argparse
import os
import sys
import time
import yaml
sys.path.append('./')

from fastai.vision.learner import load_learner

from gateways.reduct import *
from parsers.video.parse_video import *
from parsers.transcript.parse_transcript import *


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
    parser.add_argument('--model', default='', type=str, help='default model to use')
    parser.add_argument('--video_path', default='', type=str, help='path to video file')
    parser.add_argument('--target', default='speaking', type=str, help='target label class')
    parser.add_argument('--end_time_seconds', default=-1, type=float, help='end time in seconds')
    parser.add_argument('--start_time_seconds', default=-1, type=float, help='start time in seconds')
    parser.add_argument("--output_directory", default="out", type=str, help="output directory of frames")
    parser.add_argument("--max_segments", default=500, type=int, help="number of segments to try")
    parser.add_argument("--api_key_yaml", default=None, type=str, help="yaml file with api key")
    parser.add_argument("--doc_id", default="", type=str, help="doc id for reduct")
    args = parser.parse_args()
    api_key = None
    if args.api_key_yaml:
        api_file = open(args.api_key_yaml, "r")
        api_config = yaml.safe_load(api_file)
        api_file.close()
        api_key = api_config["api_key"]
    
    if args.start_time_seconds < 0:
        args.start_time_seconds = None
    if args.end_time_seconds < 0:
        args.end_time_seconds = None
    model = load_learner(args.model)
    print("args parsed")
    start_time = round(time.time() * 1000)
    if api_key:
        print("api key found")
        transcript_temp = f'{args.doc_id}_{start_time}_transcript.json'
        save_transcript(args.doc_id, transcript_temp, api_key_override=api_key, format="json")
        times_and_speakers = get_times_and_speakers(
            transcript_temp, 
            start_time_seconds=args.start_time_seconds, 
            end_time_seconds=args.end_time_seconds,
            image_for_each_segment=True,
            )
    else:
        times_and_speakers = get_times_and_speakers(
            args.input_transcript, 
            start_time_seconds=args.start_time_seconds, 
            end_time_seconds=args.end_time_seconds,
            image_for_each_segment=True,
            )
    print("times_and_speakers parsed")
    print(int(args.max_segments))
    parse_video_for_classifying_speakers(args.video_path, times_and_speakers, model, args.target, args.output_directory, num_segments_to_try=int(args.max_segments))
    end_time = round(time.time() * 1000)
    print(f"Time taken: {end_time - start_time}ms")

main()