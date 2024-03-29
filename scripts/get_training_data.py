import argparse
import os
import random
import sys
sys.path.append('./')
from controllers.video.parse_video import *
from controllers.transcript.parse_transcript import *
from controllers.command_line_args_parser import get_or_create_transcript
from utils import create_directory_if_not_exists

def main():
    parser = argparse.ArgumentParser(
        description='Process input transcript + video, and generate training data for speaker classifier. Run from reduct_face_project dir')
    parser.add_argument('--input_transcript', default='', type=str, 
                        help='input transcript filename')
    parser.add_argument('--output_directory', default='out', type=str, help='output folder of frames')
    parser.add_argument('--video_path', default='', type=str, help='path to video file')
    parser.add_argument('--speaker_face_directory', default='', type=str, help='path to directory containing images of speakers')
    parser.add_argument('--end_time_seconds', default=-1, type=float, help='end time in seconds')
    parser.add_argument('--start_time_seconds', default=-1, type=float, help='start time in seconds')
    parser.add_argument("--api_key_yaml", default=None, type=str, help="yaml file with api key")
    parser.add_argument("--doc_id", default="", type=str, help="doc id for reduct")

    args = parser.parse_args()
    print(args)
    print(os.getcwd())
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
        ignore_first_last_word=True,
        )
    # randomly shuffle times and speakers so that we get a diverse training set quickly
    random.shuffle(times_and_speakers)
    speaker_image_files = [
        os.path.join(args.speaker_face_directory, f) for f in os.listdir(args.speaker_face_directory) if os.path.isfile(
            os.path.join(args.speaker_face_directory, f))]
    print(speaker_image_files)
    create_directory_if_not_exists(args.output_directory)
    create_directory_if_not_exists(args.output_directory + '/speaking')
    create_directory_if_not_exists(args.output_directory + '/silent')
    parse_video_for_speakers_middle_n_frames(
        args.video_path, times_and_speakers, args.output_directory, speaker_image_files=speaker_image_files, use_lips_only=True)
    

main()