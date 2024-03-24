import argparse
import os
import sys
sys.path.append('./')
from parsers.video.parse_video import *
from parsers.transcript.parse_transcript import *

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

    args = parser.parse_args()
    print(args)
    print(os.getcwd())
    if args.start_time_seconds < 0:
        args.start_time_seconds = None
    if args.end_time_seconds < 0:
        args.end_time_seconds = None
    times_and_speakers = get_times_and_speakers(args.input_transcript, start_time_seconds=args.start_time_seconds, end_time_seconds=args.end_time_seconds)
    speaker_image_files = [
        os.path.join(args.speaker_face_directory, f) for f in os.listdir(args.speaker_face_directory) if os.path.isfile(
            os.path.join(args.speaker_face_directory, f))]
    print(speaker_image_files)
    parse_video_for_speakers_middle_n_frames(
        args.video_path, times_and_speakers, args.output_directory, speaker_image_files=speaker_image_files, use_lips_only=True)
    

main()