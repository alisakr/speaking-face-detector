import argparse
import os
import sys
sys.path.append('./')
from parsers.video.parse_video import *
from parsers.transcript.parse_transcript import *

def main():
    parser = argparse.ArgumentParser(
        description='Process input transcript and generate array of start,end times and speaker names. Run from reduct_face_project directory')
    parser.add_argument('--input_transcript', default='', type=str, 
                        help='input transcript filename')
    parser.add_argument('--output_folder', default='out', type=str, help='output folder of frames')
    parser.add_argument('--video_path', default='', type=str, help='path to video file')

    args = parser.parse_args()
    print(args)
    print(os.getcwd())
    times_and_speakers = get_times_and_speakers(args.input_transcript, speakers_to_include=['Emma Vigeland', 'Sam Seder', 'eugene', 'rania'])
    print(times_and_speakers)
    # parse_video_for_speakers(args.video_path, times_and_speakers, args.output_folder)
    

main()