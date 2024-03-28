import argparse
import os
import sys
sys.path.append('./')
from controllers.video.parse_video import *
from controllers.transcript.parse_transcript import *

def main():
    parser = argparse.ArgumentParser(
        description='Process input transcript and generate array of start,end times and speaker names. Run from reduct_face_project directory')
    parser.add_argument('--input_transcript', default='', type=str, 
                        help='input transcript filename')
    parser.add_argument('--output_folder', default='out', type=str, help='output folder of frames')
    parser.add_argument('--video_path', default='', type=str, help='path to video file')
    parser.add_argument('--ordered_speakers', default=None, type=str, help='each frame should contain every speaker in this order horizontally, ' +
                        'separated by commas.' + 
                        'If not provided, each frame will contain the speaker who is speaking at that time.')
    parser.add_argument('--end_time_seconds', default=-1, type=float, help='end time in seconds')

    args = parser.parse_args()
    print(args)
    print(os.getcwd())
    ordered_speakers = args.ordered_speakers.split(',') if args.ordered_speakers else None
    # ['Emma Vigeland', 'Sam Seder', 'eugene', 'rania']
    times_and_speakers = get_times_and_speakers(
        args.input_transcript, 
        speakers_to_include=ordered_speakers,
        end_time_seconds=args.end_time_seconds if args.end_time_seconds > 0 else None,
    )
    # print(times_and_speakers)
    parse_video_for_speakers(args.video_path, times_and_speakers, args.output_folder, ordered_speakers=ordered_speakers)
    

main()