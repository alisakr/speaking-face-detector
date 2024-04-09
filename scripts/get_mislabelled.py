import argparse
import os
import sys
sys.path.append('./')

from fastai.vision.learner import load_learner
from fastai.vision.all import PILImage, Transform
import numpy as np

from controllers.image import get_fast_ai_image, combine_images_horizontally, get_image_n_parts_horizontal
from constants import default_speaking_model


# Define a grayscale transformation
class GrayscaleTransform(Transform):
    def __init__(self):
        pass
    
    def encodes(self, x: PILImage):
        return x.convert('L')

def get_person(filepath_string):
    file_parts = filepath_string.split("/")
    filename = file_parts[-1]
    filename_parts = filename.split("_")
    return filename_parts[0]
    
def get_frame_time(filepath_string):
    file_parts = filepath_string.split("/")
    filename = file_parts[-1]
    filename_parts = filename.split("_")
    ms_filepart = filename_parts[1].split(".png")[0]
    try:
        middle_frame_ms = float(ms_filepart)
    except:
        middle_frame_ms = 0.0
    return middle_frame_ms

def get_file_order(filepath):
    filepath_string = str(filepath)
    file_parts = filepath_string.split("/")
    filename = file_parts[-1]
    person = get_person(filename)
    filename_parts = filename.split("_")
    if len(filename_parts) > 2:
        video_name = filename_parts[-1]
    else:
        video_name = ".png"
    middle_frame_ms = get_frame_time(filename)
    
    return (video_name, person, middle_frame_ms)

def main():
    '''
    Process image prefix and model to get single prediction
    '''
    parser = argparse.ArgumentParser(
        description='Process image and model to get single prediction')
    parser.add_argument('--model', default=default_speaking_model, type=str, help='model to use')
    parser.add_argument('--image_prefix', default='', type=str, help='path to image file')
    parser.add_argument('--target_label', default=None, type=str, help='expected_class')
    parser.add_argument('--min_word_distance_ms', default=0, type=float, help='minimum time between words in milliseconds')
    parser.add_argument('--show_mislabelled', default="True", type=str, help='show the names of mislabelled files')
    result_array = []
    args = parser.parse_args()
    model = load_learner(args.model)
    image_prefix = args.image_prefix
    path_parts = image_prefix.split('/')
    prefix = path_parts[-1]
    result_probabilities = []
    print(f"prefix is {prefix}")
    is_dir = os.path.isdir(image_prefix)
    if args.show_mislabelled.lower() == "true":
        args.show_mislabelled = True
    else:
        args.show_mislabelled = False
    if is_dir:
        files = [os.path.join(image_prefix,file) for file in os.listdir(image_prefix)]
    elif len(path_parts) > 1:
        directory = os.path.join(*path_parts[:-1])
        files = [os.path.join(directory,file) for file in os.listdir(directory) if file.startswith(prefix)]
    else:
        directory = '.'
        files = [file for file in os.listdir(directory) if file.startswith(prefix)]
    files = sorted(files, key=get_file_order)
    last_frame_time_ms = 0.0
    last_speaker = None
    min_word_distance_ms = args.min_word_distance_ms
    num_evaluated = 0
    for i, file in enumerate(files):
        frame_time_ms = get_frame_time(file)
        diff_backwards = frame_time_ms - last_frame_time_ms
        last_frame_time_ms = frame_time_ms
        speaker = get_person(file)
        if i > 0 and speaker == last_speaker and diff_backwards < min_word_distance_ms and diff_backwards > 0:
            continue
        last_speaker = speaker
        if i < len(files) - 1:
            next_frame_time_ms = get_frame_time(files[i+1])
            next_speaker = get_person(files[i+1])
            if next_speaker == speaker:
                diff_forwards = next_frame_time_ms - frame_time_ms
                if diff_forwards < min_word_distance_ms and diff_forwards > 0:
                    continue

        if "three_image" in args.model:
            image_parts = get_image_n_parts_horizontal(file, 5)
            combined_image = combine_images_horizontally(images_in_memory_copy=image_parts[1:4])
            image = get_fast_ai_image(image_object=combined_image)
        else:
            image = get_fast_ai_image(image_path=file)
        result = model.predict(image)
        num_evaluated += 1
        if result[0] != args.target_label:
            if args.show_mislabelled:
                print(f"mispredicted {file} with speaking_prob {result[2][1].item()}")
            result_array.append(file)
    print("**********")
    print(f"Mislabeled  {len(result_array)} files")
    print(f"Evaluated {num_evaluated} files")
    print(f"Percentage correct {1.0 - len(result_array)/num_evaluated}")
    if args.show_mislabelled:
        print(result_array)

main()