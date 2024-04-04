import argparse
import os
import sys
sys.path.append('./')

from fastai.vision.learner import load_learner

from controllers.image import get_fast_ai_image
from constants import default_speaking_model

def main():
    '''
    Process image prefix and model to get single prediction
    '''
    parser = argparse.ArgumentParser(
        description='Process image and model to get single prediction')
    parser.add_argument('--model', default=default_speaking_model, type=str, help='model to use')
    parser.add_argument('--image_prefix', default='', type=str, help='path to image file')
    parser.add_argument('--target_label', default=None, type=str, help='expected_class')
    result_array = []
    args = parser.parse_args()
    model = load_learner(args.model)
    image_prefix = args.image_prefix
    path_parts = image_prefix.split('/')
    prefix = path_parts[-1]
    print(f"prefix is {prefix}")
    files_to_skip = set([
    "eugene_23366.666666666668.png", 
    "scott_403203.00000000006.png", 
    "eugene_93166.66666666667.png", 
    "eugene_118166.66666666667.png",
    "rania_64433.333333333336.png",
    "rania_96933.33333333334.png",
    "kyle_739866.6666666666.png",
    "kyle_740833.3333333334.png",
    "kyle_729600.0.png",
    "kyle_729966.6666666667.png",
    "kyle_739033.3333333334.png",
    "kyle_739266.6666666666.png",
    "kyle_729866.6666666666.png",
    "kyle_729800.0000000001.png",
    "kyle_734800.0000000001.png",
    "kyle_730633.3333333334.png",
    "kyle_735066.6666666667.png",
    "kyle_739866.6666666666.png",
    "kyle_741200.0.png",
    "kyle_739600.0.png",
    "kyle_738400.0000000001.png",
    "kyle_741500.0.png",
    "emma_135433.0.png",
    "emma_139866.0.png",
    "emma_135900.0.png",
    "emma_89434.0.png",
    "emma_86100.00000000001.png",
    "emma_87300.0.png",
    "emma_88034.0.png",
    #"emma_370767.0.png",
])
    if len(path_parts) > 1:
        directory = os.path.join(*path_parts[:-1])
        files = [os.path.join(directory,file) for file in os.listdir(directory) if file.startswith(prefix) and file not in files_to_skip]
    else:
        directory = '.'
        files = [file for file in os.listdir(directory) if file.startswith(prefix) and file not in files_to_skip]
    print(f"found {len(files)} files")
    for file in files:
        image = get_fast_ai_image(image_path=file)
        result = model.predict(image)
        if result[0] != args.target_label:
            print(f"mispredicted {file} with speaking_prob {result[2][1].item()}")
            result_array.append(file)
    print("**********")
    print(f"Mislabeled  {len(result_array)} files")
    print(result_array)

main()