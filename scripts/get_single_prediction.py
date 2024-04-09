import argparse
import sys

sys.path.append('./')

from fastai.vision.learner import load_learner

from controllers.image import get_fast_ai_image
from constants import default_speaking_model

def main():
    '''
    Process image and model to get single prediction
    '''
    parser = argparse.ArgumentParser(
        description='Process image and model to get single prediction')
    parser.add_argument('--model', default=default_speaking_model, type=str, help='model to use')
    parser.add_argument('--image_path', default='', type=str, help='path to image file')
    result_array = []
    args = parser.parse_args()
    model = load_learner(args.model)
    image = get_fast_ai_image(image_path=args.image_path)
    print("going to predict")
    result = model.predict(image)
    result_array.append(result[2][1].item())
    print(result)
    print(result_array)
    print(result_array[0]+ 0.51)

main()