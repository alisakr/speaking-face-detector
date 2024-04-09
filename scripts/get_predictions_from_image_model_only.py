
import sys
sys.path.append('./')

from scripts.script_functions import make_predictions

def main():
    '''
    Example run: 
    python scripts/get_predictions_from_image_model_only.py 
    --api_key_yaml my_api_key.yaml 
    --video_path "../../Downloads/64813c41af.mp4" 
    --output_directory out
    
    check scripts/script_functions.py for more details and more optional arguments
    '''
    make_predictions(image_model_only=True)

main()