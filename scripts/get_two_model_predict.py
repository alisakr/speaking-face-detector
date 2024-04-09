import sys
sys.path.append('./')

from scripts.script_functions import make_predictions

def main():
    '''
    Process input transcript + video, and generate training data for speaker classifier. Run from reduct_face_project dir
    
    Example usage:
    
    python ./scripts/get_predictions
    '''
    make_predictions(image_model_only=False)


main()
