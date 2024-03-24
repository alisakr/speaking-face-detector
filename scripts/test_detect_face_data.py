import argparse
import sys
sys.path.append('./')

from parsers.image.parse_image import extract_faces_caffemodel, detect_faces_model_v1, save_image

def main():
    parser = argparse.ArgumentParser(
        description='Detect faces in an image. Run from reduct_face_project directory')
    parser.add_argument('--input_image', default='', type=str, 
                        help='input image filename')
    parser.add_argument('--output_folder', default='out', type=str, help='output folder of frames')
    args = parser.parse_args()
    print(args)
    initial_images = extract_faces_caffemodel(image_path=args.input_image)
    for i, image in enumerate(initial_images):
        save_image(image, f"{args.output_folder}/v1_{i}.png")
        faces = extract_faces_caffemodel(image_object=image)
        for j, face in enumerate(faces):
            save_image(face, f"{args.output_folder}/face_{i}_{j}.png")
main()