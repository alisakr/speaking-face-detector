import cv2
import numpy as np


def combine_images_horizontally(image_paths=None, output_filename=None, images_in_memory_copy=None, target_size=(100, 100)):
    # Combine multiple images into a single image horizontally
    # Parameters:
    # image_paths (list of str): The list of input image filenames, ignored if images_in_memory_copy is not None
    # output_filename (str): The output filename for the combined image
    # images_in_memory_copy (list of cv2.Matlike objects): The list of input image objects
    if image_paths is None and images_in_memory_copy is None:
        raise Exception("Either image_paths or images_in_memory_copy must be provided")
    if images_in_memory_copy is not None:
        images = images_in_memory_copy
    else:
        images = [cv2.imread(image_path) for image_path in image_paths]
    # Resize the images to a fixed size
    images = [cv2.resize(image, target_size) for image in images]
    combined_image = np.concatenate(images, axis=1)
    if output_filename is not None:
        cv2.imwrite(output_filename, combined_image)
    return combined_image


def save_image_into_n_parts_horizontal(image_path, n, save_path_prefix='', output_filenames=None, image_in_memory_copy=None):
    # Split the image into n parts horizontally and save each part as a separate jpg image
    # Parameters:
    # image_path (str): The path to the image file
    # n (int): The number of parts to split the image into
    # save_path_prefix (str): The prefix to use for the output filenames, ignored if output_filenames is not None
    # output_filenames (list of str): The list of output filenames to use, one for each part
    
    
    if output_filenames is None:
        output_filenames = [f"{save_path_prefix}_part{i+1}.jpg" for i in range(n)]
    if len(output_filenames) != n:
        raise Exception("The number of output filenames must match the number of parts to split the image into")
    parts = get_image_n_parts_horizontal(image_path, n, image_in_memory_copy)

    # Save each part as a separate image
    for i, part in enumerate(parts):
        save_path = output_filenames[i]
        cv2.imwrite(save_path, part)


def get_image_n_parts_horizontal(image_path=None, n=1, image_in_memory_copy=None):
    '''
    Split the image into n parts horizontally
    '''
    if image_path is None and image_in_memory_copy is None:
        raise Exception("Either image_path or image_in_memory_copy must be provided")
    # Load the image
    if image_in_memory_copy is not None:    
        image = image_in_memory_copy
    else:
        image = cv2.imread(image_path)
    # Split the image into n parts horizontally
    _, width = image.shape[:2]
    part_width = width // n
    parts = [image[:, i*part_width:(i+1)*part_width] for i in range(n)]
    return parts


def get_image_n_parts_vertical(image_path=None, n=1, image_in_memory_copy=None):
    '''
    Split the image into n parts vertically
    '''

    if image_path is None and image_in_memory_copy is None:
        raise Exception("Either image_path or image_in_memory_copy must be provided")
    # load the image
    if image_in_memory_copy is not None:    
        image = image_in_memory_copy
    else:
        image = cv2.imread(image_path)
    
    # split
    height, _ = image.shape[:2]
    part_height = height // n
    parts = [image[i*part_height:(i+1)*part_height, :] for i in range(n)]
    return parts


def save_image(image_object, filename):
    cv2.imwrite(filename, image_object)


def store_part_of_image(area, frame, output_file):
    cv2.imwrite(output_file, get_part_of_image(area, frame))

def get_part_of_image(area, frame, expand_percent=0):
    # Get a part of the image defined by the area
    # Parameters:   
    # area (dict): The area to crop, with keys x, y, w, h
    # frame (numpy.ndarray): The input image
    # expand_percent (float): The percentage to expand the area by in all 4 directions 0.0 to 100.0
    expand_percent /= 100
    # top 
    x = max(area['x']-int(area['w']*expand_percent), 0)
    # left
    y = max(area['y']-int(area['h']*expand_percent), 0)
    # width
    w = min(area['w']+int(area['w']*expand_percent), frame.shape[1]-x)
    # height
    h = min(area['h']+int(area['h']*expand_percent), frame.shape[0]-y)
    return frame[y:y+h, x:x+w]

def diff_image_structures(image1, image2):
    # Compare two images and return the difference
    # Parameters:
    # image1 (numpy.ndarray): The first image
    # image2 (numpy.ndarray): The second image
    # Returns:
    # numpy.ndarray: The difference between the two images
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse_over_max_sq = np.mean((gray1 - gray2) ** 2)/255**2
    return np.sqrt(mse_over_max_sq)