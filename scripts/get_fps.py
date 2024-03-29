import cv2

def get_fps(filename):
    try:
        video_capture = cv2.VideoCapture(filename)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        return fps
    except Exception as e:
        print("Error:", e)
        return None

if __name__ == "__main__":
    filename = input("Enter the path to the MP4 file: ")
    fps = get_fps(filename)
    if fps is not None:
        print("FPS of the video:", fps)