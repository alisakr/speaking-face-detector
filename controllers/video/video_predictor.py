import cv2

from entities.transcript import Transcript


class VideoPredictor:

    def __init__(self, video_path: str, transcript: Transcript):
        self.video_path = video_path
        self.transcript = transcript
        self.cap = None
        self.video_fps = 0
        self.num_frames = 0
    
    def open_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file")
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def close_video(self):
        self.cap.release()
        self.cap = None
    
    def set_video(self, set_time_ms):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, set_time_ms)
    

       
        