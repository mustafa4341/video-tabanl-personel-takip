import cv2
import os

class VideoWriter:
    def __init__(self, output_path="outputs/result.mp4", fps=30):
        self.output_path=output_path
        self.fps=fps
        self.writer=None
        os.makedirs("outputs",exist_ok=True)
        
    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (w, h)
            )
        self.writer.write(frame)
            
    def release(self):
        if self.writer is not None:
            self.writer.release()
            print(f"Video kaydedildi: {self.output_path}")
    