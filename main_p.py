import multiprocessing as mp
import time

import cv2
from imageai.Detection import ObjectDetection

from utils import vid_to_img

DETECTOR = ObjectDetection()
DETECTOR.setModelTypeAsYOLOv3()
DETECTOR.setModelPath("model/yolo.h5")
DETECTOR.loadModel()

CUSTOM = DETECTOR.CustomObjects(person=True)


def detect_person(image, count: int):
    detections = DETECTOR.detectObjectsFromImage(
        custom_objects=CUSTOM,
        input_image=image,
        minimum_percentage_probability=30,
        output_image_path=f"media/detected_images_from_video/frame__{count}.jpg"
    )

    # cv2.imwrite(f"media/detected_images_from_video/frame::{count}.jpg", detections[0])
    return detections


if __name__ == "__main__":
    cap = cv2.VideoCapture("media/input_video/video_2.mp4")

    success, files = vid_to_img(cap, folder="media/images_from_video")
    # success = True
    # files = [f"media/images_from_video/frame__{i}.jpg" for i in range(1, 245)]

    if success:
        start_time = time.time()

        pool = mp.Pool(mp.cpu_count())

        result = pool.starmap(
            detect_person,
            [(img, count) for count, img in enumerate(files)]
        )

        pool.close()

        print(f"Time taken: {time.time() - start_time}")
