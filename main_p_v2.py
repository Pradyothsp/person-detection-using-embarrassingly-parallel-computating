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

    if success:
        start_time = time.time()

        # Multiprocessing pool
        pool = mp.Pool(mp.cpu_count())

        result = pool.starmap(
            detect_person,
            [(img, count) for count, img in enumerate(files)]
        )

        pool.close()

        print("Creating video")

        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter()
        output_file_name = "media/output_video/parallel_detected_2.mp4"
        out.open(output_file_name, fourcc, fps, (width, height), True)

        _ = [out.write(cv2.imread(f"media/detected_images_from_video/frame__{i}.jpg")) for i in range(len(files))]

        print(f"Time taken: {time.time() - start_time}")
