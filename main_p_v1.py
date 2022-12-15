import multiprocessing as mp
import time

import cv2
from imageai.Detection import ObjectDetection

from utils import get_all_frames

DETECTOR = ObjectDetection()
DETECTOR.setModelTypeAsYOLOv3()
DETECTOR.setModelPath("model/yolo.h5")
DETECTOR.loadModel()

CUSTOM = DETECTOR.CustomObjects(person=True)


def detect_person(img_array, count: int):
    detections = DETECTOR.detectObjectsFromImage(
        custom_objects=CUSTOM,
        input_image=img_array,
        minimum_percentage_probability=30,
        input_type="array",
        output_type="array"
    )
    print(f"Processed frame: {count}")
    return {count: detections[0]}


if __name__ == "__main__":
    # Reading video
    cap = cv2.VideoCapture("media/input_video/video_2.mp4")

    success, all_frames = get_all_frames(cap)

    print(f"Length of all frames: {len(all_frames)}")

    if success:
        start_time = time.time()

        pool = mp.Pool(mp.cpu_count())

        result = pool.starmap(
            detect_person,
            [(img, count) for count, img in all_frames.items()]
        )

        pool.close()

        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Number of frames in videos: {number_of_frames}")

        print(f"Length of result: {len(result)}")
        # print(result[:2])

        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # if len(result) == count - 1:
        print("Creating video")
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter()
        output_file_name = "media/output_video/parallel_detected_v2.mp4"
        out.open(output_file_name, fourcc, fps, (width, height), True)

        all_images = [result[i].get(i+1) for i in range(len(result))]

        _ = [out.write(image) for image in all_images]

        out.release()

        # Release resources
        cap.release()

        print(f"Time taken: {time.time() - start_time}")
