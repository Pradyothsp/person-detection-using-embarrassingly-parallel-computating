import time

import cv2
from imageai.Detection import ObjectDetection

DETECTOR = ObjectDetection()
DETECTOR.setModelTypeAsYOLOv3()
DETECTOR.setModelPath("model/yolo.h5")
DETECTOR.loadModel()

CUSTOM = DETECTOR.CustomObjects(person=True)


def detect_person(img_array):
    detections = DETECTOR.detectObjectsFromImage(
        custom_objects=CUSTOM,
        input_image=img_array,
        minimum_percentage_probability=30,
        input_type="array",
        output_type="array"
    )
    return detections


if __name__ == "__main__":
    # Reading video
    cap = cv2.VideoCapture("media/input_video/video_2.mp4")

    if not cap.isOpened():
        print("Error opening video stream or file")

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"FPS: {fps}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter()
    output_file_name = "media/output_video/serial_detected.mp4"
    out.open(output_file_name, fourcc, fps, (width, height), True)

    ret, frame = cap.read()

    count = 1
    start_time = time.time()

    while ret:
        detection_result = detect_person(frame)
        out.write(detection_result[0])

        print(f"Frame: {count}/{number_of_frames}")

        ret, frame = cap.read()

        count += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Time taken: {time.time() - start_time}")
