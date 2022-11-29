from typing import Dict, Tuple, Any

import cv2


def vid_to_img(cap, folder: str):
    if not cap.isOpened():
        print("Error opening video stream or file")

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()

    count = 1

    files = list()

    while ret:
        file_name = f"{folder}/frame__{count}.jpg"
        cv2.imwrite(file_name, frame)
        print(f"Saved frame: {count}/{number_of_frames}")

        ret, frame = cap.read()

        count += 1

        files.append(file_name)

    return True, files


def get_all_frames(cap) -> Tuple[bool, Dict[int, Any]]:
    all_frames = dict()

    if not cap.isOpened():
        print("Error opening video stream or file")

    ret, frame = cap.read()

    count = 1

    while ret:
        all_frames[count] = frame
        ret, frame = cap.read()

        count += 1

    return True, all_frames
