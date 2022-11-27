from uuid import uuid4

import cv2
from typing import Dict


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


def get_all_frames(cap) -> Dict:
    all_frames = dict()

    if not cap.isOpened():
        print("Error opening video stream or file")

    ret, frame = cap.read()

    count = 1

    while ret:
        all_frames[count] = frame
        ret, frame = cap.read()

        count += 1

    return all_frames
