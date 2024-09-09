import cv2
import json
import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

type_ = "test"

def get_length(image_list):
    image_list = [int(str(i).split('.png')[0].split('_')[-1]) for i in image_list]
    length = 0
    for idx, item in enumerate(image_list):
        if idx != (item-1):
            return length
        else:
            length = item

    return length

if __name__ == '__main__':
    for type_ in ['train', 'valid', 'test']:
        for class_type in ['정상/', '비정상/']:
            data_root_dir = Path("../datasets/") / type_ / class_type
            for data_dir in data_root_dir.iterdir():
                if class_type == '정상/':
                    class_num = 0
                else:
                    class_num = int(data_dir.stem)

                video_dirs = list(data_dir.iterdir())

                for video_dir in tqdm.tqdm(video_dirs):
                    image_list = sorted(list(video_dir.glob("**/*.png")))
                    length =get_length(image_list)
                    if length != 0:
                        with open(str(f"{type_}_list_v1.txt"), "a") as f:
                            f.writelines(f"{video_dir} {length} {class_num}\n")
