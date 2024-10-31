# k-means 알고리즘을 활용한 앵커 박스 생성

# k = 5

import numpy as np
from sklearn.cluster import KMeans
import os
import glob
from tqdm import tqdm

def all_bbox_data(label_dir):
    
    all_bbox_list = []

    # 모든 하위 디렉토리에서 .txt 파일을 검색
    txt_files = glob.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)

    # 각 .txt 파일을 열고, 한 줄씩 rstrip()을 적용하여 출력
    for file_path in tqdm(txt_files):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_content = line.rstrip().split()
                wid = float(line_content[3]) - float (line_content[1])
                hei = float(line_content[4]) - float(line_content[2])
                bbox_content = (wid, hei)
                all_bbox_list.append(bbox_content)
                #print(f'box coord:{bbox_content} added')
    return all_bbox_list

def print_anchor(anchor_box):
    width = 1280.0 // 7
    height = 720.0 // 7
    new_anchor_box = []

    for anchor in anchor_box:
        anchor = list(anchor)
        a_w = anchor[0]
        a_h = anchor[1]

        new_w = a_w / width
        new_h = a_h / height
        new_anchor_box.append((new_w, new_h))

    print(new_anchor_box)
    return new_anchor_box


if __name__ == '__main__':
    label_dir = 'multilabel_dataset/labels'
    anchor_sub = all_bbox_data(label_dir)
    anchor_sub = np.array(anchor_sub)

    #kmeans 적용
    kmeans = KMeans(n_clusters=5).fit(anchor_sub)
    anchor_boxes = kmeans.cluster_centers_

    print(f'k-means 결과 후보 앵커 추천:\n{anchor_boxes}')

    # 정규화
    anchor_boxes = print_anchor(anchor_box=anchor_boxes)

