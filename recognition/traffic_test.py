import os
import torch
import time
from core.utils import *
import logging
from core.region_loss import RegionLoss
from core.model import YOWO
from cfg import parser
from datasets import custom_dataset
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

from collections import OrderedDict

# 훈련된 모델의 결과를 확인하기 위한 스크립트
# ------------------------------------
# path
BASE_PTH= "yowo_dataset_test"
TRAIN_FILE= "yowo_dataset_test/trainlist.txt"
TEST_FILE= "yowo_dataset_test/trainlist.txt"
TEST_VIDEO_FILE= "yowo_dataset_test/trainlist_video.txt"

RESUME_PATH = 'backup/traffic/second_train/yowo_traffic_8f_30epochs_best.pth'
IMAGE_PATH = 'yowo_dataset_test/rgb-images'

args  = parser.parse_args()
cfg   = parser.load_config(args)

epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_dataset = custom_dataset.Traffic_Dataset(BASE_PTH, TEST_FILE, dataset='traffic',
                       shape=(224, 224),
                       transform=transforms.Compose([transforms.ToTensor()]), 
                       train=False, clip_duration=8, sampling_rate=1)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 1, shuffle=False,
                                               num_workers=0, drop_last=False, pin_memory=True)


# datapallel로 훈련 시 각 레이어 앞에 module.state_dict.key가 된다. 따라서 해당 접두사를 제거해야함.
model = YOWO(cfg)
model = model.to(device)
checkpoint = torch.load(RESUME_PATH)
# print(len(checkpoint['state_dict'].keys()))
# print(len(model.state_dict().keys()))


try:
    model.load_state_dict(checkpoint['state_dict'])
    print("DataPallel model load")
except RuntimeError:
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 'module.' 접두사 제거
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    print("Torch model load")

with torch.no_grad():

    def truths_length(truths):
        for i in range(40):
            if truths[i][1] == 0:
                return i

    # Test parameters
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    eps           = 1e-5
    num_classes = 8
    #anchors = [0.070458, 0.118803, 0.126654, 0.255121, 0.159382, 0.408321, 0.230548, 0.494180, 0.352332, 0.591979]
    anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors = 5
    conf_thresh_valid = 0.01
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    fscore = 0.0
    correct_classification = 0.0
    total_detected = 0.0

    nbatch = len(test_loader)
    model.eval()

    for batch_idx, (frame_idx, data, target) in enumerate(test_loader):
        imgpath = os.path.join(IMAGE_PATH,frame_idx[0][-17], frame_idx[0][:-9], frame_idx[0][:-3]+'png')
        print('image_name:', frame_idx[0][:-3])
        #print(os.path.join(IMAGE_PATH,frame_idx[0][-17], frame_idx[0][:-9], frame_idx[0][:-3]+'png'))
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            # all_boxes = np.array(all_boxes)
            # print(all_boxes.shape)
            # [tensor(0.0703), tensor(0.0716), tensor(0.0994), tensor(0.1737), tensor(0.4942), tensor(0.1265), tensor(4),
            #  tensor(0.1245), 0, tensor(0.1229), 1, tensor(0.1256), 2, tensor(0.1242), 3, tensor(0.1250), 5, tensor(0.1254), 6, tensor(0.1259), 7]

            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                # (87, 21) -> boxes shape

            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
            total = total + num_gts
            pred_list = [] # LIST OF CONFIDENT BOX INDICES

            for i in range(len(boxes)):
                # det conf > 0.25
                if boxes[i][4] > 0.25:
                    proposals = proposals+1
                    pred_list.append(i)

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)

                    # proposal 중 가장 iou 값 높은 예측 박스 index
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou

                print(boxes[best_j])

                cls, cx, cy, cw, ch = boxes[best_j][6], boxes[best_j][0], boxes[best_j][1], boxes[best_j][2], boxes[best_j][3]

                x_min = round(float(cx - cw / 2.0) * 1280.0) 
                y_min = round(float(cy - ch / 2.0) * 720.0)
                x_max = round(float(cx + cw / 2.0) * 1280.0)
                y_max = round(float(cy + ch / 2.0) * 720.0)

                # visualize results
                image = cv2.imread(imgpath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 사각형 그리기 (초록색)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                # 클래스 텍스트 표시
                cv2.putText(image, str(cls), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

                # 이미지 출력
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.axis('off')
                plt.show()
                
                if best_iou > iou_thresh:
                    total_detected += 1
                    print(boxes[best_j])
                    # if int(boxes[best_j][6]) == box_gt[6]:
                    #     correct_classification += 1
                    
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            

        
                