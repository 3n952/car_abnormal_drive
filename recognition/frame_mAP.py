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
#from tqdm import tqdm
from collections import OrderedDict

'''
validation의 정성 평가를 하기 위한 이미지 단위(key frame)의 추론 결과 시각화 코드
'''

RESUME_PATH = 'backup/traffic/train1/yowo_traffic_10f_10epochs_best.pth'

if __name__ == '__main__':

    # load config
    args  = parser.parse_args()
    cfg   = parser.load_config(args)

    # image_path mode 1,2,3 ...
    # (1). 1: 정상주행을 포함한 주행라벨링 0(정상),1(방향지시등 불이행),2,...,7 
    # (2). 2: 정상주행을 포함하지 않은 주행라벨링 0(방향지시등 불이행), 1, .., 6 (정상을 제외하고 라벨값이 한 인덱스씩 내려옴)
    # (3). 3: 정상주행을 포함하지 않은 주행 라벨링 0,1,...,n (정상 제외하고 원하는 라벨만 골라서 학습)

    image_load_mode = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = custom_dataset.Traffic_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TEST_FILE, dataset='traffic',
                        shape=(224, 224),
                        transform=transforms.Compose([transforms.ToTensor()]), 
                        train=False, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 1, shuffle=False,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)

    # datapallel로 훈련 시 각 레이어 앞에 module.state_dict.key가 된다. 따라서 해당 접두사를 제거해야함.
    model = YOWO(cfg)
    model = model.cuda()
    checkpoint = torch.load(RESUME_PATH, map_location = device)

    try:
        model.load_state_dict(checkpoint['state_dict'])
        print('===========================================================')
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
        
    del checkpoint

    with torch.no_grad():
        def truths_length(truths):
            for i in range(40):
                if truths[i][1] == 0:
                    return i
        # Test parameters
        nms_thresh    = 0.5
        iou_thresh    = 0.5
        eps           = 1e-5
        anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
        num_anchors = 5
        conf_thresh_valid = 0.005
        proposals = 0.0

        nbatch = len(test_loader)
        model.eval()

        for batch_idx, (frame_idx, data, target) in enumerate(test_loader):
            if image_load_mode == 1:
                if frame_idx[0][-19] == 'a':
                    imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images',frame_idx[0][-17], frame_idx[0][:-9], frame_idx[0][:-3]+'png')
                else:
                    imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images', '0', frame_idx[0][:-9], frame_idx[0][:-3]+'png')

            elif image_load_mode == 2:
                imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images',str(int(frame_idx[0][-17])-1), frame_idx[0][:-9], frame_idx[0][:-3]+'png')
            
            elif image_load_mode == 3:
                #직접설정 -> str(int(frame_idx[0][-17])-1 수정
                if frame_idx[0][-17] == '2':
                    imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images',str(int(frame_idx[0][-17])-2), frame_idx[0][:-9], frame_idx[0][:-3]+'png')
                elif frame_idx[0][-17] == '3':
                    imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images',str(int(frame_idx[0][-17])-2), frame_idx[0][:-9], frame_idx[0][:-3]+'png')
                elif frame_idx[0][-17] == '4':
                    imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images',str(int(frame_idx[0][-17])-2), frame_idx[0][:-9], frame_idx[0][:-3]+'png')
                elif frame_idx[0][-17] == '5':
                    imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images',str(int(frame_idx[0][-17])-2), frame_idx[0][:-9], frame_idx[0][:-3]+'png')
                elif frame_idx[0][-17] == '7':
                    imgpath = os.path.join(cfg.LISTDATA.BASE_PTH+'/rgb-images',str(int(frame_idx[0][-17])-3), frame_idx[0][:-9], frame_idx[0][:-3]+'png')
                else:
                    print('train label is wrong')

                #etc
            else:
                print('make sure to organizing data load mode. there is no data load mode')               

            print(f'---총 {nbatch}개의 이미지 중 {batch_idx + 1}번째 이미지---')
            print(f'image_name: {frame_idx[0][:-4]}')
            # input data put on gpu
            data = data.cuda()
            with torch.no_grad():
                output = model(data)
                all_boxes = get_region_boxes(output, conf_thresh_valid, cfg.MODEL.NUM_CLASSES, anchors, num_anchors, 1, 1)

                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)

                truths = target[i].view(-1, 5)

                # total : 전체 sample에 대한 gt값 
                num_gts = truths_length(truths)
                pred_list = [] # LIST OF CONFIDENT BOX INDICES

                for i in range(len(boxes)):
                    if boxes[i][4] > 0.25:
                        proposals = proposals+1
                        pred_list.append(i)

                # visualize results
                image = cv2.imread(imgpath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    best_j = 0
                    for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        # proposal 중 가장 iou 값 높은 예측 박스 index
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou
            
                    cls, cx, cy, cw, ch = boxes[best_j][6], boxes[best_j][0], boxes[best_j][1], boxes[best_j][2], boxes[best_j][3]
                    # cx, cy 
                    tx, ty, tw, th = float(truths[i][1]), float(truths[i][2]), float(truths[i][3]), float(truths[i][4])

                    x_min = round(float(cx - cw / 2.0) * 1280.0) 
                    y_min = round(float(cy - ch / 2.0) * 720.0)
                    x_max = round(float(cx + cw / 2.0) * 1280.0)
                    y_max = round(float(cy + ch / 2.0) * 720.0)

                    tx_min = round(float(tx - tw / 2.0) * 1280.0) 
                    ty_min = round(float(ty - th / 2.0) * 720.0)
                    tx_max = round(float(tx + tw / 2.0) * 1280.0)
                    ty_max = round(float(ty + th / 2.0) * 720.0)

                    # bbox 시각화
                    # 비정상내에서만 추출하는 경우
                    if image_load_mode >= 2:
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.rectangle(image, (tx_min, ty_min), (tx_max, ty_max), (0, 255, 0), 2)
                        cv2.putText(image, str(cls.item()), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        #cv2.putText(image, 'GT', (tx_min, ty_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # 정상/비정상 분류해서 시각화하는 경우
                    else:
                        if int(cls.item()) == 0:
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(image, str(cls.item()), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.putText(image, str(cls.item()), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # 이미지 출력    
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.axis('off')
                plt.show()
                plt.close('all')  # 모든 figure 닫기

                # 이미지 저장
                #cv2.imwrite(f'assets/inference/train2/{frame_idx[0][:-4]}.png', image)
            
                