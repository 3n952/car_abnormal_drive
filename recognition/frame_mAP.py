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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

'''
frame mAp 분석
'''

if __name__ == '__main__':
    print('starting measure frame mAP....')

    # load config
    args  = parser.parse_args()
    cfg   = parser.load_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = custom_dataset.Traffic_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TEST_FILE, dataset='traffic',
                        shape=(224, 224),
                        transform=transforms.Compose([transforms.ToTensor()]), 
                        train=False, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)

    # datapallel로 훈련 시 각 레이어 앞에 module.state_dict.key가 된다. 따라서 해당 접두사를 제거해야함.
    model = YOWO(cfg)
    model = model.cuda()
    checkpoint = torch.load(cfg.TRAIN.RESUME_PATH, map_location = device)

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
        nms_thresh    = 0.6
        iou_thresh    = 0.5
        eps           = 1e-5
        num_classes = cfg.MODEL.NUM_CLASSES
        anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
        num_anchors = cfg.SOLVER.NUM_ANCHORS
        conf_thresh_valid = 0.4

        total       = 0.0
        proposals   = 0.0
        tp = 0
        fp = 0
        fn = 0

        nbatch = len(test_loader)
        model.eval()

        for batch_idx, (frame_idx, data, target) in enumerate(test_loader):        
            # input data put on gpu
            data = data.cuda()
            with torch.no_grad():
                output = model(data)
                all_boxes = get_region_boxes(output, conf_thresh_valid, cfg.MODEL.NUM_CLASSES, anchors, num_anchors, 1, 1)

                for i in range(output.size(0)):
                    # boxes = [x, y, w, h, det_conf, cls_conf, cls_max_id]
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)

                    truths = target[i].view(-1, 5)
                    num_gts = truths_length(truths)
            
                    total = total + num_gts
                    pred_list = [] # LIST OF CONFIDENT BOX INDICES
                    for i in range(len(boxes)):
                        # conf_thresh = 0.4
                        if boxes[i][4] > 0.4:
                            proposals = proposals+1
                            pred_list.append(i)

                    for i in range(num_gts):
                        box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                        best_iou = 0
                        best_j = -1
                        for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                            iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                            if iou > best_iou:
                                best_j = j
                                best_iou = iou

                        if best_iou >= iou_thresh:
                            if int(boxes[best_j][6]) == box_gt[6]:
                                # true positive
                                tp += 1
                            else:
                                # False positive 
                                fp += 1
                            
                        else:
                            fp += 1
                        
                    fn = fn + (num_gts - tp)

                precision = 1.0 * tp / (tp + fp)
                recall = 1.0 * tp / (tp + fn) 

                # fscore 
                fscore = 2.0*precision*recall/ (precision+recall+eps)
                logging("[총 %d 개의 batch 중 %d 번째 batch 까지의]\t precision: %f, recall: %f, fscore: %f" % (nbatch, batch_idx+1, precision, recall, fscore))

        print()
        print(f'recall for total test dataset: {recall}')
        print(f'precision for total test dataset: {precision}')
        print(f'f-score for total test dataset: {fscore}')
        