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

if __name__ == '__main__':

    # load config
    args  = parser.parse_args()
    cfg   = parser.load_config(args)


    ####### Test parameters
    # ---------------------------------------------------------------
    num_classes       = cfg.MODEL.NUM_CLASSES
    clip_length		  = cfg.DATA.NUM_FRAMES
    crop_size 		  = cfg.DATA.TEST_CROP_SIZE
    anchors           = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors       = cfg.SOLVER.NUM_ANCHORS
    nms_thresh        = 0.6
    conf_thresh_valid = 0.4
    image_load_mode = 3

    '''image_path mode 1,2,3 에 대한 설명
    (1). 1: 정상주행을 포함한 주행라벨링 0(정상),1(방향지시등 불이행),2,...,7  -> multilabel_dataset
    (2). 2: 정상주행을 포함하지 않은 주행라벨링 0(방향지시등 불이행), 1, .., 6 (정상을 제외하고 라벨값이 한 인덱스씩 내려옴) -> singlelabel_dataset
    (3). 3: 정상주행을 포함하지 않은 주행 라벨링 0,1,...,n (정상 제외하고 원하는 라벨만 골라서 학습) -> traffic_dataset
    '''
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
        anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
        num_anchors = 5
        conf_thresh_valid = 0.4

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

            # visualize results
            image = cv2.imread(imgpath)
            if image is None:
                print(f"Error: Unable to load image at {imgpath}. Please check the file path and file integrity.")
                continue
            else:
            # 이미지가 제대로 로드되었을 경우에만 진행
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Model inference
            with torch.no_grad():
                data = data.cuda()
                # output.shape -> 1, 60, 7, 7
                output = model(data)
                preds = []
                all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 1, 1)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    #print('boxes:', boxes)
                    boxes = nms(boxes, nms_thresh)
                    
                    for box in boxes:
                        #print(box)
                        x1 = float(box[0]-box[2]/2.0)
                        y1 = float(box[1]-box[3]/2.0)
                        x2 = float(box[0]+box[2]/2.0)
                        y2 = float(box[1]+box[3]/2.0)
                        det_conf = float(box[4])
                        #cls_out = det_conf * box[5].item()
                        preds.append([[x1,y1,x2,y2], det_conf, box[6].item()])

            for dets in preds:
                x1 = int(dets[0][0] * 1280.0)
                y1 = int(dets[0][1] * 720.0)
                x2 = int(dets[0][2] * 1280.0)
                y2 = int(dets[0][3] * 720.0) 
                # det_conf
                cls_scores = dets[1]
                scores = None
                
                if cls_scores > 0.4:
                    scores = cls_scores
                else: continue

                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

                if scores is not None:
                    font  = cv2.FONT_HERSHEY_SIMPLEX
                    coord = []
                    text  = []
                    text_size = []

                    text.append("({:.2f}) ".format(scores) + str(dets[2]))
                    text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                    coord.append((x1-5, y1-5))
                    #cv2.rectangle(image, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                    for t in range(len(text)):
                        cv2.putText(image, text[t], coord[t], font, 0.25, (0, 255, 0), 1)
            
            # 이미지 출력    
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            plt.close('all')  # 모든 figure 닫기

            #이미지 저장
            #try:
            #    os.mkdir(f'results/image/multilabel_train/2nd_multi_sample/{frame_idx[0][:-9]}')
            #except:
            #    pass

            ## 이미지 저장
            #cv2.imwrite(f'results/image/multilabel_train/2nd_multi_sample/{frame_idx[0][:-9]}/{frame_idx[0][:-4]}.png', image)