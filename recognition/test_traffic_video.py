import cv2
import numpy as np
from collections import OrderedDict
import torch
from core.optimization import *
from cfg import parser
from core.utils import *
from core.model import YOWO

####### Load configuration arguments
# ---------------------------------------------------------------
args  = parser.parse_args()
cfg   = parser.load_config(args)

####### Create model
# ---------------------------------------------------------------
model = YOWO(cfg)
model = model.cuda()
# model = nn.DataParallel(model, device_ids=None) # in multi-gpu case

####### Load resume path if necessary
# ---------------------------------------------------------------
if cfg.TRAIN.RESUME_PATH:
    print("===================================================================")
    print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
    checkpoint = torch.load(cfg.TRAIN.RESUME_PATH, map_location = 'cuda')
    
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded model f-score: ", checkpoint['best_score'])
        print("===================================================================")
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 'module.' 접두사 제거
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

    del checkpoint

####### Test parameters
# ---------------------------------------------------------------
num_classes       = cfg.MODEL.NUM_CLASSES
clip_length		  = cfg.DATA.NUM_FRAMES
crop_size 		  = cfg.DATA.TEST_CROP_SIZE
anchors           = [float(i) for i in cfg.SOLVER.ANCHORS]
num_anchors       = cfg.SOLVER.NUM_ANCHORS
nms_thresh        = 0.5
conf_thresh_valid = 0.1

model.eval()

####### Data preparation and inference 
# ---------------------------------------------------------------

# 입력 영상 불러오기
video_path = 'D:/singlelabel_dataset/video/p01_20230108_123006_an7_347_04.mp4'
video_split = video_path.split('/')
print('video inference starting...')
#img_mean, img_std = calculate_mean_std(video_path)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'this video has {fps} fps')

# 비디오 저장
output_video_path = 'examples/traffic_dataset/'+video_split[3]
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps-5, (frame_width, frame_height))

queue = []
while(cap.isOpened()):
    # frame h, w, c 순
    ret, frame_og = cap.read()

    # 빈 frame 예외처리
    if not ret:
        break

    # queue가 빈 경우 초기 프레임으로 대체
    if len(queue) <= 0: # At initialization, populate queue with initial frame
    	for i in range(clip_length):
            queue.append(frame_og)

    # 새 프레임 추가 및 오래된 프레임 제거
    queue.append(frame_og)
    queue.pop(0)

    # ================case 1===============================
    # resize
    imgs = [cv2.resize(img, (crop_size, crop_size), interpolation= cv2.INTER_LINEAR) for img in queue]
    
    # 이미지를 255.0으로 나누어 [0, 1] 범위로 정규화
    imgs = [img / 255.0 for img in imgs]
    
    # numpy 배열을 PyTorch 텐서 형태처럼 (C, H, W)로 변환
    # 현재 이미지 형태는 (H, W, C)이므로 이를 (C, H, W)로 변환
    imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

    # 3, concat dim, 224, 224 (+ numpy 변환)
    imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)
    imgs = imgs.astype(np.float32)
    imgs = np.ascontiguousarray(imgs)

    # tensor 변환
    imgs = torch.from_numpy(imgs)
    imgs = torch.unsqueeze(imgs, 0)

    # img to gpu
    # clip 단위
    imgs_gpu = imgs.cuda()

    # Model inference
    with torch.no_grad():
        
        # output.shape -> 1, 60, 7, 7
        output = model(imgs_gpu)
        preds = []
        all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
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
        if cls_scores > 0.1:
            scores = cls_scores
        else: continue

        cv2.rectangle(frame_og, (x1,y1), (x2,y2), (0,255,0), 2)
        if scores is not None:
            blk   = np.zeros(frame_og.shape, np.uint8)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text  = []
            text_size = []

            text.append("({:.2f}) ".format(scores) + str(dets[2]))
            text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
            coord.append((x1-3, y1-10))
            cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
            frame_og = cv2.addWeighted(frame_og, 1.0, blk, 0.25, 1)
            for t in range(len(text)):
                cv2.putText(frame_og, text[t], coord[t], font, 0.25, (0, 0, 0), 1)

    out.write(frame_og)
    cv2.imshow('frame',frame_og)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
