import os
import torch
import time
from core.utils import *



# custom(traffic)-----------------------

def train_traffic(cfg, epoch, model, train_loader, loss_module, optimizer):
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # region loss

    #loss 초기화
    loss_module.reset_meters()
    l_loader = len(train_loader)

    # training loop
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        output = model(data)
        loss, loss_cls, loss_box = loss_module(output, target, epoch, batch_idx, l_loader)

        loss.backward()
        # total frame 수 // batchsize -> batch 개수 = step
        steps = cfg.TRAIN.TOTAL_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE

        if batch_idx % steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # save result every 998 batches
        # if batch_idx % 998 == 0: # From time to time, reset averagemeters to see improvements
        #     loss_module.reset_meters()

    t1 = time.time()
    logging('trained with %f samples in %d seconds' % (len(train_loader.dataset), (t1-t0)))
    print('')
    
    return loss, loss_cls , loss_box


@torch.no_grad()
def test_traffic(cfg, epoch, model, test_loader):

    def truths_length(truths):
        for i in range(40):
            # zero 초기화 했기 때문에, 0인 값이 탐색시 해당 인덱스전까지가 정답이 라벨링된 값이다. ex 3개가 라벨링 되어있다면, 3 인덱스에서 0이 발견됨.
            if truths[i][1] == 0:
                return i

    # Test parameters
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    eps           = 1e-5
    num_classes = cfg.MODEL.NUM_CLASSES
    anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors = cfg.SOLVER.NUM_ANCHORS
    conf_thresh_valid = 0.01
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    fscore = 0.0

    correct_classification = 0.0
    total_detected = 0.0

    nbatch = len(test_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    for batch_idx, (frame_idx, data, target) in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            #output = model(data).data
            output = model(data)

            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)

            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)

                # nms 결과 box 반환 및 저장
                if cfg.TRAIN.DATASET == 'traffic' :
                    detection_path = os.path.join('custom_detections', 'detections_'+str(epoch), frame_idx[i])
                    current_dir = os.path.join('custom_detections', 'detections_'+str(epoch))
                    if not os.path.exists('custom_detections'):
                        os.mkdir('custom_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)
                else:
                    print('wrong directory')

                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        # 1280 w, 720 h은 해상도에 맞게 조정
                        x1 = round(float(box[0]-box[2]/2.0) * 1280.0)
                        y1 = round(float(box[1]-box[3]/2.0) * 720.0)
                        x2 = round(float(box[0]+box[2]/2.0) * 1280.0)
                        y2 = round(float(box[1]+box[3]/2.0) * 720.0)
                        det_conf = float(box[4])

                        # for j in range((len(box)-5)//2):
                        #     cls_conf = float(box[5+2*j].item())
                        #     prob = det_conf * cls_conf
                        #     f_detect.write(str(int(box[6])) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

                        cls_conf = float(box[5].item())
                        prob = det_conf * cls_conf
                        
                        # 후보 anchor 중에 conf thres 넘는 값이면 추가함
                        f_detect.write(str(int(box[6])) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
                
                
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
                        #iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=True)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou

                    if best_iou > iou_thresh:
                        total_detected += 1
                        if int(boxes[best_j][6]) == box_gt[6]:
                            correct_classification += 1

                    if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                        correct = correct+1

            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)
            logging("[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))

    classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
    localization_recall = 1.0 * total_detected / (total + eps)

    print("Classification accuracy: %.3f" % classification_accuracy)
    print("Localization recall: %.3f" % localization_recall)

    return fscore