import time
import json
import torch
import torch.nn.functional as F
from builtins import range as xrange
import math
import torch.nn as nn
from torch.autograd import Variable
from core.utils import *
import numpy as np
from core.FocalLoss import *

# this function works for building the groud truth 
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh):
    # nH, nW here are number of grids in y and x directions (7, 7 here)
    nB = target.size(0) # batch size
    nA = num_anchors    # 5 for our case
    nC = num_classes
    anchor_step = len(anchors)//num_anchors

    # noobject_scale -> default 1
    # 객체 없는 경우
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale

    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    # for each grid there are nA anchors
    # nAnchors is the number of anchor for one image
    nAnchors = nA*nH*nW
    nPixels  = nH*nW

    # for each image
    for b in xrange(nB):
        # get all anchor boxes in one image
        # (4 * nAnchors)

        # transposed
        # cur_pre_boxes shape is (4,245) if use .t()
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()

        # initialize iou score for each anchor
        cur_ious = torch.zeros(nAnchors)

        #for t in xrange(50):
            # for each anchor 4 coordinate parameters, already in the coordinate system for the whole image
            # this loop is for anchors in each image
            # for each anchor 5 parameters are available (class, x, y, w, h)

        for t in xrange(40):
            # 실제로 라벨링 되어 있지 않은 값 예외처리
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH

            # groud truth boxes
            # (4, 245) if use .t()
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()

            # bbox_ious is the iou value between prediction and groud truth
            # bbox_ious 반환 shape = (1,245)
            # 정답과 예측을 비교했을 때 245개의 iou값 (각 앵커와 정답하나 사이에 대한 값 245개)
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            # print(torch.unique(cur_ious))
            # print(cur_ious.shape)

        # if iou > a given threshold, it is seen as it includes an object
        # sil_thresh = 0.60
        # conf_mask_t shape = (1, 245)
        conf_mask_t = conf_mask.view(nB, -1)

        # IOU 비교 시 예측한 위치에 객체가 있다고 판단하는 경우-> gt와 비교한 iou가 threshold 넘은 경우: 0을 할당
        conf_mask_t[b][cur_ious>sil_thresh] = 0
        conf_mask_tt = conf_mask_t[b].view(nA, nH, nW)

        # conf_mask[b] shape -> (5,7,7)
        conf_mask[b] = conf_mask_tt 
        # conf_mask의 의미: iou가 0.4이상인 box의 mask
        # print(torch.unique(conf_mask))
        # print(torch.sum(conf_mask == 0))
        # print(conf_mask.shape)
    

    # number of ground truth
    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        # anchors for one batch (at least batch size, and for some specific classes, there might exist more than one anchor)
        for t in xrange(40):
            # 객체 없는 경우 break
            if target[b][t*5+1] == 0:
                break

            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            # min_dist = 10000

            # the values saved in target is ratios
            # times by the width and height of the output feature maps nW and nH
            # gx, gy range -> 0 ~ 6
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            
            # grid idx 파악을 위한 gi, gj
            gi = int(gx) 
            gj = int(gy) 

            # if gi >= 7:
            #     gi = 6

            # if gj >= 7:
            #     gj = 6

            # target w / h
            gw = target[b][t*5+3] * nW
            gh = target[b][t*5+4] * nH
            gt_box = [0, 0, gw, gh]

            for n in xrange(nA):
                # get anchor parameters (2 values) anchor step = 2
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                # only consider the size (width and height) of the anchor box
                # cfg 설정된 anchor와 정답 bbox의 iou를 비교
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                # get the best anchor form with the highest iou
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
            #   ㄴ>정답 bbox와 grid별 anchor의 iou값을 계산해서 가장 높은 값을 파악한 결과.
            #print(f'{b}번째 batch에서 {t}번째 정답과 비교했을 때 가장 높은 iou 앵커 인덱스 = {best_n}')

            # then we determine the parameters for an anchor (4 values together)
            gt_box = [gx, gy, gw, gh]

            # find corresponding prediction box
            # pred boxes has (nB*nA*nH*nW, 4) shape tensor
            # 가장 높은 iou를 가진 앵커와 같은 위치의 pred box 값 좌표 추출
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
            #print(pred_box)

            # only consider the best anchor box, for each image
            # 예측 box grid 좌표 mask에 1 할당
            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1

            # in this cell of the output feature map, there exists an object
            # object scale -> 3
            conf_mask[b][best_n][gj][gi] = object_scale
            #print(torch.sum(conf_mask == 3))
            #print(f'conf_mask ====\n {torch.unique(conf_mask)}')
            
            # 중심으로 부터 x, y 좌표 거리
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            
            try:
                tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
                th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            except ValueError:
                print(f"gw: {gw}, anchor: {anchors[anchor_step * best_n]}")


            # confidence equals to iou of the corresponding anchor
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]

            # if ious larger than 0.5, we justify it as a correct prediction
            if iou > 0.5:
                nCorrect = nCorrect + 1

    # true values are returned
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    # for our model anchors has 10 values and number of anchors is 5
    # parameters: 24, 10 float values, 24, 5
    def __init__(self, cfg):
        super(RegionLoss, self).__init__()
        self.num_classes    = cfg.MODEL.NUM_CLASSES
        self.batch          = cfg.TRAIN.BATCH_SIZE
        self.anchors        = [float(i) for i in cfg.SOLVER.ANCHORS]
        self.num_anchors    = cfg.SOLVER.NUM_ANCHORS
        self.anchor_step    = len(self.anchors)//self.num_anchors    # each anchor has 2 parameters
        self.object_scale   = cfg.SOLVER.OBJECT_SCALE
        self.noobject_scale = cfg.SOLVER.NOOBJECT_SCALE
        self.class_scale    = cfg.SOLVER.CLASS_SCALE
        self.coord_scale    = cfg.SOLVER.COORD_SCALE
        self.focalloss      = FocalLoss(class_num=self.num_classes, gamma=2, size_average=False)
        self.thresh = 0.4
        

        # value, count, sum을 이용하여 average를 구함 -> 진행 배치까지의 평균 loss 계산
        self.l_x = AverageMeter()
        self.l_y = AverageMeter()
        self.l_w = AverageMeter()
        self.l_h = AverageMeter()
        self.l_conf = AverageMeter()
        self.l_cls = AverageMeter()
        self.l_total = AverageMeter()

    def reset_meters(self):
        self.l_x.reset()
        self.l_y.reset()
        self.l_w.reset()
        self.l_h.reset()
        self.l_conf.reset()
        self.l_cls.reset()
        self.l_total.reset()


    def forward(self, output, target, epoch, batch_idx, l_loader, train_mode):
        # output : B*A*(4+1+num_classes)*H*W
        # B: number of batches
        # A: number of anchors
        # 4: 4 parameters for each bounding box
        # 1: confidence score
        # num_classes
        # H: height of the image (in grids)
        # W: width of the image (in grids)
        # for each grid cell, there are A*(4+1+num_classes) parameters

        t0 = time.time()

        # output batch size
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        # output h , w
        nH = output.data.size(2)
        nW = output.data.size(3)

        # resize the output (all parameters for each anchor can be reached)
        output   = output.view(nB, nA, (5+nC), nH, nW)
        #print(output.shape)

        # anchor's parameter tx
        # gpu mode => x    = torch.sigmoid(output.index_select(2, torch.cuda.LongTensor([0])).view(nB, nA, nH, nW))
        # cpu mode => x    = torch.sigmoid(output.index_select(2, torch.LongTensor([0])).view(nB, nA, nH, nW))
      
        x    = torch.sigmoid(output.index_select(2, torch.cuda.LongTensor([0])).view(nB, nA, nH, nW))
        # anchor's parameter ty
        y    = torch.sigmoid(output.index_select(2, torch.cuda.LongTensor([1])).view(nB, nA, nH, nW))
        # anchor's parameter tw
        w    = output.index_select(2, torch.cuda.LongTensor([2])).view(nB, nA, nH, nW)
        # anchor's parameter th
        h    = output.index_select(2, torch.cuda.LongTensor([3])).view(nB, nA, nH, nW)
        # confidence score for each anchor
        conf = torch.sigmoid(output.index_select(2, torch.cuda.LongTensor([4])).view(nB, nA, nH, nW))

        # anchor's parameter class label
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        #cls  = output.index_select(2, torch.linspace(5,5+nC-1,nC).long())

        #ouput 3번째 차원(13개)의 5 ~ 11 인덱스에 접근
        output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        
        # resize the data structure so that for every anchor there is a class label in the last dimension
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        # for the prediction of localization of each bounding box, there exist 4 parameters (tx, ty, tw, th)
        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)

        # tx and ty
        # grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
        # grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
        # 각 그리드 시작점 좌표
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()

        # for each anchor there are anchor_step variables (with the structure num_anchor*anchor_step)
        # for each row(anchor), the first variable is anchor's width, second is anchor's height

        # pw and ph
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        # anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0]))
        # anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1]))

        # for each pixel (grid) repeat the above process (obtain width and height of each grid)
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

        # prediction of bounding box localization
        # x.data and y.data: top left corner of the anchor
        # grid_x, grid_y: tx and ty predictions made by yowo
        x_data = x.data.view(-1)
        y_data = y.data.view(-1)
        w_data = w.data.view(-1)
        h_data = h.data.view(-1)
        # x_data = x.view(-1)
        # y_data = y.view(-1)
        # w_data = w.view(-1)
        # h_data = h.view(-1)

        pred_boxes[0] = x_data + grid_x    # bx
        pred_boxes[1] = y_data + grid_y    # by
        pred_boxes[2] = torch.exp(w_data) * anchor_w    # bw
        pred_boxes[3] = torch.exp(h_data) * anchor_h    # bh
        # the size -1 is inferred from other dimensions

        # pred_boxes (nB*nA*nH*nW, 4)
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        # 예측과 정답을 비교하여 반환
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, 
                                                                                                    nH, nW, self.noobject_scale, self.object_scale, self.thresh)
        
        # 이진 마스크 적용 tensor 값 중 1인 원소를 TRUE로 반환 아닐 시 False
        # shape 1, 5, 7, 7 
        cls_mask = (cls_mask == 1)

        #  keep those with high box confidence scores (greater than 0.25) as our final predictions
        # conf shape 8, 5 ,7, 7
        nProposals = int((conf > 0.25).sum().data.item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls = Variable(tcls.view(-1)[cls_mask.view(-1)].long().cuda())
        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  

        # cpu train mode
        # tx    = Variable(tx)
        # ty    = Variable(ty)
        # tw    = Variable(tw)
        # th    = Variable(th)
        # tconf = Variable(tconf)
        # tcls = Variable(tcls.view(-1)[cls_mask.view(-1)].long())
        # coord_mask = Variable(coord_mask)
        # conf_mask  = Variable(conf_mask.sqrt())
        # cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC))
        # cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        # losses between predictions and targets (ground truth)
        # In total 6 aspects are considered as losses: 

        # 4 for bounding box location, 2 for prediction confidence and classification seperately
        loss_x = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.SmoothL1Loss(reduction='sum')(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(reduction='sum')(conf*conf_mask, tconf*conf_mask)/2.0
        # try focal loss with gamma = 2
        #print(f'cls: {cls.shape}')
        #print(f'tcls: {tcls.shape}')
        loss_cls = self.class_scale * self.focalloss(cls, tcls)

        # localization loss
        loss_box = loss_x + loss_y + loss_w + loss_h + loss_conf

        # sum of loss
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        #print(loss)
        t4 = time.time()

        if train_mode:
            self.l_x.update(loss_x.data.item(), self.batch)
            self.l_y.update(loss_y.data.item(), self.batch)
            self.l_w.update(loss_w.data.item(), self.batch)
            self.l_h.update(loss_h.data.item(), self.batch)
            self.l_conf.update(loss_conf.data.item(), self.batch)
            self.l_cls.update(loss_cls.data.item(), self.batch)
            self.l_total.update(loss.data.item(), self.batch)

        # if batch_idx % self.batch == 0: 
        #     print('Epoch: [%d][%d/%d]:\t nGT %d, correct %d, proposals %d, loss: x %.2f(%.2f), '
        #           'y %.2f(%.2f), w %.2f(%.2f), h %.2f(%.2f), conf %.2f(%.2f), '
        #           'cls %.2f(%.2f), total %.2f(%.2f)'
        #            % (epoch, batch_idx, l_loader, nGT, nCorrect, nProposals, self.l_x.val, self.l_x.avg,
        #             self.l_y.val, self.l_y.avg, self.l_w.val, self.l_w.avg,
        #             self.l_h.val, self.l_h.avg, self.l_conf.val, self.l_conf.avg,
        #             self.l_cls.val, self.l_cls.avg, self.l_total.val, self.l_total.avg))

            # step마다 결과 출력(매 배치마다 결과 출력해서 loss가 잘 수렴하는지 확인하기.)
            if batch_idx % 1 == 0:
                print('Epoch: [%d][%d/%d]:\t nGT %d, correct %d, proposals %d, loss: x %.2f(%.2f), '
                    'y %.2f(%.2f), w %.2f(%.2f), h %.2f(%.2f), conf %.2f(%.2f), ''cls %.2f(%.2f), total %.2f(%.2f)'
                    % (epoch, batch_idx, l_loader, nGT, nCorrect, nProposals, self.l_x.val, self.l_x.avg,
                        self.l_y.val, self.l_y.avg, self.l_w.val, self.l_w.avg,
                        self.l_h.val, self.l_h.avg, self.l_conf.val, self.l_conf.avg,
                        self.l_cls.val, self.l_cls.avg, self.l_total.val, self.l_total.avg))
            
        

        else:
            pass
        

        return loss, loss_cls, loss_box









