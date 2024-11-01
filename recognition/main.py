from __future__ import print_function
import os
import sys
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from datasets import custom_dataset 
from core.optimization import *
from cfg import parser
from core.utils import *
from core.region_loss import RegionLoss
from core.model import YOWO, get_fine_tuning_parameters


####### Load configuration arguments
# ---------------------------------------------------------------
args  = parser.parse_args()
cfg   = parser.load_config(args)


####### Check backup directory, create if necess    ary
# ---------------------------------------------------------------
if not os.path.exists(cfg.BACKUP_DIR):
    os.makedirs(cfg.BACKUP_DIR)

# train 학습 모니터링을 위한 log 세분화
# backup_dir = os.path.join(cfg.BACKUP_DIR, str(cfg.DATA.NUM_FRAMES)+'f_'+str(cfg.TRAIN.END_EPOCH))
# if not os.path.exists(backup_dir):
#     os.makedirs(backup_dir)

if __name__ =='__main__':
    ####### Create model
    # ---------------------------------------------------------------
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOWO(cfg)
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None) # in multi-gpu case
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging('Total number of trainable parameters: {}'.format(pytorch_total_params))

    seed = int(time.time())
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO: add to config e.g. 0,1,2,3
        torch.cuda.manual_seed(seed)


    ####### Create optimizer
    # ---------------------------------------------------------------
    parameters = get_fine_tuning_parameters(model, cfg)
    optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    best_score   = 0 # initialize best score
    # optimizer = optim.SGD(parameters, lr=cfg.TRAIN.LEARNING_RATE/batch_size, momentum=cfg.SOLVER.MOMENTUM, dampening=0, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


    ####### Load resume path if necessary
    # ---------------------------------------------------------------
    if cfg.TRAIN.RESUME_PATH:
        print("===================================================================")
        print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
        checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
        cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch'] + 1
        best_score = checkpoint['score']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded model score: ", checkpoint['score'])
        print("===================================================================")
        del checkpoint
    else:
        pass


    ####### Create backup directory if necessary
    # ---------------------------------------------------------------
    if not os.path.exists(cfg.BACKUP_DIR):
        os.mkdir(cfg.BACKUP_DIR)


    ####### Data loader, training scheme and loss function 
    # ---------------------------------------------------------------
    # multithread core set
    torch.set_num_threads(cfg.DATA_LOADER.NUM_WORKERS)

    # dataset loader
    dataset = cfg.TRAIN.DATASET
    assert dataset == 'traffic', 'invalid dataset'

    train_dataset = custom_dataset.Traffic_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TRAIN_FILE, dataset=dataset,
                                                shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                                                transform=transforms.Compose([transforms.ToTensor()]), 
                                                train=True, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)

    test_dataset  = custom_dataset.Traffic_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TEST_FILE, dataset=dataset,
                        shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                        transform=transforms.Compose([transforms.ToTensor()]), 
                        train=False, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)

    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)

    loss_module   = RegionLoss(cfg).cuda()

    # 임포트 모듈 중 train_traffic, test_traffic 가져오기 opimization.py참고
    train = getattr(sys.modules[__name__], 'train_traffic')
    test  = getattr(sys.modules[__name__], 'test_traffic')


    ####### Training and Testing Schedule
    # ---------------------------------------------------------------
    if cfg.TRAIN.EVALUATE:
        logging('evaluating ...')
        test(cfg, 0, model, test_loader)
    else:
        train_total_loss_list = []
        train_loss_cls_list = []
        train_loss_box_list = []

        val_total_loss_list = []
        val_loss_cls_list = []
        val_loss_box_list = []

        score_list = []
        best_score = 0
        #best_loss = 99999

        for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH + 1):
            # Adjust learning rate
            lr_new = adjust_learning_rate(optimizer, epoch, cfg)

            # Train and test yowo model
            ############ training
            logging('training at epoch %d, lr %f' % (epoch, lr_new))
            train_loss, train_loss_cls, train_loss_box = train(cfg, epoch, model, train_loader, loss_module, optimizer)
            #print(train_loss, train_loss_cls, train_loss_box)

            # reload to cpu
            train_loss = convert2cpu(train_loss)
            train_loss_cls = convert2cpu(train_loss_cls)
            train_loss_box = convert2cpu(train_loss_box)

            # loss list 
            train_total_loss_list.append(train_loss.item())
            train_loss_cls_list.append(train_loss_cls.item())
            train_loss_box_list.append(train_loss_box.item())

            ############ validation
            logging('validating at epoch %d' % (epoch))
            val_loss, val_loss_cls , val_loss_box, fscore = test(cfg, epoch, model, test_loader, loss_module)

            # reload to cpu
            val_loss = convert2cpu(val_loss)
            val_loss_cls = convert2cpu(val_loss_cls)
            val_loss_box = convert2cpu(val_loss_box)

            val_total_loss_list.append(val_loss.item())
            val_loss_cls_list.append(val_loss_cls.item())
            val_loss_box_list.append(val_loss_box.item())

            # score_list.append(fscore)
            # fscore_average = sum(score_list) / len(score_list)

            # Save the model to backup directory
            is_best = fscore > best_score
           
            
            if is_best:
                print('New best score is achieved:', fscore)
                print("Previous score was: ", best_score)
                best_score = fscore

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_score,
                'train_loss': train_total_loss_list,
                'train_cls_loss': train_loss_cls_list,
                'train_box_loss': train_loss_box_list,
                'val_loss': val_total_loss_list,
                'val_cls_loss': val_loss_cls_list,
                'val_box_loss': val_loss_box_list
                }

            save_checkpoint(state, is_best, epoch, cfg.BACKUP_DIR, cfg.TRAIN.DATASET, cfg.DATA.NUM_FRAMES)
            #save_checkpoint2(state, is_best, epoch,cfg.TRAIN.END_EPOCH, cfg.BACKUP_DIR, cfg.TRAIN.DATASET, cfg.DATA.NUM_FRAMES)
            logging('Weights are saved to backup directory: %s' % (cfg.BACKUP_DIR))
