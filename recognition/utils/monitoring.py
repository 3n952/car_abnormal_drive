from __future__ import print_function
import torch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils import *


checkpoint_path = '/Users/QBIC/Desktop/workspace/car_abnormal_driving/recognition/backup/traffic/second_test/yowo_traffic_8f30epochs_last.pth'

checkpoint = torch.load(checkpoint_path)
epochs = checkpoint['epoch']
best_score = checkpoint['score']
losses = checkpoint['loss']
cls_loss = checkpoint['cls_loss']
box_loss = checkpoint['box_loss']

print("===================================================================")
print("total epochs: ", checkpoint['epoch'])
print("Loaded model best score: ", max(checkpoint['score']))

fig, axs = plt.subplots(1, 3, figsize=(10, 5))

axs[0, 0].plot(losses, 'r')
axs[0, 0].set_xlabel('Total Loss')
axs[0, 0].set_ylabel('Epochs')
axs[0, 0].set_title('train total loss')

# 두 번째 plot (1행 2열)
axs[0, 1].plot(cls_loss, 'g')
axs[0, 0].set_xlabel('CLS Loss')
axs[0, 0].set_ylabel('Epochs')
axs[0, 1].set_title('train classification loss')

# 세 번째 plot (2행 1열)
axs[0, 2].plot(box_loss, 'b')
axs[0, 0].set_xlabel('LCZ Loss')
axs[0, 0].set_ylabel('Epochs')
axs[0, 2].set_title('train localization loss')

plt.tight_layout()
plt.show()
