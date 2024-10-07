from __future__ import print_function
import torch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils import *


checkpoint_path = 'backup/traffic/fourth_train/yowo_traffic_8f_100epochs_last.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(checkpoint_path, map_location =device)

epochs = checkpoint['epoch']
print("===================================================================")
print("total epochs: ", epochs)
print("Loaded model best score: ", max(checkpoint['score']))

best_score = checkpoint['score']
losses = checkpoint['loss']

try:
    cls_loss = checkpoint['cls_loss']
    box_loss = checkpoint['box_loss']

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(losses, 'r')
    axs[0].set_ylabel('Total Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_title('train total loss')

    # 두 번째 plot (1행 2열)
    axs[1].plot(cls_loss, 'g')
    axs[1].set_ylabel('CLS Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_title('train classification loss')

    # 세 번째 plot (2행 1열)
    axs[2].plot(box_loss, 'b')
    axs[2].set_ylabel('LCZ Loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_title('train localization loss')

except KeyError:
    losses = [loss.detach().numpy() for loss in losses]
    plt.plot(losses, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('train total loss')

plt.tight_layout()
plt.show()
