from __future__ import print_function
import torch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils import *


checkpoint_path = 'backup/traffic/third_train/yowo_traffic_8f_16epochs_best.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(checkpoint_path, map_location =device)

epochs = checkpoint['epoch']
print("===================================================================")
print("total epochs: ", epochs)
print("Loaded model best score: ", checkpoint['best_score'])

train_losses = checkpoint['train_loss']
val_losses = checkpoint['val_loss']

try:
    train_cls_loss = checkpoint['train_cls_loss']
    train_box_loss = checkpoint['train_box_loss']
    val_cls_loss = checkpoint['val_cls_loss']
    val_box_loss = checkpoint['val_box_loss']


    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    # training loss
    axs[0,0].plot(train_losses, 'r')
    axs[0,0].set_ylabel('Total train Loss')
    axs[0,0].set_xlabel('Epochs')
    axs[0,0].set_title('train total loss')

    axs[0,1].plot(train_cls_loss, 'g')
    axs[0,1].set_ylabel('CLS train Loss')
    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_title('train classification loss')

    axs[0,2].plot(train_box_loss, 'b')
    axs[0,2].set_ylabel('LCZ train Loss')
    axs[0,2].set_xlabel('Epochs')
    axs[0,2].set_title('train localization loss')
    
    ### validation loss
    axs[1,0].plot(val_losses, 'r')
    axs[1,0].set_ylabel('Total validation Loss')
    axs[1,0].set_xlabel('Epochs')
    axs[1,0].set_title('validation total loss')

    axs[1,1].plot(val_cls_loss, 'g')
    axs[1,1].set_ylabel('CLS validation Loss')
    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_title('validation classification loss')

    axs[1,2].plot(val_box_loss, 'b')
    axs[1,2].set_ylabel('LCZ validation Loss')
    axs[1,2].set_xlabel('Epochs')
    axs[1,2].set_title('validation localization loss')

except KeyError:
    losses = [loss.detach().numpy() for loss in train_losses]
    plt.plot(losses, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('train total loss')

plt.tight_layout()
plt.show()
