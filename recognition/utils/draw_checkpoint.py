from __future__ import print_function
import torch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cfg import parser
from core.utils import *


# ---------------------------------------------------------------
args  = parser.parse_args()
cfg   = parser.load_config(args)

print("===================================================================")
checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
epochs = checkpoint['epoch']
best_score = checkpoint['score']
print(epochs)
print(best_score)
# print("Loaded model score: ", checkpoint['score'])
# print("===================================================================")

# plt.figure(figsize= (10, 5))
# plt.plot(epochs, best_score, label = 'Training score')
# plt.xlabel('Epochs')
# plt.ylabel('Score')
# plt.legend()
# plt.show()