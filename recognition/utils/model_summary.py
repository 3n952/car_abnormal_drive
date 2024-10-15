from torchinfo import summary
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cfg import parser
from core.model import YOWO

args  = parser.parse_args()
cfg   = parser.load_config(args)
model = YOWO(cfg)

# (batch 8, 3channel, 8 clip dur, w, h) size of input tensor
summary(model, input_size=(8, 3, 8, 224, 224))