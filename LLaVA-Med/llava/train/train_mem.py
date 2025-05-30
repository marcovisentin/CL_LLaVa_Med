import os
import sys
import inspect
sys.path.append('/vol/biomedic3/mv320/projects/VLMs/MEG_x_CL/LLaVA-Med')
from llava.train.train import train
from llava.train.train_simpler import train as train_simpler

if __name__ == "__main__":
    
    train_simpler(attn_implementation="flash_attention_2")