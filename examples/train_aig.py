from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate_aig
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# DATA_DIR = '/home/wjx/Xmg_gate/Xmg_gate/examples/data/train'
DATA_DIR = '/home/jwt/1/aig_npz'

# DATA_DIR = '/home/wjx/python-deepgate/examples/data/train'
if __name__ == '__main__':
    circuit_path = os.path.join(DATA_DIR, 'graphs.npz')
    label_path = os.path.join(DATA_DIR, 'graphs.npz')
    num_epochs = 60
    
    print('[INFO] Parse Dataset')
    dataset = deepgate_aig.NpzParser(DATA_DIR, circuit_path, label_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print("train_dataset =", train_dataset)
    print('[INFO] Create Model and Trainer')
    #model = deepgate.Model() #other model
    model = deepgate_aig.Aig_Model() #model for aig

    trainer = deepgate_aig.Trainer(model, distributed=True)
    trainer.set_training_args(prob_rc_func_weight=[1.0, 0.0, 0.0], lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    print('[INFO] Stage 2 Training ...')
    trainer.set_training_args(prob_rc_func_weight=[3.0, 0.0, 2.0], lr=1e-4, lr_step=50)
    trainer.train(num_epochs, train_dataset, val_dataset)
    