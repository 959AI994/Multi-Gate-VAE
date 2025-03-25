import os
import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='Pytorch training script of DG_VAE.')
    parser.add_argument('--exp_id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    
    # Model
    parser.add_argument('--model', type=str, default='DG_VAE', help='Model name', choices=[
        'DG_VAE', 'DG_AE', 'AE'
    ])
    parser.add_argument('--dim_hidden', type=int, default=64, help='Dimension of hidden layer')
    parser.add_argument('--dim_feature', type=int, default=6, help='Dimension of input feature')
    parser.add_argument('--s_rounds', type=int, default=4, help='Number of rounds for source node')
    parser.add_argument('--t_rounds', type=int, default=4, help='Number of rounds for target node')
    parser.add_argument('--layernorm', action='store_true', help='Enable layernorm')
    
    # Training
    # parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument('--type', type=str, required=True, choices=['aig', 'mig', 'xmg', 'xag'],help='Circuit type to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')    
    parser.add_argument('--resume', action='store_true')    
    # parser.add_argument("--local-rank", default=0, type=int)

    args = parser.parse_args()
    return args