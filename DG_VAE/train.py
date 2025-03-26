from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate
import os
from config import get_parse_args

import deepgate.digae_layer
import deepgate.digae_model
import deepgate.digvae_model
import deepgate.dg_ae_model_aig
import deepgate.dg_ae_model_mig
import deepgate.dg_ae_model_xag
import deepgate.dg_ae_model_xmg
import torch
import torch.distributed as dist

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':

    args = get_parse_args()

    DATA_DIR = f'/home/xqgrp/wangjingxin/datasets/mixgate_data/4npz_1500/{args.type}_npz/'

    circuit_path = os.path.join(DATA_DIR, 'graphs.npz')
    # label_path = os.path.join(DATA_DIR, 'graphs.npz')
    label_filename = 'graphs.npz' if args.type == 'aig' else 'labels.npz'
    label_path = os.path.join(DATA_DIR, label_filename)
    
    model_map = {
        'aig': deepgate.dg_ae_model_aig.Model,
        'mig': deepgate.dg_ae_model_mig.Model,
        'xmg': deepgate.dg_ae_model_xmg.Model,
        'xag': deepgate.dg_ae_model_xag.Model
    }

    print('[INFO] Parse Dataset')
    dataset = deepgate.NpzParser(DATA_DIR, circuit_path, label_path,args.type)
    train_dataset, val_dataset = dataset.get_dataset()
    
    print('[INFO] Create Model')
    if 'DG' in args.model:
        encoder = deepgate.digae_layer.DirectMultiGCNEncoder(   # 用这个作为encoder
            dim_hidden=args.dim_hidden, dim_feature=args.dim_feature, enable_reverse=True, 
            s_rounds=args.s_rounds, t_rounds=args.t_rounds, 
            layernorm=args.layernorm
        )
    else:
        encoder = deepgate.digae_layer.DirectedGCNConvEncoder(
            in_channels=3, hidden_channels=64, out_channels=64,
            alpha=1.0, beta=0.0, self_loops=True, adaptive=False
        )
    decoder = deepgate.digae_layer.DirectedInnerProductDecoder()  # 用这个作为decoder

    if 'VAE' in args.model:
        model = deepgate.digvae_model.DirectedGVAE(encoder, args.dim_hidden, decoder)
    else:
        # model = deepgate.digae_model.DirectedGAE(encoder, decoder)  #用这个作为整体的model
        model_class = model_map[args.type]
        model = model_class(
            struct_encoder=encoder,
            # num_rounds=args.num_rounds,
            dim_hidden=args.dim_hidden,
            enable_encode=True,
            enable_reverse=True
        )

    trainer = deepgate.Trainer(
        args, model, 
        training_id = args.exp_id, batch_size=args.batch_size, 
        distributed=True
        # patience=10,  # 容忍10个epoch没有改善
        # delta=0.0002    # 损失改善需大于0.001
    )
    if args.resume:
        trainer.resume()
    
    # 三阶段训练配置
    stage_configs = [
        {'epochs': 100, 'weights': [1.0, 0.0, 0.0], 'lr': 1e-4},  # 阶段1: 仅rc_loss
        {'epochs': 60, 'weights': [1.0, 5.0, 0.0], 'lr': 1e-4},  # 阶段2: rc_loss + prob_loss
        {'epochs': 60, 'weights': [1.0, 4.0, 4.0], 'lr': 1e-4}   # 阶段3: 全损失
    ]

    for stage_idx, config in enumerate(stage_configs):
        print(f'\n{"="*40}')
        print(f'[STAGE {stage_idx+1}] Start Training')
        print(f'|-- Epochs: {config["epochs"]}')
        print(f'|-- Loss Weights: {config["weights"]}')
        print(f'|-- Learning Rate: {config["lr"]}')
        
        # 设置当前阶段参数
        trainer.set_training_args(
            rc_prob_func_weight=config["weights"],
            lr=config["lr"],
            lr_step=50
        )
        # 执行阶段训练
        trainer.train(config["epochs"], train_dataset, val_dataset)
        # 保存阶段模型
        if trainer.local_rank == 0:
            trainer.save(os.path.join(trainer.log_dir, f'stage_{stage_idx+1}.pth'))

    # trainer.set_training_args(lr=args.lr, lr_step=50)
    # trainer.train(args.num_epochs, train_dataset, val_dataset)
    
    print('\n[INFO] All training stages completed!')
    
