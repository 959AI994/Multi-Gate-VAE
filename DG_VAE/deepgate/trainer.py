from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import numpy as np
import time
from progress.bar import Bar
from torch_geometric.loader import DataLoader

from .arch.mlp import MLP
from .utils.utils import zero_normalization, AverageMeter
from .utils.logger import Logger
from .utils.model_utils import load_model

from .preprocessing import general_train_test_split_edges

class Trainer():
    def __init__(self,
                 args, 
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 prob_rc_func_weight = [3.0, 1.0, 2.0],
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=32, num_workers=4, 
                 distributed = False
                 ):
        super(Trainer, self).__init__()
        # Config
        self.args = args
        self.emb_dim = emb_dim
        self.device = device
        self.lr = lr
        self.lr_step = -1
        self.prob_rc_func_weight = prob_rc_func_weight
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log_dir = os.path.join(save_dir, training_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # Log Path
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_path = os.path.join(self.log_dir, 'log-{}.txt'.format(time_str))
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = 'cuda:%d' % self.local_rank
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            print('Training in single device: ', self.device)
        
        # Loss and Optimizer
        self.reg_loss = nn.L1Loss().to(self.device)
        self.clf_loss = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        self.readout_rc = MLP(emb_dim * 2, 32, 1, num_layer=3, p_drop=0.1, norm_layer='batchnorm').to(self.device)
        self.model_epoch = 0
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
        
    def set_training_args(self, prob_rc_func_weight=[], lr=-1, lr_step=-1, device='null'):
        if len(prob_rc_func_weight) == 3 and prob_rc_func_weight != self.prob_rc_func_weight:
            print('[INFO] Update prob_rc_func_weight from {} to {}'.format(self.prob_rc_func_weight, prob_rc_func_weight))
            self.prob_rc_func_weight = prob_rc_func_weight
        if lr > 0 and lr != self.lr:
            print('[INFO] Update learning rate from {} to {}'.format(self.lr, lr))
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if lr_step > 0 and lr_step != self.lr_step:
            print('[INFO] Update learning rate step from {} to {}'.format(self.lr_step, lr_step))
            self.lr_step = lr_step
        if device != 'null' and device != self.device:
            print('[INFO] Update device from {} to {}'.format(self.device, device))
            self.device = device
            self.model = self.model.to(self.device)
            self.reg_loss = self.reg_loss.to(self.device)
            self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
            self.readout_rc = self.readout_rc.to(self.device)
        
    def save(self, path):
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.model_epoch = checkpoint['epoch']
        self.model.load(path)
        print('[INFO] Continue training from epoch {:}'.format(self.model_epoch))
        return path
    
    def resume(self):
        model_path = os.path.join(self.log_dir, 'model_last.pth')
        if os.path.exists(model_path):
            self.model, self.optimizer, self.model_epoch = load_model(self.model, model_path, optimizer=self.optimizer, local_rank=self.local_rank, device=self.device)
            return True
        else:
            return False
        
    def run_batch(self, batch):
        # 增强型split验证
        print(f"[流程验证] split前edge_index存在: {hasattr(batch, 'edge_index')}")
        # 添加split前的边索引保护
        if not hasattr(batch, 'edge_index') or batch.edge_index is None:
            print("[紧急修复] 创建临时edge_index")
            batch.edge_index = torch.tensor([[0], [0]], device=self.device)
        # 执行边分割
        batch = general_train_test_split_edges(batch)
        # 分割后多重验证
        edge_attr_exists = hasattr(batch, 'train_pos_edge_index')
        print(f"[流程验证] split后train_pos_edge_index存在: {edge_attr_exists}")
        # 存在性修复
        if not edge_attr_exists:
            print("[紧急修复] 添加train_pos_edge_index属性")
            batch.train_pos_edge_index = batch.edge_index.clone()
        # 空值过滤
        if batch.train_pos_edge_index is None:
            print("[空值修复] 生成默认边索引")
            batch.train_pos_edge_index = torch.tensor([[0], [0]], device=self.device)
        # 类型强制转换（增强版）
        if not isinstance(batch.train_pos_edge_index, torch.Tensor):
            print(f"[类型修复] 检测到异常类型{type(batch.train_pos_edge_index)}，进行转换")
            try:
                batch.train_pos_edge_index = torch.tensor(batch.train_pos_edge_index, 
                                                        device=self.device,
                                                        dtype=torch.long)
            except Exception as e:
                print(f"[类型转换失败] 错误信息: {e}, 使用备用方案")
                batch.train_pos_edge_index = torch.tensor([[0], [0]], device=self.device)
        # 维度修复（增强版）
        if batch.train_pos_edge_index.dim() != 2:
            print(f"[维度修复] 异常维度{batch.train_pos_edge_index.shape}，进行重塑")
            try:
                batch.train_pos_edge_index = batch.train_pos_edge_index.view(2, -1)
            except:
                print("[维度修复失败] 使用默认维度")
                batch.train_pos_edge_index = torch.tensor([[0], [0]], device=self.device)
        # 设备同步（强制版）
        if batch.train_pos_edge_index.device != self.device:
            print(f"[设备同步] 从{batch.train_pos_edge_index.device}同步到{self.device}")
            batch.train_pos_edge_index = batch.train_pos_edge_index.to(self.device)
        
        # 最终验证
        print(f"[最终检查] edge_index形状: {batch.train_pos_edge_index.shape}")
        print(f"[最终检查] edge_index类型: {type(batch.train_pos_edge_index)}")
        print(f"[最终检查] edge_index设备: {batch.train_pos_edge_index.device}")

        # Reconstruction
        u = batch.x.clone()
        v = batch.x.clone()
            
        edge_id_before = id(batch.train_pos_edge_index) # debug

        s, t = self.model.encode(u, v, batch.train_pos_edge_index)

        print(f"[数据流验证] edge_index ID是否变化: {edge_id_before == id(batch.train_pos_edge_index)}")

        loss, pred_bin, gt_bin = self.model.recon_loss(s, t, batch.train_pos_edge_index)
        
        # Variational
        if 'VAE' in self.args.model:
            s_kl = -0.5/ u.size(0) * (1 + 2*self.model.s_logstd - self.model.s_mu**2 - torch.exp(self.model.s_logstd)**2).sum(1).mean()
            t_kl = -0.5/ v.size(0) * (1 + 2*self.model.t_logstd - self.model.t_mu**2 - torch.exp(self.model.t_logstd)**2).sum(1).mean()
            kl = s_kl + t_kl
        else:
            kl = 0

        # 集成prob loss和func loss
        hs, hf = self.model(batch)
        prob = self.model.pred_prob(hf)
        # Task 1: Probability Prediction 
        prob_loss = self.reg_loss(prob, batch['prob'])
        # Task 2: Functional Similarity 
        node_a = hf[batch['tt_pair_index'][0]]
        node_b = hf[batch['tt_pair_index'][1]]
        emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
        emb_dis_z = zero_normalization(emb_dis)
        tt_dis_z = zero_normalization(batch['tt_dis'])
        func_loss = self.reg_loss(emb_dis_z, tt_dis_z)

        loss_status = {
            'recon_loss': loss, 
            'kl_loss': kl,
            'pred_bin': pred_bin,
            'gt_bin': gt_bin,
            'prob_loss': prob_loss, 
            'func_loss': func_loss
        }
        
        return loss_status
    
    def train(self, num_epoch, train_dataset, val_dataset):
        # Distribute Dataset
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                    num_workers=self.num_workers, sampler=train_sampler)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                     num_workers=self.num_workers, sampler=val_sampler)
        else:
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        # AverageMeter
        batch_time = AverageMeter()
        recon_loss_stats = AverageMeter()
        kl_loss_stats = AverageMeter()
        prob_loss_stats=AverageMeter()
        func_loss_stats=AverageMeter()
        acc_stats = AverageMeter()
        tp_stats, fp_stats, tn_stats, fn_stats = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        
        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataset = train_dataset
                    self.model.train()
                    self.model.to(self.device)
                else:
                    dataset = val_dataset
                    self.model.eval()
                    self.model.to(self.device)
                    torch.cuda.empty_cache()
                if self.local_rank == 0:
                    bar = Bar('[{} {:}] {:}/{:}'.format(phase, self.model_epoch, epoch, num_epoch), max=len(dataset))
                for iter_id, batch in enumerate(dataset):
                    self.optimizer.zero_grad()
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    loss_status = self.run_batch(batch)
                    loss = loss_status['recon_loss'] + loss_status['kl_loss']+loss_status['prob_loss']+loss_status['func_loss']
                    if phase == 'train':
                        loss.backward()
                    self.optimizer.step()
                    # Print and save log
                    pred_bin = loss_status['pred_bin'].detach().cpu()
                    gt_bin = loss_status['gt_bin'].detach().cpu()
                    pred_bin = pred_bin.numpy()
                    gt_bin = gt_bin.numpy()
                    acc = np.sum(pred_bin == gt_bin) / len(pred_bin)
                    TP = np.sum((pred_bin == 1) & (gt_bin == 1)) / len(pred_bin)
                    FP = np.sum((pred_bin == 1) & (gt_bin == 0)) / len(pred_bin)
                    TF = np.sum((pred_bin == 0) & (gt_bin == 0)) / len(pred_bin)
                    FN = np.sum((pred_bin == 0) & (gt_bin == 1)) / len(pred_bin)
                    
                    batch_time.update(time.time() - time_stamp)
                    recon_loss_stats.update(loss_status['recon_loss'].item())
                    kl_loss_stats.update(loss_status['kl_loss'].item())
                    prob_loss_stats.update(loss_status['prob_loss'].item())
                    func_loss_stats.update(loss_status['func_loss'].item())
                    acc_stats.update(acc)
                    tp_stats.update(TP)
                    fp_stats.update(FP)
                    tn_stats.update(TF)
                    fn_stats.update(FN)
                    if self.local_rank == 0:
                        Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        Bar.suffix += '|Recon: {:.4f} |KL: {:.4f} |ACC: {:.2f} |Prob: {:.4f} |Func: {:.4f} '.format(recon_loss_stats.avg, kl_loss_stats.avg, acc_stats.avg * 100,prob_loss_stats.avg,func_loss_stats.avg)
                        Bar.suffix += '|TP: {:.2f} |FP: {:.2f} |TN: {:.2f} |FN: {:.2f} '.format(tp_stats.avg * 100, fp_stats.avg * 100, tn_stats.avg * 100, fn_stats.avg * 100)
                        Bar.suffix += '|Net: {:.2f}s '.format(batch_time.avg)
                        bar.next()
                if phase == 'train' and self.model_epoch % 10 == 0:
                    self.save(os.path.join(self.log_dir, 'model_{:}.pth'.format(self.model_epoch)))
                    self.save(os.path.join(self.log_dir, 'model_last.pth'))
                if self.local_rank == 0:
                    self.logger.write('{}| Epoch: {:}/{:} |Recon: {:.4f} |KL: {:.4f} |ACC: {:.2f} |Prob: {:.4f} |Func: {:.4f}|Net: {:.2f}s\n'.format(
                        phase, epoch, num_epoch, recon_loss_stats.avg, kl_loss_stats.avg, acc_stats.avg*100, prob_loss_stats.avg,func_loss_stats.avg,batch_time.avg))
                    bar.finish()
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            
