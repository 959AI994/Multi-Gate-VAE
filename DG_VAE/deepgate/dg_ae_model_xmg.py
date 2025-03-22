from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
# 不再使用 generate_hs_init
from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
# 引入 DirectedGAE 的相关部分
from .digae_layer import DirectedInnerProductDecoder 

EPS        = 1e-15
MAX_LOGSTD = 10

class Model(nn.Module):
    '''
    Recurrent Graph Neural Networks for XMG Circuits.
    Integrated version: 结构信息由外部结构编码器获得（在此版本中由上层输入），
    与功能隐藏状态联合构成节点状态，仅更新功能隐藏状态 hf.
    '''
    def __init__(self, 
                 struct_encoder,  # 外部结构编码器，用于获得结构编码 s（例如来自 DirectedGAE）
                 num_rounds = 1, 
                 dim_hidden = 128, 
                 enable_encode = True,
                 enable_reverse = True):
        super(Model, self).__init__()
        
        # 结构编码器 (来自 DirectedGAE)
        self.struct_encoder = struct_encoder
        self.decoder = DirectedInnerProductDecoder()
        self.hs_linear = nn.Linear(dim_hidden * 2, dim_hidden)
        self.hs_decompose = nn.Linear(dim_hidden, dim_hidden * 2)

        # 配置
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse

        # 维度
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        # 功能部分聚合器：输入为结构和功能拼接后的节点状态
        self.aggr_and_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        self.aggr_not_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        self.aggr_xor_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        self.aggr_maj_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        self.aggr_or_func  = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        
        # 更新模块 (GRU) 仅用于更新功能隐藏状态 hf
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_xor_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_maj_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_or_func  = GRU(self.dim_hidden, self.dim_hidden)

        # Readout 网络
        self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, 
                                p_drop=0.2, norm_layer='batchnorm', act_layer='relu')

    def forward(self, G):
        device = next(self.parameters()).device
        num_nodes = len(G.gate)
        num_layers_f = max(G.forward_level).item() + 1

        # 使用外部结构编码器获得结构编码 s（以及 t，可用于后续重构任务）
        x, edge_index = G.x, G.edge_index
        one_hot = torch.nn.functional.one_hot(G.xmg_x[:, 1].to(int), num_classes=6).to(device)
        s, t = self.xmg_struct_encoder(one_hot, one_hot, edge_index)  # s 为结构信息，t 可用于后续重构

        # 初始化功能隐藏状态 hf（结构信息 s 保持不变）
        hf = torch.zeros(num_nodes, self.dim_hidden, device=device)
        hs = self.hs_linear(torch.cat([s, t], dim=-1))
        # 初始节点状态为结构信息与功能状态的拼接
        node_state = torch.cat([hs, hf], dim=-1)

        # 获取各类门的掩码
        not_mask = G.gate.squeeze(1) == 2  # NOT 门
        and_mask = G.gate.squeeze(1) == 3  # AND 门
        or_mask  = G.gate.squeeze(1) == 4  # OR 门
        maj_mask = G.gate.squeeze(1) == 1  # MAJ 门
        xor_mask = G.gate.squeeze(1) == 5  # XOR 门

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                layer_mask = G.forward_level == level

                # AND Gate 更新
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, dim=1)
                    msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                # NOT Gate 更新
                l_not_node = G.forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, dim=1)
                    msg = self.aggr_not_func(node_state, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)

                # XOR Gate 更新
                l_xor_node = G.forward_index[layer_mask & xor_mask]
                if l_xor_node.size(0) > 0:
                    xor_edge_index, xor_edge_attr = subgraph(l_xor_node, edge_index, dim=1)
                    msg = self.aggr_xor_func(node_state, xor_edge_index, xor_edge_attr)
                    xor_msg = torch.index_select(msg, dim=0, index=l_xor_node)
                    hf_xor = torch.index_select(hf, dim=0, index=l_xor_node)
                    _, hf_xor = self.update_xor_func(xor_msg.unsqueeze(0), hf_xor.unsqueeze(0))
                    hf[l_xor_node, :] = hf_xor.squeeze(0)

                # Majority Gate 更新
                l_maj_node = G.forward_index[layer_mask & maj_mask]
                if l_maj_node.size(0) > 0:
                    maj_edge_index, maj_edge_attr = subgraph(l_maj_node, edge_index, dim=1)
                    msg = self.aggr_maj_func(node_state, maj_edge_index, maj_edge_attr)
                    maj_msg = torch.index_select(msg, dim=0, index=l_maj_node)
                    hf_maj = torch.index_select(hf, dim=0, index=l_maj_node)
                    _, hf_maj = self.update_maj_func(maj_msg.unsqueeze(0), hf_maj.unsqueeze(0))
                    hf[l_maj_node, :] = hf_maj.squeeze(0)

                # OR Gate 更新
                l_or_node = G.forward_index[layer_mask & or_mask]
                if l_or_node.size(0) > 0:
                    or_edge_index, or_edge_attr = subgraph(l_or_node, edge_index, dim=1)
                    msg = self.aggr_or_func(node_state, or_edge_index, or_edge_attr)
                    or_msg = torch.index_select(msg, dim=0, index=l_or_node)
                    hf_or = torch.index_select(hf, dim=0, index=l_or_node)
                    _, hf_or = self.update_or_func(or_msg.unsqueeze(0), hf_or.unsqueeze(0))
                    hf[l_or_node, :] = hf_or.squeeze(0)

                # 更新节点状态：重新拼接不变的结构编码 s 与最新的 hf
                node_state = torch.cat([hs, hf], dim=-1)

        # 返回结构编码 s 与功能隐藏状态 hf
        return hs,hf
    
    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        return torch.clamp(prob, min=0.0, max=1.0)
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if k not in state_dict:
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
    def load_pretrained(self, pretrained_model_path = ''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)

    def recon_loss(self, hs, pos_edge_index, neg_edge_index=None):
        s, t = self.hs_decompose(hs).chunk(2, dim=-1)
        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        pos_pred_bin = (pos_pred > 0.5).float()
        pos_gt = torch.ones_like(pos_pred)
        pos_loss = -torch.log(pos_pred + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        neg_pred_bin = (neg_pred > 0.5).float()
        neg_gt = torch.zeros_like(neg_pred)
        neg_loss = -torch.log(1 - neg_pred + EPS).mean()
        
        pred_bin = torch.cat([pos_pred_bin, neg_pred_bin], dim=0)
        gt_bin = torch.cat([pos_gt, neg_gt], dim=0)
        pred_bin = pred_bin.int()
        gt_bin = gt_bin.int()

        return pos_loss + neg_loss, pred_bin, gt_bin