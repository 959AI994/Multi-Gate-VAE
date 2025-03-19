# ----------------------------------------------- 新版：集成ae-hs和hf的model-----------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import LSTM, GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph  # 确保这些工具函数可用
# from .utils.utils import generate_hs_init  # 移除 hs 相关初始化
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from .arch.mlp import MLP  # 确保这些模块已正确导入或复制到相应目录
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv

# 引入 DirectedGAE 的相关部分
from .digae_layer import DirectedInnerProductDecoder  

EPS        = 1e-15
MAX_LOGSTD = 10


class IntegratedModel(nn.Module):
    def __init__(self, struct_encoder, num_rounds=1, dim_hidden=128, enable_encode=True, enable_reverse=False):
        super(IntegratedModel, self).__init__()

        # 结构编码器 (来自 DirectedGAE)
        self.struct_encoder = struct_encoder
        self.decoder = DirectedInnerProductDecoder()

        # 功能编码器 
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode  # TODO:review,此参数可能不再需要？因为结构编码已分离
        self.enable_reverse = enable_reverse  # TODO:review,如果结构编码器使用，这里可能也需要？
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32   # TODO:review

        # 功能部分网络
        self.aggr_and_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        self.aggr_not_func = TFMlpAggr(self.dim_hidden * 1, self.dim_hidden)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)

        # Readout (用于功能预测)
        self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm',
                                act_layer='relu')

    def forward(self, data):
        # 结构编码
        x, edge_index = data.x, data.edge_index
        s, t = self.struct_encoder(x, x, edge_index)

        # 功能编码
        device = next(self.parameters()).device
        num_nodes = len(x)
        num_layers_f = max(data.forward_level).item() + 1
        # num_layers_b = max(data.backward_level).item() + 1

        # 初始化功能隐藏状态
        hf = torch.zeros(num_nodes, self.dim_hidden).to(device)
        node_state = torch.cat([s, hf], dim=-1) # TODO:review,将s作为功能部分输入
        and_mask = data.gate.squeeze(1) == 1
        not_mask = data.gate.squeeze(1) == 2
        for level in range(1,num_layers_f):
            layer_mask = data.forward_level == level

            # AND Gate
            l_and_node = data.forward_index[layer_mask & and_mask]
            if l_and_node.size(0) > 0:
                and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index,
                                                         dim=1)  # subgraph function is available

                # Update function hidden state
                msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
                and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                hf[l_and_node, :] = hf_and.squeeze(0)

            # NOT Gate
            l_not_node = data.forward_index[layer_mask & not_mask]
            if l_not_node.size(0) > 0:
                not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index,
                                                         dim=1)  # subgraph function is available
                # Update function hidden state
                msg = self.aggr_not_func(hf, not_edge_index, not_edge_attr) # 直接输入hf
                not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                hf[l_not_node, :] = hf_not.squeeze(0)
            node_state = torch.cat([s, hf], dim=-1)

        # 返回结构编码 (s, t) 和功能编码 (hf)
        return s, t, hf

    def pred_prob(self, hf):
        """预测概率 (用于功能任务)"""
        prob = self.readout_prob(hf)
        prob = torch.clamp(prob, min=0.0, max=1.0)
        return prob

    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
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

# ----------------------------------------------- 旧版：hs和hf版本的model-----------------------------------------------------------------
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import torch
# import os
# from torch import nn
# from torch.nn import LSTM, GRU
# from .utils.dag_utils import subgraph, custom_backward_subgraph
# from .utils.utils import generate_hs_init

# from .arch.mlp import MLP
# from .arch.mlp_aggr import MlpAggr
# from .arch.tfmlp import TFMlpAggr
# from .arch.gcn_conv import AggConv

# class Model(nn.Module):
#     '''
#     Recurrent Graph Neural Networks for Circuits.
#     '''
#     def __init__(self, 
#                  num_rounds = 1, 
#                  dim_hidden = 128, 
#                  enable_encode = True,
#                  enable_reverse = False
#                 ):
#         super(Model, self).__init__()
        
#         # configuration
#         self.num_rounds = num_rounds
#         self.enable_encode = enable_encode
#         self.enable_reverse = enable_reverse        # TODO: enable reverse

#         # dimensions
#         self.dim_hidden = dim_hidden
#         self.dim_mlp = 32

#         # Network 
#         self.aggr_and_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
#         self.aggr_not_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
#         self.aggr_and_func = TFMlpAggr(self.dim_hidden*2, self.dim_hidden)
#         self.aggr_not_func = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
            
#         self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
#         self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
#         self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
#         self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)

#         # Readout 
#         self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')

#         # # consider the embedding for the LSTM/GRU model initialized by non-zeros
#         # self.one = torch.ones(1)
#         # # self.hs_emd_int = nn.Linear(1, self.dim_hidden)
#         # self.hf_emd_int = nn.Linear(1, self.dim_hidden)
#         # self.one.requires_grad = False

#     def forward(self, G):
#         device = next(self.parameters()).device
#         num_nodes = len(G.gate)
#         num_layers_f = max(G.forward_level).item() + 1
#         num_layers_b = max(G.backward_level).item() + 1
        
#         # initialize the structure hidden state
#         if self.enable_encode:
#             hs = torch.zeros(num_nodes, self.dim_hidden)
#             hs = generate_hs_init(G, hs, self.dim_hidden)
#         else:
#             hs = torch.zeros(num_nodes, self.dim_hidden)
        
#         # initialize the function hidden state
#         # hf = self.hf_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
#         # hf = hf.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
#         hf = torch.zeros(num_nodes, self.dim_hidden)
#         hs = hs.to(device)
#         hf = hf.to(device)
        
#         edge_index = G.edge_index

#         node_state = torch.cat([hs, hf], dim=-1)
#         and_mask = G.gate.squeeze(1) == 1
#         not_mask = G.gate.squeeze(1) == 2

#         for _ in range(self.num_rounds):
#             for level in range(1, num_layers_f):
#                 # forward layer
#                 layer_mask = G.forward_level == level

#                 # AND Gate
#                 l_and_node = G.forward_index[layer_mask & and_mask]
#                 if l_and_node.size(0) > 0:
#                     and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, dim=1)
                    
                    
#                     # Update structure hidden state
#                     msg = self.aggr_and_strc(hs, and_edge_index, and_edge_attr)
#                     and_msg = torch.index_select(msg, dim=0, index=l_and_node)
#                     hs_and = torch.index_select(hs, dim=0, index=l_and_node)
#                     _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
#                     hs[l_and_node, :] = hs_and.squeeze(0)
#                     # Update function hidden state
#                     msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
#                     and_msg = torch.index_select(msg, dim=0, index=l_and_node)
#                     hf_and = torch.index_select(hf, dim=0, index=l_and_node)
#                     _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
#                     hf[l_and_node, :] = hf_and.squeeze(0)

#                 # NOT Gate
#                 l_not_node = G.forward_index[layer_mask & not_mask]
#                 if l_not_node.size(0) > 0:
#                     not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, dim=1)
#                     # Update structure hidden state
#                     msg = self.aggr_not_strc(hs, not_edge_index, not_edge_attr)
#                     not_msg = torch.index_select(msg, dim=0, index=l_not_node)
#                     hs_not = torch.index_select(hs, dim=0, index=l_not_node)
#                     _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
#                     hs[l_not_node, :] = hs_not.squeeze(0)
#                     # Update function hidden state
#                     msg = self.aggr_not_func(hf, not_edge_index, not_edge_attr)
#                     not_msg = torch.index_select(msg, dim=0, index=l_not_node)
#                     hf_not = torch.index_select(hf, dim=0, index=l_not_node)
#                     _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
#                     hf[l_not_node, :] = hf_not.squeeze(0)
                
#                 # Update node state
#                 node_state = torch.cat([hs, hf], dim=-1)

#         node_embedding = node_state.squeeze(0)
#         hs = node_embedding[:, :self.dim_hidden]
#         hf = node_embedding[:, self.dim_hidden:]

#         return hs, hf
    
#     def pred_prob(self, hf):
#         prob = self.readout_prob(hf)
#         prob = torch.clamp(prob, min=0.0, max=1.0)
#         return prob
    
#     def load(self, model_path):
#         checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
#         state_dict_ = checkpoint['state_dict']
#         state_dict = {}
#         for k in state_dict_:
#             if k.startswith('module') and not k.startswith('module_list'):
#                 state_dict[k[7:]] = state_dict_[k]
#             else:
#                 state_dict[k] = state_dict_[k]
#         model_state_dict = self.state_dict()
        
#         for k in state_dict:
#             if k in model_state_dict:
#                 if state_dict[k].shape != model_state_dict[k].shape:
#                     print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
#                         k, model_state_dict[k].shape, state_dict[k].shape))
#                     state_dict[k] = model_state_dict[k]
#             else:
#                 print('Drop parameter {}.'.format(k))
#         for k in model_state_dict:
#             if not (k in state_dict):
#                 print('No param {}.'.format(k))
#                 state_dict[k] = model_state_dict[k]
#         self.load_state_dict(state_dict, strict=False)
        
#     def load_pretrained(self, pretrained_model_path = ''):
#         if pretrained_model_path == '':
#             pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
#         self.load(pretrained_model_path)
