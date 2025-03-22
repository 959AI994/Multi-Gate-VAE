from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch_geometric.data import Data
from .utils.data_utils import construct_node_feature
from .utils.dag_utils import return_order_info

class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, \
                 tt_pair_index=None, tt_sim=None, \
                 rc_pair_index=None, is_rc=None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
        super().__init__()
        self.edge_index = edge_index
        self.tt_pair_index = tt_pair_index
        self.x = x
        self.y = y
        self.tt_sim = tt_sim
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
        # self.rc_pair_index = rc_pair_index
        # self.is_rc = is_rc
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        else:
            return 0

# Notice: 如果是xmg需要将num_gate_types设置成6，如果是mig就设置成5
def parse_pyg_mlpgate(x, edge_index, y, tt_sim, tt_pair_index, num_gate_types=6):
    x_torch = construct_node_feature(x, num_gate_types)#对于每个节点的门的种类，生成one hot编码  torch.tensor([[0, 1, 0], [1, 0, 0]])

    tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
    tt_pair_index = tt_pair_index.contiguous()
    tt_sim = torch.tensor(tt_sim)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    if len(edge_index) == 0:
        edge_index = edge_index.contiguous()
        forward_index = torch.LongTensor([i for i in range(len(x))])
        backward_index = torch.LongTensor([i for i in range(len(x))])
        forward_level = torch.zeros(len(x))
        backward_level = torch.zeros(len(x))
    else:
        edge_index = edge_index.contiguous()
        forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))

    graph = OrderedData(x=x_torch, edge_index=edge_index, y = y, tt_pair_index=tt_pair_index, tt_sim=tt_sim,
                        forward_level=forward_level, forward_index=forward_index, 
                        backward_level=backward_level, backward_index=backward_index)
    graph.use_edge_attr = False

    graph.prob = torch.tensor(y).reshape((len(x), 1))

    return graph

