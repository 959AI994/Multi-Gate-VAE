# ----------------------------------------------- 不带is_rc标签的parser-----------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Callable, List
import os.path as osp

import numpy as np 
import torch
import shutil
import os
import copy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .utils.data_utils import read_npz_file
from .utils.aiger_utils import aig_to_xdata_bak
from .utils.circuit_utils import get_fanin_fanout, read_file, add_node_index, feature_gen_connect
# from .parser_func import *

class NpzParser():
    '''
        Parse the npz file into an inmemory torch_geometric.data.Data object
    '''
    def __init__(self, data_dir, circuit_path, label_path, circuit_type,random_shuffle=True, trainval_split=0.9): 
        self.data_dir = data_dir
        self.circuit_type = circuit_type  # 保存电路类型
        dataset = self.inmemory_dataset(data_dir, circuit_path, label_path,circuit_type)
        if random_shuffle:
            perm = torch.randperm(len(dataset))
            dataset = dataset[perm]
        data_len = len(dataset)
        training_cutoff = int(data_len * trainval_split)
        self.train_dataset = dataset[:training_cutoff]
        self.val_dataset = dataset[training_cutoff:]
        
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, circuit_path, label_path,circuit_type, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'npz_inmm_dataset'
            self.circuit_type = circuit_type  # 保存电路类型
            self.root = root
            self.circuit_path = circuit_path
            self.label_path = label_path
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'inmemory'
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path, self.label_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass
        
        def process(self):
            # 根据类型动态导入
            if self.circuit_type == 'aig':
                from .parser_func import parse_pyg_mlpgate
                tt_key = 'tt_sim'  # AIG使用tt_sim
            else:
                from .parser_func_others import parse_pyg_mlpgate
                tt_key = 'tt_dis'  # 其他类型使用tt_dis
                labels = read_npz_file(self.label_path)['labels'].item()

            data_list = []
            tot_pairs = 0 

            circuits = read_npz_file(self.circuit_path)['circuits'].item()
            # if not aig then: 
            # labels = read_npz_file(self.label_path)['labels'].item()
            j = 0

            for cir_idx, cir_name in enumerate(circuits):
                if cir_name != "D_FF_0" and cir_name != "register_cc" and cir_name != "D_FF_1" and cir_name != "Main_led_brightness_control_PWM" and cir_name != "ProgramCounter" and cir_name != "TenHertz" and cir_name != "dlatch":
                    print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
                    #--------------------if model type = aig exchange "circuits" with "labels"--------------------
                    x = circuits[cir_name]["x"]
                    edge_index = circuits[cir_name]["edge_index"]

                    # tt_dis = circuits[cir_name][tt_key] # in aig_graphs there is no tt_dis but tt_sim
                    # tt_pair_index = circuits[cir_name]['tt_pair_index']
                    # prob = circuits[cir_name]['prob']
                    # 类型相关特征
                    if self.circuit_type == 'aig':
                        tt_dis = circuits[cir_name][tt_key]
                        tt_pair_index = circuits[cir_name]['tt_pair_index']
                        prob = circuits[cir_name]['prob']
                    else:
                        tt_dis = labels[cir_name][tt_key]  # 从label数据源获取
                        tt_pair_index = labels[cir_name]['tt_pair_index']
                        prob = labels[cir_name]['prob']

                    if len(tt_pair_index) == 0 :
                        print('No tt or rc pairs: ', cir_name)
                        continue

                    tot_pairs += len(tt_dis)

                    graph = parse_pyg_mlpgate(
                        x, edge_index, prob, tt_dis, tt_pair_index 
                    )
                    graph.gate = torch.tensor(circuits[cir_name]['gate'])
                    graph.name = cir_name
                    data_list.append(graph)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            # print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'

     
class BenchParser():
    def __init__(self, gate_to_index = {'INPUT': 0, 'MAJ': 1, 'NOT': 2, 'AND': 3, 'OR': 4, 'XOR': 5}):
        self.gate_to_index = gate_to_index
        pass
    
    def read_bench(self, bench_path):

        if self.circuit_type == 'aig':
            from .parser_func import parse_pyg_mlpgate
        else:
            from .parser_func_others import parse_pyg_mlpgate

        circuit_name = os.path.basename(bench_path).split('.')[0]
        x_data = read_file(bench_path)
        x_data, num_nodes, _ = add_node_index(x_data)
        x_data, edge_index = feature_gen_connect(x_data, self.gate_to_index)
        for idx in range(len(x_data)):
            x_data[idx] = [idx, int(x_data[idx][1])]
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph  
    