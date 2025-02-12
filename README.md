### Introduction
We can use the code in this repository to extract embeddings of different modes, such as xag, mig, xmg, aig.

### Prompt
（1）导入deepgate包 <br>
export PYTHONPATH=~/multi_gate/multi_gate_2:$PYTHONPATH <br>
（2）运行 <br>
cd examples/ <br>
python -m torch.distributed.launch --nproc_per_node=2  train.py<br>

### Tools
（1）merge.py:可以用merge.py脚本来合并graphs.py和labels.py中的内容<br>
（2）parser.py： <br>
    ①bench_parser类中，训练xmg,mig,xag的时候需要将edge_index进行转置，“edge_index = edge_index.t().contiguous()”，aig则不需要转置，把该行代码注释掉即可<br>
    ②共六个聚合器，需要考虑6种，gate_to_index = {'INPUT': 0, 'MAJ': 1, 'NOT': 2, 'AND': 3, 'OR': 4, 'XOR': 5} <br>
    ③聚合器命名要和gate_to_index对应上：<br>
        node_state = torch.cat([hs, hf], dim=-1) <br>
        not_mask = G.gate.squeeze(1) == 2  # NOT门的掩码 <br>
        and_mask = G.gate.squeeze(1) == 3  # AND门的掩码 <br>
        or_mask = G.gate.squeeze(1) == 4   # OR门的掩码 <br>
        maj_mask = G.gate.squeeze(1) == 1  # MAJ门的掩码 <br>
        xor_mask = G.gate.squeeze(1) == 5  # XOR门的掩码 <br>