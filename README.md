### Introduction
We can use the code in this repository to extract embeddings of different modes, such as xag, mig, xmg, aig.

### Prompt
（1）导入deepgate包
export PYTHONPATH=~/multi_gate/multi_gate_2:$PYTHONPATH
（2）运行
cd examples/
python -m torch.distributed.launch --nproc_per_node=2  train.py

### Tools
（1）merge.py:可以用merge.py脚本来合并graphs.py和labels.py中的内容
（2）parser.py：①bench_parser类中，训练xmg,mig,xag的时候需要将edge_index进行转置，“edge_index = edge_index.t().contiguous()”，aig则不需要转置，把该行代码注释掉即可；
②共六个聚合器，需要考虑6种，gate_to_index = {'INPUT': 0, 'MAJ': 1, 'NOT': 2, 'AND': 3, 'OR': 4, 'XOR': 5}；