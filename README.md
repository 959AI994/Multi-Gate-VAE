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