# Evolution-aware Message Passing Neural Network (EvolMPNN)

### Required packages
The code has been tested running under Python 3.8.16 with the following major packages installed (along with their dependencies):

- pytorch == 1.21.1
- torch_geometric == 2.3.0
- torchdrug == 0.2.1

### Data requirement
All three datasets we used in the paper are available in the folder data.

### Code execution
Demo execution:
```
python main_batch.py --dataset GB1 --split low_vs_high --feature_generator CNN --model EvolMPNN --full_residue_feat True --lr 0.01 --gnn_layers 2 --former_layers 2 --num_heads 4 --epochs 100 --batch_size 256 --gpu_id 1
```
