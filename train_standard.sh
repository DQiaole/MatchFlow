#!/bin/bash
# MatchFlow(G)
python main.py --name matchflow-g-chairs --stage chairs --validation chairs --output ckpts/matchflow-g --num_steps 120000 --lr 0.00025 --image_size 384 512 --wdecay 0.0001 --batch_size 8
python main.py --name matchflow-g-things --stage things --validation sintel --output ckpts/matchflow-g --restore_ckpt ckpts/matchflow-g/matchflow-g-chairs.pth --num_steps 120000 --lr 0.000125 --image_size 416 736 --wdecay 0.0001 --batch_size 6
python main.py --name matchflow-g-sintel --stage sintel --validation sintel --output ckpts/matchflow-g --restore_ckpt ckpts/matchflow-g/matchflow-g-things.pth --num_steps 120000 --lr 0.000125 --image_size 384 768 --wdecay 0.0001 --gamma 0.85 --batch_size 6
python main.py --name matchflow-g-kitti --stage kitti --validation kitti --output ckpts/matchflow-g --restore_ckpt ckpts/matchflow-g/matchflow-g-sintel.pth --num_steps 50000 --lr 0.000125 --image_size 288 960 --wdecay 0.00001 --gamma 0.85 --batch_size 6

# MatchFlow(R)
python main.py --name matchflow-r-chairs --stage chairs --validation chairs --output ckpts/matchflow-r --num_steps 120000 --lr 0.00025 --image_size 384 512 --wdecay 0.0001 --batch_size 8 --raft
python main.py --name matchflow-r-things --stage things --validation sintel --output ckpts/matchflow-r --restore_ckpt ckpts/matchflow-r/matchflow-r-chairs.pth --num_steps 120000 --lr 0.000125 --image_size 416 736 --wdecay 0.0001 --batch_size 6 --raft
