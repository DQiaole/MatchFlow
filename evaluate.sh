#!/bin/bash
# MatchFlow(G)
python main.py --validation kitti sintel --restore_ckpt ckpts/matchflow-g/matchflow-g-things.pth --image_size 416 736 --eval_only
python main.py --validation sintel_submission --restore_ckpt ckpts/matchflow-g/matchflow-g-sintel.pth --image_size 384 768 --eval_only
python main.py --validation kitti_submission --restore_ckpt ckpts/matchflow-g/matchflow-g-kitti.pth --image_size 288 960 --eval_only

# MatchFlow(R)
python main.py --validation kitti sintel --restore_ckpt ckpts/matchflow-r/matchflow-r-things.pth --image_size 416 736 --raft --eval_only
