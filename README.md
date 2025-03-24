# SgVA-CLIP: Semantic-guided Visual Adapting of Vision-Language Models for Few-shot Image Classification
参考训练脚本
```
python -W ignore main_5.py --phase metatrain --gpu 0 --save-path ”“
--train-way 50 --test-way 10 --train-shot 16 --val-shot 16 --train-query 50 --val-query 50 
--loss newi2i+kd+i2t
--para-update prompt+v_adapter 
--adapter-dim _dim_x2 
--visual-pre 'T' 
--head Vision-Text 
--network CLIP 
--backbone RN50 
--test-head Fuse_af 
--dataset ImageNet 
--n-classes 1000 
--warm-up True 
--num-epoch 100 
--ctx-init ensemble 
--n-ctx 4 
--lr-decay cosine 
--episodes-per-batch 1 
--lam2 0.1
```
