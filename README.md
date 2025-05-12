# SgVA-CLIP: Semantic-guided Visual Adapting of Vision-Language Models for Few-shot Image Classification
[[Paper]](https://ieeexplore.ieee.org/abstract/document/10243119)
[[Arxiv]](https://arxiv.org/abs/2211.16191)

**Model Architecture**

<img src="Fig1.jpg" width="900px">

**Running Environment**

Refer to [CLIP](https://github.com/openai/CLIP) and [CoOp](https://github.com/KaiyangZhou/CoOp)

**Change base_datadir, base_dir in main.py**
  
**Change _IMAGE_DATASET_DIR in dataset/xx.py**

For example, change _IMAGE_DATASET_DIR in dataset/dtd.py (Every dataset loading python file should be changed.)
注意：每个数据集路径下的class_names.txt是一个所有类别名逐行排列的txt文件（不带序号），如果没有可以自行生成。The class_names.txt under each dataset path is a txt file in which all category names are arranged line by line, without serial numbers.

**Dataset Preparation**
For data preparation, please refer to [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)

**Train Example**

```
python -W ignore main.py --phase metatrain --gpu 0 --save-path ''
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
```
注意：train-way最好要能被n-classes整除（如果n-classes是质数，建议train-way=n-classes），测试时test-way也要能被n-classes整除（如果n-classes是质数，建议test-way=1），val-shot可以和train-shot保持一致。

**Test Example**

```
python -W ignore main.py --phase metatest --gpu 0 --save-path ''
--train-way 49 --test-way 10 --train-shot 16 --val-shot 16 --train-query 50 --val-query 50 
--loss newi2i+kd+i2t 
--para-update prompt+v_adapter 
--adapter-dim _dim_x2 
--visual-pre 'T' 
--head Vision-Text 
--network CLIP 
--backbone RN50 
--test-head Fuse_af 
--dataset SUN397 
--n-classes 397 
--warm-up True 
--num-epoch 100 
--ctx-init 'a photo of a' 
--n-ctx 4 
--lr-decay cosine 
--episodes-per-batch 1 
```
注意：测试状态下使用--phase metatest；train-way最好要能被n-classes整除（如果n-classes是质数，建议train-way=n-classes），测试时test-way也要能被n-classes整除（如果n-classes是质数，建议test-way=1），val-shot可以和train-shot保持一致。

**Cite**
```
@article{peng2023sgva,
  title={Sgva-clip: Semantic-guided visual adapting of vision-language models for few-shot image classification},
  author={Peng, Fang and Yang, Xiaoshan and Xiao, Linhui and Wang, Yaowei and Xu, Changsheng},
  journal={IEEE Transactions on Multimedia},
  volume={26},
  pages={3469--3480},
  year={2023},
  publisher={IEEE}
}
```
