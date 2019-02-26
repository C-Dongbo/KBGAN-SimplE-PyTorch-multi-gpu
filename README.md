# KBGAN with TransE, TransD, ComplEx, SimplE
## This repository is the using multi gpu pytorch-v0.4.1 to training KBGAN-SimplE implement version for KBGAN.

> The origin repo: https://github.com/cai-lw/KBGAN
> Liwei Cai and William Yang Wang, "KBGAN: Adversarial Learning for Knowledge Graph Embeddings", in *Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT 2018)*.    
> Paper: https://arxiv.org/abs/1711.04071

> Seyed Mehran Kazemi and Dabid Poole, "SimplE Embedding for Link Prediction in Knowledge Graphs", in *NIPS 2018*
> Paper: https://arxiv.org/pdf/1802.04868.pdf



##Summary
---
* In the existing KBGAN paper, experiments were conducted using ComplEx and Translation based models(TransE, TransD)
* In addition, SimplE models were implemented in pytorch 0.4.1 version
* And enabled SimplE model for the KBGAN Framework
* The best performance is currently under testing.


## Dependencies
* Python 3.6
* PyTorch 0.4.1
* PyYAML
* nvidia-smi


## Usage
- - -
* Pretrain(for example):   
python pretrain.py --config=config_wn18.yaml --pretrain_config=TransE  
python pretrain.py --config=config_wn18.yaml --pretrain_config=SimplE  
(this will generate a pretrained model file)
* Adversarial train(for example):  
 python gan_train.py --config=config_wn18.yaml --g_config=SimplE --d_config=TransE  
(make sure that G model and D model are both pretrained)   

- - -
Feel free to explore and modify parameters in config files. Default parameters are those used in experiments reported in the paper.  
Decrease **test_batch_size** in config files if you experience GPU memory exhaustion. (this would make the program runs slower, but would not affect the test result)
