# KBGAN with TransE, TransD, ComplEx, SimplE
## This repository is the using multi gpu pytorch-v0.4.1 to training KBGAN-SimplE implemented version for KBGAN Framework.

> The origin repo: https://github.com/cai-lw/KBGAN
> Liwei Cai and William Yang Wang, "KBGAN: Adversarial Learning for Knowledge Graph Embeddings", in *Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT 2018)*.    
> Paper: https://arxiv.org/abs/1711.04071

> Seyed Mehran Kazemi and Dabid Poole, "SimplE Embedding for Link Prediction in Knowledge Graphs", in *NIPS 2018*
> Paper: https://arxiv.org/pdf/1802.04868.pdf



## Summary
- - -
* In the existing KBGAN paper, experiments were conducted using ComplEx and Translation based models(TransE, TransD)
* In addition, SimplE models were implemented in pytorch 0.4.1 version (multi gpu). SimplE model has better performance than ComplEx.
* Maybe you do not necessary multi gpu training for benchmark dataset like WN18,fb15k. But you need to training using multi gpu for your own large scale dataset.
* And enabled SimplE model for the KBGAN Framework.
* The best performance is currently under testing.



## eval (Continually updated)
* WN18 on ComplEx (simgle model) : Test_H@1 = 0.7367 , Test_H@10 = 0.9450
* WN18 on SimplE (single model) : Test_H@1 = 0.8379, Test_H@10 = 0.9493




## Dependencies
* Python 3.5
* PyTorch 0.4.1 (cuda 9.0)
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
* Eval (for example):
python gan_eval.py --config=config_wn18.yaml --g_config=TransE --d_config=SimplE --kbgan_config={"your gan model"}

- - -
Feel free to explore and modify parameters in config files. Default parameters are those used in experiments reported in the paper.  
Decrease **test_batch_size** in config files if you experience GPU memory exhaustion. (this would make the program runs slower, but would not affect the test result)
