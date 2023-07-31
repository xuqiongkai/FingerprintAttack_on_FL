# FingerprintAttack 

This repository is dedicated to **Fingerprint Attack: Client De-Anonymization in Federated Learning** (accepted to ECAI 2023). We demonstrate a new fingerprinting attack on the shuffle module in federated learning (FL).


## Step 0: Get Started
- Install PyTorch: python=3.10, pytorch, torchvision, torchaudio pytorch-cuda=11.7, sentencepiece, transformers, datasets, sklearn;
- Install local FLSim: 1. download/clone FLSim [code](https://github.com/facebookresearch/FLSim.git); 2. add some modifications to save gradient updates (more details in next section);
- Install opacus_lab (for refactoring GPT2): download/clone opacus_lab [code](https://github.com/facebookresearch/Opacus-lab.git)

## Step 1: Run FL Sumulator
```
python lm_exp.py --config-file configs/news_fed_config.json 
python lm_exp.py --config-file configs/dial_fed_config.json  
```

Note that: gradients of the models can be recorded by add the following code to Opacus.
```
TODO
```

## Step 2: Run Analysis (Clustering and Alignment)
```
python analyse_cluster.py --cluster greedy_match

python analyse_cluster.py --cluster spectral

python analyse_cluster.py --cluster kmean
```

### Citation
To appear at ECAI 2023:
```bibtex
@inproceedings{
  xu2023fingerprint,
  title={Fingerprint Attack: Client De-Anonymization in Federated Learning},
  author={Qiongkai Xu and Trevor Cohn and Olga Ohrimenko},
  booktitle={The 26th European Conference on Artificial Intelligence},
  year={2023}
}

```



