# FingerprintAttack 

This repository is dedicated to **Fingerprint Attack: Client De-Anonymization in Federated Learning** (accepted to ECAI 2023). We demonstrate a new fingerprinting attack on the shuffle module in federated learning (FL).


## Step 0: Get Started
- Install PyTorch: python=3.10, pytorch-cuda=11.7, pytorch, torchvision, torchaudio, sentencepiece, transformers, datasets, sklearn etc.;
- Install local FLSim (for simulating FL): 1. download/clone FLSim [code](https://github.com/facebookresearch/FLSim.git); 2. add some modifications to save gradient updates (more details in next section); 3. install local FLSim;
- Install opacus_lab (for refactoring GPT2): download/clone opacus_lab [code](https://github.com/facebookresearch/Opacus-lab.git) and add dir_path to lm_exp.py.

## Step 1: Run FL Simulator
```
python lm_exp.py --config-file configs/news_fed_config.json 
python lm_exp.py --config-file configs/dial_fed_config.json  
```

Note that: the gradients of the models can be recorded by adding the following code to '_update_clients' (FLSim: trainers/sync_trainer.py) and some minor corresponding changes.
```
import os
path = os.path.join(self.cfg.client.store_models_dir, '{}_e{}.pt'.format(client.name, epoch))
torch.save([(name, w.grad.data.cpu()) for name, w in client_delta.model.named_parameters()], path)
```

## Step 2: Run Analysis (Clustering and Alignment)
```
python analyse_cluster.py --cluster greedy_match

python analyse_cluster.py --cluster spectral

python analyse_cluster.py --cluster kmean
```

## Citation
This work is to appear in ECAI 2023:
```bibtex
@inproceedings{
  xu2023fingerprint,
  title={Fingerprint Attack: Client De-Anonymization in Federated Learning},
  author={Qiongkai Xu and Trevor Cohn and Olga Ohrimenko},
  booktitle={The 26th European Conference on Artificial Intelligence},
  year={2023}
}

```



