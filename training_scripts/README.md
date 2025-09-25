# About
For reproducibility the training scripts that were used to fine-tuning models are given in this directory.

# Freezing of layers
- CHGNet, the layers to be frozen were added by explicitly setting `requires_grad` to `False` for example:
```
for param in chgnet.atom_embedding.parameters():
    param.requires_grad = False
```
- MACE: Freezing of layers were done based on (http://arxiv.org/abs/2502.15582).

 
