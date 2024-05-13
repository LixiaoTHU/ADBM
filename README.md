# Adversarial Diffusion Bridge Model

This is the code for the submission\_2155 entitled Adversarial Diffusion Bridge Model for Reliable Adversarial Purification.

Taking CIFAR-10 as an example, the following commands can be used to train and test the ADBM model.

### Training ADBM on CIFAR-10
```bash
accelerate launch --multi_gpu --num_processes 4 main.py --configs ft_fz_wcls_cifar10_sde.yaml
```

### Test ADBM on CIFAR-10 with PGD-200 Linf attack (EOT and full gradient)
```bash
python evaluate.py --gpu 4 \
                --scheduler ddim \
                --steps 5 \
                --noise_level 100 \
                --stack 1 \
                --configs configs/ft_fz_wcls_cifar10_sde.yaml \
                --attack_method PGD \
                --attack_type Linf \
                --unet_ckpt /dir/to/trained/ADBM/UNet/ckpt
```


### Requirements
- Python 3.8.16
- PyTorch 1.12.1 
- CUDA 11.3
- numpy
- accelerate
