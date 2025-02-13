# Adversarial Diffusion Bridge Model

This is the code for ADBM: Adversarial Diffusion Bridge Model for Reliable Adversarial Purification.

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

### Acknowledgement

If you find that our work is helpful to you, please star this project and consider cite:

```
@inproceedings{li2024adbm,
  author    = {Li, Xiao and Sun, Wenxuan and Chen, Huanran and Li, Qiongxiu and Liu, Yining and He, Yingzhe and Shi, Jie and Hu, Xiaolin},
  title     = {ADBM: Adversarial diffusion bridge model for reliable adversarial purification},
  booktitle = International Conference on Learning Representations (ICLR),
  year      = {2025}
}
```
