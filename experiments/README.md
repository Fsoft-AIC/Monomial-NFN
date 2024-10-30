Follow the experiment instructions from [Permutation Equivariant Neural Functionals paper](https://github.com/AllanYangZhou/nfn)

The experiments require additional dependencies. In your virtual env:
```bash
pip install -r requirements.txt
pip install -e experiments/perceiver-pytorch
```
We use Python 3.8.16.

## Predicting generalization for small CNNs
### Getting CNN Zoo data
Download the [CIFAR10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz) data  (originally from [Unterthiner et al, 2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy)) into `./experiments/data`, and extract them. Change `data_root_dir` in `main.yaml` if you want to store the data somewhere else.

Download the [Augment data for CIFAR10](https://drive.google.com/file/d/1lpaRwbSunFKz_x-6HAa98ZqcSrG157ks/view?usp=sharing) into `./experiments` and extract them


### Our model, Tanh, with augment [-1,1]:
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnps_inv_tanh nfnet.model_type="S_tanh_abs_gen" nfnet.head_cls.h_size=200 dset.filter_activation="tanh" dset.augment_factor=2 dset.scale_low=-1 dset.scale_high=1 dset.load_augmented=True dset.augment_path="experiments/augmented_data/augment_zoo_cifar_tanh_-1_1" 
```
### Our model, ReLU, with augment [1,10000]:
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnps_inv nfnet.head_cls.h_size=200 dset.filter_activation="relu" dset.augment_factor=2 dset.scale_low=1 dset.scale_high=10000 dset.load_augmented=True dset.augment_path="experiments/augmented_data/augment_zoo_cifar_10000" 
```
### Our model, Tanh, no augment: 
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnps_inv_tanh nfnet.model_type="S_tanh_abs_gen" nfnet.head_cls.h_size=200 dset.filter_activation="tanh" dset.augment_factor=1 
```
### For other baseline, no augment:
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=np_inv  nfnet.head_cls.h_size=1000 dset.filter_activation="tanh" dset.augment_factor=1 
```
### For other baseline, with augment:
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnp_inv nfnet.head_cls.h_size=1000 dset.filter_activation="tanh" dset.augment_factor=2 dset.scale_low=-1 dset.scale_high=1 dset.load_augmented=True dset.augment_path="experiments/augmented_data/augment_zoo_cifar_tanh_-1_1" 
```
Choices for nfnet:
* `hnp_inv` : HNP
* `np_inv` : NP
* `stat` : STATNN
* `hnps_inv` : MNF (ours) for ReLU case
* `hnps_inv_tanh` : MNF (ours) for Tanh case

To test with other augmented dataset, change the scale_low, scale_high and augment_path accordingly. 

* Dataset for ReLU: "experiments/augmented_data/augment_zoo_cifar_{scale_high}"  where scale_high can be chosen from 10, 100, 1000, 10000 (scale_low is always 1). Remember to set dset.filter_activation="relu"
* Dataset for Tanh: "experiments/augmented_data/augment_zoo_cifar_tanh_-1_1"  (scale_low = -1, scale_high = 1). Remember to set dset.filter_activation="tanh"

## Classify NFNs

Datasets of SIREN weights trained on the MNIST, FashionMNIST, and CIFAR image datasets, available from [this link](https://drive.google.com/drive/folders/15CdOTPWHqDcS4SwbIdm100rXkTYZPcC5?usp=sharing). Download these datasets into `./experiments/data` and untar them:
```sh
tar -xvf siren_mnist_wts.tar  # creates a folder siren_mnist_wts/
tar -xvf siren_fashion_wts.tar  # creates a folder siren_fashion_wts/
tar -xvf siren_cifar_wts.tar  # creates a folder siren_cifar_wts/
```


- CIFAR: 
```
python -m experiments.launch_classify_siren +setup=cifar_hnps warmup_steps=20000
```
- MNIST: 
```
python -m experiments.launch_classify_siren +setup=mnist_hnps warmup_steps=10000
```
- Fashion MNIST: 
```
python -m experiments.launch_classify_siren +setup=fashion_hnps warmup_steps=20000
```
choices for +setup:  {dataset}_{model} 

dataset: `cifar` , `mnist`, `fashion`

model: 
* `hnp` : HNP
* `hnps`: MNF (ours)
* `np`: NP
* `mlp`: MLP 

## Weight space style editing (Stylize INRs)

- MNIST:
```
python -m experiments.launch_stylize_siren +setup=mnist nfnet=equiv_io_hnps
```
- CIFAR: 
```
python -m experiments.launch_stylize_siren +setup=cifar nfnet=equiv_io_hnps
```
Choices for nfnet: 
* `equiv`: HNP
* `equiv_io`: NP
* `equiv_io_hnps` : MNF (ours)
* `mlp`: MLP