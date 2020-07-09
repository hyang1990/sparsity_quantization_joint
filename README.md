# Automatic Neural Network Compression by Sparsity-Quantization Joint Learning
Code for paper "Automatic Neural Network Compression by Sparsity-Quantization Joint Learning: A Constrained Optimization-based Approach" (https://arxiv.org/pdf/1910.05897.pdf)
```
@inproceedings{yang2019learning,
  title={Automatic Neural Network Compression by Sparsity-Quantization Joint Learning: A Constrained Optimization-based Approach},
  author={Yang, Haichuan and Gui, Shupeng and Zhu, Yuhao and Liu, Ji},
  booktitle={CVPR},
  year={2020}
}
```
## Prerequisites

```
Python (3.6)
PyTorch 1.0
```

## Training and testing


### example

#### Training (prune & quantize) on MNIST with LeNet-5 (compression rate ~ 2000x):

```
python train.py --dataset mnist --arch caffelenet -b 256 --projint 0 --dualint 0 --log_interval -1 --nnz_budget 0.0054 --bit_budget 3 --lr 0.1 --lr_sched cos --warmupT 10 --epoch 120 --logdir log/mnist_letnet5_2000x --rho 0.05 --wd 5e-4 --prox --momentum 0.93 --pretrain ./pretrained/mnist_lenet5.pt
```

#### Training (prune & quantize) on ImageNet with AlexNet (compression rate ~ 40x):

```
python train.py --dataset imagenet --arch alexnet --data_dir ILSVRC_CLS/ -b 256 --projint 0 --dualint 0 --log_interval 100 --nnz_budget 0.12 --bit_budget 6 --lr 0.01 --lr_sched cos --warmupT 0 --epoch 90 --logdir log/imagenet_alex1 --rho 1e-3 --wd 1e-4 --prox --momentum 0.9 --pretrain pytorch --dp 0.5
```

#### We can use the trained model to further compress AlexNet:

~ 60x
```
python train.py --dataset imagenet --arch alexnet --data_dir ILSVRC_CLS/ -b 256 --projint 0 --dualint 0 --log_interval 100 --nnz_budget 0.1 --bit_budget 5 --lr 0.01 --lr_sched cos --warmupT 0 --epoch 90 --logdir log/imagenet_alex2 --rho 1e-3 --wd 1e-4 --prox --momentum 0.9 --pretrain log/imagenet_alex1/model_latest.pt --dp 0.5
```
~ 100x
```
python train.py --dataset imagenet --arch alexnet --data_dir ILSVRC_CLS/ -b 256 --projint 0 --dualint 0 --log_interval 100 --nnz_budget 0.08 --bit_budget 4 --lr 0.01 --lr_sched cos --warmupT 0 --epoch 90 --logdir log/imagenet_alex3 --rho 1e-3 --wd 1e-4 --prox --momentum 0.9 --pretrain log/imagenet_alex2/model_latest.pt --dp 0.5
```

Fine-tuning on the compressed model:
```
python finetune.py --dataset imagenet --arch alexnet --data_dir ILSVRC_CLS/ -b 256 --log_interval 100 --lr 0.001 --lr_sched cos --warmupT 0 --epoch 60 --logdir log/imagenet_alex3/ft --wd 1e-4 --prox --momentum 0.9 --pretrain ./log/imagenet_alex3/model_latest.pt --dp 0.5
```


### usage

#### Iterative training for joint pruning and quantization

```
usage: train.py [-h] [--dataset DATASET] [--data_dir DATA_DIR] [--arch ARCH]
                [-j N] [--epochs N] [--warmupT WARMUPT] [--start_epoch N]
                [-b N] [--lr LR] [--lr_sched LR_SCHED]
                [--bit_budget BIT_BUDGET] [--nnz_budget NNZ_BUDGET]
                [--momentum M] [--weight_decay W] [--rho RHO] [--gclip GCLIP]
                [--projint PROJINT] [--dualint DUALINT] [-e]
                [--pretrain PRETRAIN] [--logdir LOGDIR]
                [--save-every SAVE_EVERY] [--mgpu]
                [--log_interval LOG_INTERVAL] [--seed SEED] [--eval_tr]
                [--prox] [--dp DP] [--quant] [--prune] [--bwlb BWLB]
                [--bits_epoch BITS_EPOCH] [--kdtemp KDTEMP] [--optim OPTIM]
                [--fixedbits]

Prune-Quant training in pytorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset used in the experiment
  --data_dir DATA_DIR   dataset dir in this machine
  --arch ARCH, -a ARCH
  -j N, --workers N     number of data loading workers
  --epochs N            number of total epochs to run
  --warmupT WARMUPT     number of total iterations for warmup
  --start_epoch N       manual epoch number (useful on restarts)
  -b N, --batch_size N  mini-batch size (default: 128)
  --lr LR, --learning-rate LR
                        initial learning rate
  --lr_sched LR_SCHED   lr scheduler
  --bit_budget BIT_BUDGET
                        bit budget ({1,2...,8})
  --nnz_budget NNZ_BUDGET
                        number of nonzero budget (0.0~1.0)
  --momentum M          momentum
  --weight_decay W, --wd W
                        weight decay
  --rho RHO             admm hyperparameter rho
  --gclip GCLIP         gradient clip
  --projint PROJINT     how many batches to wait before sparse projection of
                        primal weights
  --dualint DUALINT     how many batches to wait before updating duplicate and
                        dual weights
  -e, --evaluate        evaluate model on validation set
  --pretrain PRETRAIN   file to load pretrained model
  --logdir LOGDIR       The directory used to save the trained models
  --save-every SAVE_EVERY
                        Saves checkpoints at every specified number of epochs
  --mgpu                enable using multiple gpus
  --log_interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  --seed SEED           random seed
  --eval_tr             evaluate training set
  --prox                use proximal op for primal update
  --dp DP               dropout rate
  --quant               only perform quantization
  --prune               only perform pruning
  --bwlb BWLB           the lower bound of bitwidth
  --bits_epoch BITS_EPOCH
                        maximum epochs allowing update bits
  --kdtemp KDTEMP       knowledge distillation temperature
  --optim OPTIM         optimizer to use
  --fixedbits           use fixed bitwidth
```


#### Making additional fine-tuning (not necessary for MNIST and CIFAR-10)

```
usage: finetune.py [-h] [--dataset DATASET] [--data_dir DATA_DIR]
                   [--arch ARCH] [-j N] [--epochs N] [--warmupT WARMUPT]
                   [--start_epoch N] [-b N] [--lr LR] [--lr_sched LR_SCHED]
                   [--momentum M] [--weight_decay W] [--gclip GCLIP] [-e]
                   --pretrain PRETRAIN [--logdir LOGDIR]
                   [--save-every SAVE_EVERY] [--mgpu]
                   [--log_interval LOG_INTERVAL] [--seed SEED] [--eval_tr]
                   [--prox] [--dp DP] [--quant] [--weightbits WEIGHTBITS]
                   [--optim OPTIM] [--kdtemp KDTEMP]

Prune-Quant finetune in pytorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset used in the experiment
  --data_dir DATA_DIR   dataset dir in this machine
  --arch ARCH, -a ARCH
  -j N, --workers N     number of data loading workers
  --epochs N            number of total epochs to run
  --warmupT WARMUPT     number of total iterations for warmup
  --start_epoch N       manual epoch number (useful on restarts)
  -b N, --batch_size N  mini-batch size (default: 128)
  --lr LR, --learning-rate LR
                        initial learning rate
  --lr_sched LR_SCHED   lr scheduler
  --momentum M          momentum
  --weight_decay W, --wd W
                        weight decay
  --gclip GCLIP         gradient clip
  -e, --evaluate        evaluate model on validation set
  --pretrain PRETRAIN   file to load pretrained model
  --logdir LOGDIR       The directory used to save the trained models
  --save-every SAVE_EVERY
                        Saves checkpoints at every specified number of epochs
  --mgpu                enable using multiple gpus
  --log_interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  --seed SEED           random seed
  --eval_tr             evaluate training set
  --prox                use proximal op for primal update
  --dp DP               dropout rate
  --quant               only perform quantization
  --weightbits WEIGHTBITS
                        mannual weightbits
  --optim OPTIM         optimizer to use
  --kdtemp KDTEMP       knowledge distillation temperature
```