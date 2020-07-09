import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import os


def get_data_loaders(data_dir, batch_size=128, val_batch_size=128, num_workers=0, nsubset=-1,
                     normalize=None):
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if nsubset > 0:
        rand_idx = torch.randperm(len(train_dataset))[:nsubset]
        print('use a random subset of data:')
        print(rand_idx)
        train_sampler = SubsetRandomSampler(rand_idx)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # use 10K training data to see the training performance
    train_loader4eval = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        sampler=SubsetRandomSampler(torch.randperm(len(train_dataset))[:10000]))

    return train_loader, val_loader, train_loader4eval
