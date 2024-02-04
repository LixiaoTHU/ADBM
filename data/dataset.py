import torch, os
from torchvision import datasets, transforms
from .autoaugment import *


def get_data_scaler(args):
  """Data normalizer. Assume data are always in [0, 1]."""
  if args.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(args):
  """Inverse data normalizer."""
  if args.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x
  

def get_dataset(args):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  root = "/home/user/data4/diffusion/datasets/"
  if args.dataset == 'CIFAR10':

    transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.CIFAR10(os.path.join(root, 'CIFAR10'), train = True, transform = transform_train, download = True)
    testset = datasets.CIFAR10(os.path.join(root, 'CIFAR10'), train = False, transform = transform_test, download = True)
  
  elif args.dataset == 'CIFAR100':

    transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.CIFAR100(os.path.join(root, 'CIFAR100'), train = True, transform = transform_train, download = True)
    testset = datasets.CIFAR100(os.path.join(root, 'CIFAR100'), train = False, transform = transform_test, download = True)

  elif args.dataset == "SVHN":
    transform_train = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.SVHN(os.path.join(root, 'SVHN'), split = "train", transform = transform_train, download = True)
    testset = datasets.SVHN(os.path.join(root, 'SVHN'), split = "test", transform = transform_test, download = True)

  elif args.dataset == "TinyImageNet":
    transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200/train'), transform = transform_train)
    testset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200/val'), transform = transform_test)


  else:
    raise NotImplementedError(
      f'Dataset {args.dataset} not yet supported.')


  return trainset, testset

def get_cls_dataset(args):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  root = "/home/user/data4/diffusion/datasets/"
  if args.dataset == 'CIFAR10':
    
    if args.use_aa:
      transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      CIFARPolicy(),
                      transforms.ToTensor()
                      ])
    else:
      transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor()
                      ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.CIFAR10(os.path.join(root, 'CIFAR10'), train = True, transform = transform_train, download = True)
    testset = datasets.CIFAR10(os.path.join(root, 'CIFAR10'), train = False, transform = transform_test, download = True)
  
  elif args.dataset == 'CIFAR100':

    if args.use_aa:
      transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      CIFARPolicy(),
                      transforms.ToTensor()
                      ])
    else:
      transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(15),
                      transforms.ToTensor()
                      ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.CIFAR100(os.path.join(root, 'CIFAR100'), train = True, transform = transform_train, download = True)
    testset = datasets.CIFAR100(os.path.join(root, 'CIFAR100'), train = False, transform = transform_test, download = True)

  elif args.dataset == "SVHN":
    if args.use_aa:
      transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      SVHNPolicy(),
                      transforms.ToTensor()
                      ])
    else:
      transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor()
                    ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.SVHN(os.path.join(root, 'SVHN'), split = "train", transform = transform_train, download = True)
    testset = datasets.SVHN(os.path.join(root, 'SVHN'), split = "test", transform = transform_test, download = True)
    
  elif args.dataset == "TinyImageNet":
    if args.use_aa:
      transform_train = transforms.Compose([
                      transforms.RandomCrop(64, padding=4),
                      transforms.RandomHorizontalFlip(),
                      ImageNetPolicy(),
                      transforms.ToTensor()
                      ])
    else:
      transform_train = transforms.Compose([
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ])
    transform_test = transforms.Compose([
                    transforms.ToTensor()
                    ])

    trainset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200/train'), transform = transform_train)
    testset = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200/val'), transform = transform_test)


  else:
    raise NotImplementedError(
      f'Dataset {args.dataset} not yet supported.')


  return trainset, testset