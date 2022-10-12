import numpy as np
import torchvision.transforms as T
import torchvision.datasets as datasets
# import torchvision.IterableDataset as IterableDataset
from torch.utils.data import ConcatDataset, Subset
from utils.utils0 import logging, timeLog

import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
#----------------------------------------------------------
def get_ds_attr(dataset):
   imgsz = 32; channels = 3
   if dataset == 'CIFAR10' or dataset == 'SVHN':
      nclass = 10
   elif dataset == 'CIFAR100':
      nclass = 100
   elif dataset == 'ImageNet': 
      nclass = 1000; imgsz = 64
   elif dataset == 'ImageNet64': 
      nclass = 1000; imgsz = 64
   elif dataset == 'MNIST':
      nclass = 10; channels = 1
   elif dataset  == 'FashionMNIST':
      nclass = 10; channels = 1
   elif dataset  == 'EMNIST':
      nclass = 10; channels = 1   
   elif dataset.endswith('64'):
      nclass = 1; imgsz = 64
      if 'brlr' in dataset or 'twbg' in dataset:
         nclass = 2
   elif dataset.endswith('256'):
      nclass = 1; imgsz = 256
      if 'brlr' in dataset or 'twbg' in dataset:
         nclass = 2
   elif dataset.endswith('128'):
      nclass = 1; imgsz = 128
      if 'brlr' in dataset or 'twbg' in dataset:
         nclass = 2
   elif dataset.endswith('1024'):
      nclass = 1; imgsz = 1024         
   else:
      raise ValueError('Unknown dataset: %s ...' % dataset)

   return { "nclass": nclass, "image_size": imgsz, "channels": channels }

#----------------------------------------------------------
def gen_lsun_balanced(dataroot, nms, tr, indexes):
   sub_dss = []
   for i,nm in enumerate(nms):
      sub_dss += [Subset(datasets.LSUN(dataroot, classes=[nm], transform=tr), indexes)]
   return ConcatUniClassDataset(sub_dss)

#----------------------------------------------------------
# Concatenate uni-class datasets into one dataset. 
class ConcatUniClassDataset:
   def __init__(self, dss):
      self.dss = dss
      self.top = [0]
      num = 0
      for ds in self.dss:
         num += len(ds)
         self.top += [num]
         
   def __len__(self):
      # print(self.top[len(self.top)-1])
      return self.top[len(self.top)-1]
         
   def __getitem__(self, index):
      cls = -1
      for i,top in enumerate(self.top):
         # print('i {} and top{} and index{}'.format(i,top,index))
         if index < top:
            cls = i-1
            break
      if cls < 0:
         raise IndexError
      # print((self.dss[cls])[index-top][0].shape)
      return ((self.dss[cls])[index-top][0], cls)
         
#----------------------------------------------------------
def get_ds(dataset, dataroot, is_train, do_download, do_augment):
   tr = get_tr(dataset, is_train, do_augment)
   if dataset == 'SVHN':
      if is_train:
         train_ds = datasets.SVHN(dataroot, split='train', transform=tr, download=do_download)
         extra_ds = datasets.SVHN(dataroot, split='extra', transform=tr, download=do_download)
         return ConcatDataset([train_ds, extra_ds])
      else:
         return datasets.SVHN(dataroot, split='test', transform=tr, download=do_download) 
   elif dataset == 'MNIST':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)    
   elif dataset =='CIFAR10':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)
   elif dataset =='CIFAR100':
          return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)
   elif dataset =='FashionMNIST':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)
   elif dataset.startswith('ImageNet'):
      return ImageFolder(dataroot,transform=tr)   
   elif dataset == 'EMNIST':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download,split='byclass')
   elif dataset.startswith('celebahq_'):
      return ImageFolder(dataroot,transform=tr)               
   elif dataset.startswith('lsun_'):
      if  dataset.endswith('64'):
         nm = dataset[len('lsun_'):len(dataset)-len('64')] + ('_train' if is_train else '_val')
      elif dataset.endswith('256'):
         nm = dataset[len('lsun_'):len(dataset)-len('256')] + ('_train' if is_train else '_val')
      elif dataset.endswith('128'):
         nm = dataset[len('lsun_'):len(dataset)-len('128')] + ('_train' if is_train else '_val')   
      if nm is None:
         raise ValueError('Unknown dataset set: %s ...' % dataset)
      else:
         if nm.startswith('brlr'):
            indexes = list(range(1300000)) if is_train else list(range(1300000,1315802))
            return gen_lsun_balanced(dataroot, ['bedroom_train', 'living_room_train'], tr, indexes)
         elif nm.startswith('twbg'):
            indexes = list(range(700000)) if is_train else list(range(700000,708264))
            return gen_lsun_balanced(dataroot, ['tower_train', 'bridge_train'], tr, indexes)
         else:
            timeLog('Loading LSUN %s ...' % nm)
            return datasets.LSUN(dataroot, classes=[ nm ], transform=tr)
   else:
      raise ValueError('Unknown dataset: %s ...' % dataset)

#----------------------------------------------------------
def to_pm1(input):
   return input*2-1

#----------------------------------------------------------
def get_tr(dataset, is_train, do_augment):
   if dataset == 'ImageNet':  
      tr = T.Compose([ T.Resize(256), T.CenterCrop(224) ])
   elif dataset == 'ImageNet64':  
      tr = T.Compose([ T.Resize(64), T.CenterCrop(64) ])  
   elif dataset == 'MNIST':
      tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32
   elif dataset == 'FashionMNIST':
      tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32
   elif dataset == 'EMNIST':
      tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32   
   elif dataset.endswith('64'):
      tr = T.Compose([ T.Resize(64), T.CenterCrop(64) ])
   elif dataset.endswith('256'):
      tr = T.Compose([ T.Resize(256), T.CenterCrop(256) ])
   elif dataset.endswith('128'):
      tr = T.Compose([ T.Resize(128), T.CenterCrop(128) ]) 
   elif dataset.endswith('1024'):
      tr = T.Compose([ T.Resize(1024), T.CenterCrop(1024) ])     
   else:
     tr = T.Compose([ ])
     if do_augment:
        tr = T.Compose([
              tr, 
              T.Pad(4, padding_mode='reflect'),
              T.RandomHorizontalFlip(),
              T.RandomCrop(32),
        ])

   return T.Compose([ tr, T.ToTensor(), to_pm1 ])    

# class ImageFolder(IterableDataset):
#     def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
#                  repeat=1, cache='none',transform=None):
#         self.repeat = repeat
#         self.cache = cache
#         self.tr= transform
#         if split_file is None:
#             filenames = sorted(os.listdir(root_path))
#         else:
#             with open(split_file, 'r') as f:
#                 filenames = json.load(f)[split_key]
#         if first_k is not None:
#             filenames = filenames[:first_k]

#         self.files = []
#         for filename in filenames:
#             file = os.path.join(root_path, filename)

#             if cache == 'none':
#                 self.files.append(Image.open(file).convert('RGB'))

#     def __len__(self):
#       #   print(len(self.files) * self.repeat)
#         return len(self.files) * self.repeat

#     def __iter__(self):
#         if self.cache == 'none':
#             # return transforms.ToTensor()(Image.open(x).convert('RGB'))
#             return iter(self.tr(self.files))
class ImageFolder(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none',transform=None):
        self.repeat = repeat
        self.cache = cache
        self.tr= transform
        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(self.tr(Image.open(file).convert('RGB')))
               #  self.files.append(transforms.ToTensor()(
               #      Image.open(file).convert('RGB')))

    def __len__(self):
      #   print(len(self.files) * self.repeat)
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
      #   print(idx)
        if self.cache == 'none':
            # return transforms.ToTensor()(Image.open(x).convert('RGB'))
            return (self.tr(Image.open(x).convert('RGB')),0)

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x
