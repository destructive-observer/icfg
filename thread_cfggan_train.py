from tkinter import N
from numpy.core.records import fromfile
import torch
import torch.random
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.init import normal_
import numpy as np

from data import get_ds_attr, get_ds
import thread_netdef
from thread_cfggan import cfggan
from utils.utils0 import raise_if_absent, add_if_absent_, logging, timeLog, raise_if_nonpositive_any
import utils_biggan

CDCGANx = 'cdcganx'
DCGANx = 'dcganx'
DCGAN4 = 'dcgan4'
Resnet4 = 'resnet4'
Resnet3 = 'resnet3'
FCn = 'fcn'
Balance2 = 'balance2'
Balance = 'balance'
Resnet256 = 'resnet256'
Resnet128 = 'resnet128'
Resnet1024 = 'resnet1024'
Biggan='biggan'
cudnn.benchmark = True
Resnet_L='resnet_large'


#----------------------------------------------------------
def proc(rank, gpu,opt): 
   check_opt_(opt)
   torch.manual_seed(opt.seed+ rank)
   np.random.seed(opt.seed+ rank)
   if torch.cuda.is_available():
      torch.cuda.manual_seed_all(opt.seed)
   torch.cuda.manual_seed(opt.seed + rank)
   torch.cuda.manual_seed_all(opt.seed + rank)
   torch.backends.cudnn.benchmark = True
   opt.device = torch.device('cuda:{}'.format(gpu))
   ds_attr = get_ds_attr(opt.dataset)
   opt.image_size = ds_attr['image_size']
   opt.channels = ds_attr['channels']

   from_file = None
   saved = None
   if opt.dataset =='ImageNet':
     I_set = 'I64'     
     opt.resolution = utils_biggan.imsize_dict[I_set]
     opt.n_classes = utils_biggan.nclass_dict[I_set]
     opt.n_classes = 10
     opt.G_activation = utils_biggan.activation_dict[opt.G_nl]
     opt.D_activation = utils_biggan.activation_dict[opt.D_nl]
   elif opt.dataset =='CIFAR10':
     opt.resolution = ds_attr['image_size']
     opt.n_classes =  ds_attr['nclass']
     opt.G_activation = utils_biggan.activation_dict[opt.G_nl]
     opt.D_activation = utils_biggan.activation_dict[opt.D_nl]
   elif opt.dataset.startswith('lsun_'):
     opt.resolution = ds_attr['image_size']
     opt.n_classes =  ds_attr['nclass']
     opt.G_activation = utils_biggan.activation_dict[opt.G_nl]
     opt.D_activation = utils_biggan.activation_dict[opt.D_nl]
   else:
     opt.n_classes=None
     opt.resolution =None
     opt.G_activation=None
     opt.D_activation=None
   if opt.saved:
      from_file = torch.load(opt.saved, map_location=opt.device if torch.cuda.is_available() else 'cpu')
      saved = opt.saved
      batch_size = opt.batch_size
      cfg_N = opt.batch_size * opt.cfg_x_epo
      logging('WARNING: from file is begin')
      opt = from_file['opt']
      opt.batch_size = batch_size
      opt.cfg_N = cfg_N
   print( ds_attr['nclass'])
   print(opt.n_classes)
  #  opt.n_classes = ds_attr['nclass']   
   # print(opt)
   def d_config(requires_grad):  # D
      if opt.d_model == DCGANx:
         return thread_netdef.Discriminator(opt.d_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.d_depth, 
                                do_bias=not opt.do_no_bias)
      
      elif opt.d_model == Biggan:
        return thread_netdef.biggan_D(resolution=opt.resolution,D_ch=opt.g_dim,D_attn=str(opt.g_dim))       
      else:
         raise ValueError('d_model must be dcganx.')
   def g_config(requires_grad):  # G
      if opt.g_model == DCGANx:
             return thread_netdef.Generator(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.g_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.d_model == Biggan:
            return thread_netdef.biggan_G(resolution=opt.resolution,G_ch=opt.g_dim,G_attn=str(opt.g_dim))       
      else:
         raise ValueError('g_model must be dcganx or fcn.')
   def z_gen(num):
      return normal_(torch.Tensor(num, opt.z_dim), std=opt.z_std),torch.zeros(num).random_(0, 10)
   def z_y_gen_function_new(num,dim_z, nclasses):
      return normal_(torch.Tensor(num, opt.z_dim), std=opt.z_std),torch.zeros(num).random_(0, 10)
   def z_y_gen_function(num,num_labels):
      one_hot_labels = torch.FloatTensor(num, num_labels)
      one_hot_labels.zero_()
      rand_y = torch.from_numpy(
                np.random.randint(0, num_labels, size=(num,1)))
      one_hot_labels.scatter_(1, rand_y.view(num,1), 1)
      return normal_(torch.Tensor(num, opt.z_dim), std=opt.z_std),one_hot_labels
   z_y_gen = None
   # def z_y_gen(num,dim,n_classes,device):
   #    if n_classes is not None:
   #       return utils_biggan.prepare_z_y(num,opt.z_dim,opt.n_classes,device=device)
   #    else:
   #       return z_gen(num)
   if opt.dataset == 'ImageNet' and opt.model =='biggan':
   #   z_y_gen = utils_biggan.prepare_z_y
     opt.n_classes = 10
     z_y_gen = z_y_gen_function_new
     ds = utils_biggan.get_data_loaders(**{**vars(opt), 'batch_size': opt.batch_size,
                                      'start_itr': 0,'dataset':'I64','distributed':True})
    #  loader = loader[0]
     train_sampler = torch.utils.data.distributed.DistributedSampler(ds,
                                                                    num_replicas=opt.world_size,
                                                                    rank=rank)
     loader = torch.utils.data.DataLoader(ds,
                                               batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=False,
                                               sampler=train_sampler,
                                               drop_last = True)
   elif opt.dataset == 'CIFAR10' and opt.model =='biggan':
     z_y_gen = utils_biggan.prepare_z_y
     ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
     timeLog('#train = %d' % len(ds))     
     loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
                       num_workers=opt.num_workers, 
                       pin_memory=torch.cuda.is_available())
   elif opt.dataset == 'lsun_bedroom64' and opt.model =='biggan':
   #   z_y_gen = utils_biggan.prepare_z_y
     print(opt.n_classes)
     z_y_gen = z_y_gen_function_new
     ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
     timeLog('#train = %d' % len(ds))     
     loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
                       num_workers=opt.num_workers, 
                       pin_memory=torch.cuda.is_available())
   elif opt.dataset == 'lsun_church64' and opt.model =='biggan':
     z_y_gen = utils_biggan.prepare_z_y
     ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
     timeLog('#train = %d' % len(ds))     
     loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
                       num_workers=opt.num_workers, 
                       pin_memory=torch.cuda.is_available())
   elif opt.dataset == 'MNIST' and opt.model ==CDCGANx:
     z_y_gen = z_y_gen_function
     ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
     timeLog('#train = %d' % len(ds))     
     loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
                       num_workers=opt.num_workers, 
                       pin_memory=torch.cuda.is_available()) 
   elif opt.dataset == 'CIFAR10' and opt.model ==CDCGANx:
     z_y_gen = z_y_gen_function
     ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
     timeLog('#train = %d' % len(ds))     
     loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
                       num_workers=opt.num_workers, 
                       pin_memory=torch.cuda.is_available()) 
   elif opt.dataset == 'CIFAR100' and opt.model ==CDCGANx:
     z_y_gen = z_y_gen_function
     ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
     timeLog('#train = %d' % len(ds))     
     loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
                       num_workers=opt.num_workers, 
                       pin_memory=torch.cuda.is_available())  
   else:
     ds = get_ds(opt.dataset, opt.dataroot, is_train=True, do_download=opt.do_download, do_augment=opt.do_augment)
     timeLog('#train = %d' % len(ds))
     train_sampler = torch.utils.data.distributed.DistributedSampler(ds,
                                                                    num_replicas=opt.world_size,
                                                                    rank=rank)
     loader = torch.utils.data.DataLoader(ds,
                                               batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=False,
                                               sampler=train_sampler,
                                               drop_last = True)     
    #  loader = DataLoader(ds, opt.batch_size, shuffle=True, drop_last=True, 
    #                    num_workers=opt.num_workers, 
    #                    pin_memory=torch.cuda.is_available())
     opt.n_classes = None  
   cfggan(opt, d_config, g_config, z_gen, loader,from_file,saved,z_y_gen)

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, ['d_model','d_dim','d_depth','z_dim','z_std','g_model','g_dim','g_depth','dataset','dataroot','num_workers','batch_size','norm_type'], 'cfggan_train')
   add_if_absent_(opt, ['do_no_bias','do_augment'], False)
   add_if_absent_(opt, ['do_download'], True)
   add_if_absent_(opt, ['seed'], 1)

   raise_if_nonpositive_any(opt, ['d_depth','g_depth','d_dim','z_dim','z_std','g_dim','batch_size'])  
