import sys
import os
import argparse

from utils.utils0 import raise_if_absent, add_if_absent_, set_if_none, raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults
from cfggan_train import proc as cfggan_train
from cfggan_train import DCGANx, Resnet4, FCn, Resnet3,DCGAN4,Balance,Resnet256,Resnet128,Resnet1024,Balance2,CDCGANx
from cfggan import RMSprop_str
import utils_biggan
Biggan = 'biggan'
Resnet_L='resnet_large'
FC2 = 'fc2'
MNIST = 'MNIST'
SVHN = 'SVHN'
Bedroom64 = 'lsun_bedroom64'
ImageNet = 'ImageNet'
ImageNet64 = 'ImageNet64'
Church64 = 'lsun_church_outdoor64'
Brlr64 = 'lsun_brlr64'
Twbg64 = 'lsun_twbg64'
CIFAR10 = 'CIFAR10'
CIFAR100 = 'CIFAR100'
Tower64='lsun_tower64'
FashionMNIST = 'FashionMNIST'
EMNIST='EMNIST'
Bedroom128 = 'lsun_bedroom128'
Bedroom256 = 'lsun_bedroom256'
CelebaHQ = 'celebahq_1024'
Celeba128 = 'celebahq_128'
Celeba256 = 'celebahq_256'
#----------------------------------------------------------
def add_args_(parser):
   #---  proc
   parser.add_argument('--seed', type=int, default=1, help='Random seed.')   

   parser.add_argument('--dataset', type=str, choices=[MNIST, SVHN, Bedroom64, Church64,CIFAR10,CIFAR100,Tower64,Brlr64,Twbg64,FashionMNIST,EMNIST,Bedroom256,Bedroom128,CelebaHQ,Celeba128,Celeba256,ImageNet,ImageNet64], required=True, help='Dataset.')
   parser.add_argument('--dataroot', type=str, default='.')
   parser.add_argument('--model', type=str, choices=[CDCGANx,DCGANx,Resnet4,FC2,Resnet3,DCGAN4,Balance,Resnet256,Resnet128,Resnet1024,Balance2,Resnet_L,Biggan], help='Model.')   
   parser.add_argument('--norm_type', type=str, default='bn', choices=['bn','none'], help="'bn': batch normalization, 'none': no normalization")   
   parser.add_argument('--batch_size', type=int, default=64, help='Number of images for training.')   
   parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for retrieving images.')   

   #---  cfggan
   parser.add_argument('--cfg_T', type=int, help='T for ICFG.')
   parser.add_argument('--cfg_U', type=int, default=1, help='U (discriminator update frequency) for ICFG.')
   parser.add_argument('--cfg_N', type=int, default=640, help='N (number of generated examples used for approximator training).')  
   parser.add_argument('--num_stages', type=int, default=10000, help='Number of stages.')   
   parser.add_argument('--cfg_eta', type=float, help='Generator step-size eta.')
   parser.add_argument('--lr', type=float, help='Learning rate used for training discriminator and approximator.')
   parser.add_argument('--lr_d', type=float, help='Learning rate used for training discriminator and approximator.')
   parser.add_argument('--lr_g', type=float, help='Learning rate used for training discriminator and approximator.')
   parser.add_argument('--cfg_x_epo', type=int, default=10, help='Number of epochs for approximator training.')
      
   parser.add_argument('--gen', type=str, help='Pathname for saving generated images.')
   parser.add_argument('--real_path', type=str, help='Pathname for saving real images.')    
   parser.add_argument('--save', type=str, default='', help='Pathname for saving models.') 
   parser.add_argument('--save_interval', type=int, default=-1, help='Interval for saving models. -1: no saving.')    
   parser.add_argument('--gen_interval', type=int, default=50, help='Interval for generating images. -1: no generation.')  
   parser.add_argument('--num_gen', type=int, default=10, help='Number of images to be generated.')     
   parser.add_argument('--gen_nrow', type=int, default=5, help='Number of images in each row when making a collage of generated of images.')
   parser.add_argument('--diff_max', type=float, default=40, help='Stop training if |D(real)-D(gen)| exceeds this after passing the initial starting-up phase.')
   parser.add_argument('--lamda',type=float ,default='0', help='0 stands for no constraint or others stands for lamda value')
   parser.add_argument('--scale',type=float ,default='0.1', help='scale for the eposion, value is [0,1]')
   parser.add_argument('--gptype', type=int, default='0', help='0-0 centered 1-1 centered 2-newtype.') 
   parser.add_argument('--alpha',type=float ,default='1', help='use w distance to regulation regression')
   parser.add_argument('--verbose', action='store_true', help='If true, display more info.')   
   parser.add_argument('--saved', type=str, default='', help='Pathname for the saved model.')

   return parser

def prepare_parser(parser):
  ### Dataset/Dataloader stuff ###
#   parser.add_argument(
#     '--dataset', type=str, default='I128_hdf5',
#     help='Which Dataset to train on, out of I128, I256, C10, C100;'
#          'Append "_hdf5" to use the hdf5 version for ISLVRC '
#          '(default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=True,
    help='Augment with random crops and flips (default: %(default)s)')
#   parser.add_argument(
#     '--num_workers', type=int, default=8,
#     help='Number of dataloader workers; consider using less for HDF5 '
#          '(default: %(default)s)')
  parser.add_argument(
    '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
    help='Pin data into memory through dataloader? (default: %(default)s)') 
  parser.add_argument(
    '--shuffle', action='store_true', default=True,
    help='Shuffle the data (strongly recommended)? (default: %(default)s)')
  parser.add_argument(
    '--load_in_mem', action='store_true', default=False,
    help='Load all data into memory? (default: %(default)s)')
  parser.add_argument(
    '--use_multiepoch_sampler', action='store_true', default=False,
    help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
  
  
#   ### Model stuff ###
#   parser.add_argument(
#     '--model', type=str, default='BigGAN',
#     help='Name of the model module (default: %(default)s)')
  parser.add_argument(
    '--G_param', type=str, default='SN',
    help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
          ' or None (default: %(default)s)')
  parser.add_argument(
    '--D_param', type=str, default='SN',
    help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
         ' or None (default: %(default)s)')    
  parser.add_argument(
    '--G_ch', type=int, default=64,
    help='Channel multiplier for G (default: %(default)s)')
  parser.add_argument(
    '--D_ch', type=int, default=64,
    help='Channel multiplier for D (default: %(default)s)')
  parser.add_argument(
    '--G_depth', type=int, default=1,
    help='Number of resblocks per stage in G? (default: %(default)s)')
  parser.add_argument(
    '--D_depth', type=int, default=1,
    help='Number of resblocks per stage in D? (default: %(default)s)')
  parser.add_argument(
    '--D_thin', action='store_false', dest='D_wide', default=True,
    help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
  parser.add_argument(
    '--G_shared', action='store_true', default=True,
    help='Use shared embeddings in G? (default: %(default)s)')
  parser.add_argument(
    '--shared_dim', type=int, default=128,
    help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
         '(default: %(default)s)')
  parser.add_argument(
    '--dim_z', type=int, default=120,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--z_var', type=float, default=1.0,
    help='Noise variance: %(default)s)')    
  parser.add_argument(
    '--hier', action='store_true', default=True,
    help='Use hierarchical z in G? (default: %(default)s)')
  parser.add_argument(
    '--cross_replica', action='store_true', default=False,
    help='Cross_replica batchnorm in G?(default: %(default)s)')
  parser.add_argument(
    '--mybn', action='store_true', default=False,
    help='Use my batchnorm (which supports standing stats?) %(default)s)')
  parser.add_argument(
    '--G_nl', type=str, default='relu',
    help='Activation function for G (default: %(default)s)')
  parser.add_argument(
    '--D_nl', type=str, default='relu',
    help='Activation function for D (default: %(default)s)')
  parser.add_argument(
    '--G_attn', type=str, default='64',
    help='What resolutions to use attention on for G (underscore separated) '
         '(default: %(default)s)')
  parser.add_argument(
    '--D_attn', type=str, default='64',
    help='What resolutions to use attention on for D (underscore separated) '
         '(default: %(default)s)')
  parser.add_argument(
    '--norm_style', type=str, default='bn',
    help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
         'ln [layernorm], gn [groupnorm] (default: %(default)s)')
         
#   ### Model init stuff ###
#   parser.add_argument(
#     '--seed', type=int, default=0,
#     help='Random seed to use; affects both initialization and '
#          ' dataloading. (default: %(default)s)')
  parser.add_argument(
    '--G_init', type=str, default='ortho',
    help='Init style to use for G (default: %(default)s)')
  parser.add_argument(
    '--D_init', type=str, default='ortho',
    help='Init style to use for D(default: %(default)s)')
  parser.add_argument(
    '--skip_init', action='store_true', default=False,
    help='Skip initialization, ideal for testing when ortho init was used '
          '(default: %(default)s)')
  
  ### Optimizer stuff ###
  parser.add_argument(
    '--G_lr', type=float, default=5e-5,
    help='Learning rate to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_lr', type=float, default=2e-4,
    help='Learning rate to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B1', type=float, default=0.0,
    help='Beta1 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B1', type=float, default=0.0,
    help='Beta1 to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B2', type=float, default=0.999,
    help='Beta2 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B2', type=float, default=0.999,
    help='Beta2 to use for Discriminator (default: %(default)s)')
    
  ### Batch size, parallel, and precision stuff ###
#   parser.add_argument(
#     '--batch_size', type=int, default=64,
#     help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--G_batch_size', type=int, default=0,
    help='Batch size to use for G; if 0, same as D (default: %(default)s)')
  parser.add_argument(
    '--num_G_accumulations', type=int, default=1,
    help='Number of passes to accumulate G''s gradients over '
         '(default: %(default)s)')  
  parser.add_argument(
    '--num_D_steps', type=int, default=2,
    help='Number of D steps per G step (default: %(default)s)')
  parser.add_argument(
    '--num_D_accumulations', type=int, default=1,
    help='Number of passes to accumulate D''s gradients over '
         '(default: %(default)s)')
  parser.add_argument(
    '--split_D', action='store_true', default=False,
    help='Run D twice rather than concatenating inputs? (default: %(default)s)')
  parser.add_argument(
    '--num_epochs', type=int, default=100,
    help='Number of epochs to train for (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--G_fp16', action='store_true', default=False,
    help='Train with half-precision in G? (default: %(default)s)')
  parser.add_argument(
    '--D_fp16', action='store_true', default=False,
    help='Train with half-precision in D? (default: %(default)s)')
  parser.add_argument(
    '--D_mixed_precision', action='store_true', default=False,
    help='Train with half-precision activations but fp32 params in D? '
         '(default: %(default)s)')
  parser.add_argument(
    '--G_mixed_precision', action='store_true', default=False,
    help='Train with half-precision activations but fp32 params in G? '
         '(default: %(default)s)')
  parser.add_argument(
    '--accumulate_stats', action='store_true', default=False,
    help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
  parser.add_argument(
    '--num_standing_accumulations', type=int, default=16,
    help='Number of forward passes to use in accumulating standing stats? '
         '(default: %(default)s)')        
    
  ### Bookkeping stuff ###  
  parser.add_argument(
    '--G_eval_mode', action='store_true', default=False,
    help='Run G in eval mode (running/standing stats?) at sample/test time? '
         '(default: %(default)s)')
  parser.add_argument(
    '--save_every', type=int, default=2000,
    help='Save every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_save_copies', type=int, default=2,
    help='How many copies to save (default: %(default)s)')
  parser.add_argument(
    '--num_best_copies', type=int, default=2,
    help='How many previous best checkpoints to save (default: %(default)s)')
  parser.add_argument(
    '--which_best', type=str, default='IS',
    help='Which metric to use to determine when to save new "best"'
         'checkpoints, one of IS or FID (default: %(default)s)')
  parser.add_argument(
    '--no_fid', action='store_true', default=False,
    help='Calculate IS only, not FID? (default: %(default)s)')
  parser.add_argument(
    '--test_every', type=int, default=5000,
    help='Test every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_inception_images', type=int, default=50000,
    help='Number of samples to compute inception metrics with '
         '(default: %(default)s)')
  parser.add_argument(
    '--hashname', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config '
         '(default: %(default)s)') 
  parser.add_argument(
    '--base_root', type=str, default='',
    help='Default location to store all weights, samples, data, and logs '
           ' (default: %(default)s)')
  parser.add_argument(
    '--weights_root', type=str, default='weights',
    help='Default location to store weights (default: %(default)s)')
  parser.add_argument(
    '--logs_root', type=str, default='logs',
    help='Default location to store logs (default: %(default)s)')
  parser.add_argument(
    '--samples_root', type=str, default='samples',
    help='Default location to store samples (default: %(default)s)')  
  parser.add_argument(
    '--pbar', type=str, default='mine',
    help='Type of progressbar to use; one of "mine" or "tqdm" '
         '(default: %(default)s)')
  parser.add_argument(
    '--name_suffix', type=str, default='',
    help='Suffix for experiment name for loading weights for sampling '
         '(consider "best0") (default: %(default)s)')
  parser.add_argument(
    '--experiment_name', type=str, default='',
    help='Optionally override the automatic experiment naming with this arg. '
         '(default: %(default)s)')
  parser.add_argument(
    '--config_from_name', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config '
         '(default: %(default)s)')
         
  ### EMA Stuff ###
  parser.add_argument(
    '--ema', action='store_true', default=False,
    help='Keep an ema of G''s weights? (default: %(default)s)')
  parser.add_argument(
    '--ema_decay', type=float, default=0.9999,
    help='EMA decay rate (default: %(default)s)')
  parser.add_argument(
    '--use_ema', action='store_true', default=False,
    help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
  parser.add_argument(
    '--ema_start', type=int, default=0,
    help='When to start updating the EMA weights (default: %(default)s)')
  
  ### Numerical precision and SV stuff ### 
  parser.add_argument(
    '--adam_eps', type=float, default=1e-6,
    help='epsilon value to use for Adam (default: %(default)s)')
  parser.add_argument(
    '--BN_eps', type=float, default=1e-5,
    help='epsilon value to use for BatchNorm (default: %(default)s)')
  parser.add_argument(
    '--SN_eps', type=float, default=1e-6,
    help='epsilon value to use for Spectral Norm(default: %(default)s)')
  parser.add_argument(
    '--num_G_SVs', type=int, default=1,
    help='Number of SVs to track in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SVs', type=int, default=1,
    help='Number of SVs to track in D (default: %(default)s)')
  parser.add_argument(
    '--num_G_SV_itrs', type=int, default=1,
    help='Number of SV itrs in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SV_itrs', type=int, default=1,
    help='Number of SV itrs in D (default: %(default)s)')
  
  ### Ortho reg stuff ### 
  parser.add_argument(
    '--G_ortho', type=float, default=0.0, # 1e-4 is default for BigGAN
    help='Modified ortho reg coefficient in G(default: %(default)s)')
  parser.add_argument(
    '--D_ortho', type=float, default=0.0,
    help='Modified ortho reg coefficient in D (default: %(default)s)')
  parser.add_argument(
    '--toggle_grads', action='store_true', default=True,
    help='Toggle D and G''s "requires_grad" settings when not training them? '
         ' (default: %(default)s)')
  
  ### Which train function ###
  parser.add_argument(
    '--which_train_fn', type=str, default='GAN',
    help='How2trainyourbois (default: %(default)s)')  
  
  ### Resume training stuff
  parser.add_argument(
    '--load_weights', type=str, default='',
    help='Suffix for which weights to load (e.g. best0, copy0) '
         '(default: %(default)s)')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Resume training? (default: %(default)s)')
  
  ### Log stuff ###
  parser.add_argument(
    '--logstyle', type=str, default='%3.3e',
    help='What style to use when logging training metrics?'
         'One of: %#.#f/ %#.#e (float/exp, text),'
         'pickle (python pickle),'
         'npz (numpy zip),'
         'mat (MATLAB .mat file) (default: %(default)s)')
  parser.add_argument(
    '--log_G_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in G? '
         '(default: %(default)s)')
  parser.add_argument(
    '--log_D_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in D? '
         '(default: %(default)s)')
  parser.add_argument(
    '--sv_log_interval', type=int, default=10,
    help='Iteration interval for logging singular values '
         ' (default: %(default)s)') 
   
  return parser

def add_sep_for_dict(split='-', **dict):
    str_list=''
    for k, v in dict.items():
        str_list += k+str(v)+split
    return str_list
#----------------------------------------------x------------
def check_args_(opt):
   if opt.batch_size is None:
      opt.batch_size = 64 # Batch size. 
   opt.z_dim = 100 # Dimensionality of input random vectors.
   opt.z_std = 1.0 # Standard deviation for generating input random vectors.
   opt.approx_redmax = 5
   opt.approx_decay = 0.1
   opt.cfg_N = opt.batch_size * opt.cfg_x_epo
   
   def is_32x32():
      return opt.dataset in [MNIST, SVHN]   
   def is_32x32_monocolor():
      return opt.dataset == MNIST
      
   #***  Setting meta-parameters to those used in the CFG-GAN paper. 
   #---  network architecture, learning rate, and T
   if opt.model is None:
      opt.model = Resnet4 if opt.dataset.endswith('64') else DCGANx
      
   if opt.model == CDCGANx:
      opt.d_model = opt.g_model = CDCGANx
      opt.d_depth = opt.g_depth = 3 if is_32x32() else 4
      opt.d_dim = opt.g_dim = 32 if is_32x32_monocolor() else 64
      # opt.num_class = 10
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)   
   elif opt.model == DCGANx:
      opt.d_model = opt.g_model = DCGANx
      opt.d_depth = opt.g_depth = 3 if is_32x32() else 4
      opt.d_dim = opt.g_dim = 32 if is_32x32_monocolor() else 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Resnet4:
      opt.d_model = opt.g_model = Resnet4
      opt.d_depth = opt.g_depth = 4
      opt.d_dim = opt.g_dim = 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)   
   elif opt.model == FC2:
      opt.d_model = DCGANx
      opt.d_dim = 32 if is_32x32_monocolor() else 64
      opt.d_depth = 3 if is_32x32() else 4
      opt.g_model = FCn; opt.g_depth = 2; opt.g_dim = 512
      set_if_none(opt, 'lr', 0.0001)
      set_if_none(opt, 'cfg_T', 25)
   elif opt.model == Resnet3:
      opt.d_model = opt.g_model = Resnet3
      opt.d_depth = opt.g_depth = 3
      opt.d_dim = opt.g_dim = 32
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == DCGAN4:
      opt.d_model = opt.g_model = DCGANx
      opt.d_depth = opt.g_depth = 3 if is_32x32() else 4
      opt.d_dim = opt.g_dim = 32 if is_32x32_monocolor() else 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Resnet_L:
      opt.d_model = opt.g_model = Resnet4
      opt.d_depth = opt.g_depth = 4
      # opt.d_dim = opt.g_dim = 128
      opt.d_dim = opt.g_dim = 64*2
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)   
   elif opt.model == Balance2:
      opt.d_model = Resnet4
      opt.g_model = DCGANx
      opt.d_depth = 2
      opt.g_depth = 3
      opt.d_dim = 32
      opt.g_dim = 32
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)   
   elif opt.model == Balance:
      opt.d_model = Resnet4
      opt.g_model = DCGANx
      opt.d_depth = 3
      opt.g_depth = 4
      opt.d_dim = 64
      opt.g_dim = 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Resnet256:
      opt.d_model = opt.g_model = Resnet4
      opt.d_depth = 6
      opt.g_depth = 5
      opt.d_dim = opt.g_dim = 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)    
   elif opt.model == Resnet128:
      opt.d_model = opt.g_model = Resnet4
      opt.d_depth = 5
      opt.g_depth = 4
      opt.d_dim = opt.g_dim = 128
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Resnet1024:
      opt.d_model = Resnet4
      opt.g_model = Resnet1024
      opt.d_depth = 8
      opt.g_depth = 7
      opt.d_dim = opt.g_dim = 32
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Biggan:
      opt.d_model = Biggan
      opt.g_model = Biggan
      opt.d_depth = 8
      opt.g_depth = 7
      opt.d_dim = opt.g_dim = 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 1)         
   else:
      raise ValueError('Unknown model: %s' % opt.model)

   #---  eta (generator step-size)
   if opt.cfg_eta is None:
      if opt.dataset == MNIST:
         dflt = { DCGANx+'bn': 0.5,  DCGANx+'none': 2.5, FC2+'bn': 0.1 }
      elif opt.dataset == SVHN:
         dflt = { DCGANx+'bn': 0.25, DCGANx+'none': 0.5, FC2+'bn': 0.5 }
      elif opt.dataset == CIFAR10:
         dflt = { DCGANx+'bn': 0.25, DCGANx+'none': 0.5, FC2+'bn': 0.5 }
      else:
         dflt = { Resnet4+'bn': 1, Resnet4+'none': 2.5, FC2+'bn': 0.5 }
      
      opt.cfg_eta = dflt.get(opt.model+opt.norm_type)
      if opt.cfg_eta is None:
         raise ValueError("'cfg_eta' is missing.")

   #---  optimization 
   opt.optim_type=RMSprop_str; opt.optim_eps=1e-18; opt.optim_a1=0.95; opt.optim_a2=-1
   # RMSprop used in the paper adds epsilon *before* sqrt, but pyTorch does 
   # this *after* sqrt, and so this setting is close to but not exactly the same as the paper.  
   # Adam or RMSprop with pyTorch default values can be used too, but 
   # learning rate may have to be re-tuned.
      
   #***  Setting default values for generating examples
   gen_dir_dict = {}
   gen_dir = None
   if opt.gen is None and opt.num_gen > 0 and opt.gen_interval > 0:
       # gen_dir_dict['gen'] = ''
        gen_dir_dict['dataset'] = opt.dataset
        gen_dir_dict['cfg_eta'] = opt.cfg_eta
        gen_dir_dict['lr_g'] = opt.lr_g
        gen_dir_dict['lr_d'] = opt.lr_d
        gen_dir_dict['alpha'] = opt.alpha
        gen_dir_dict['cfg_T'] = opt.cfg_T
        gen_dir_dict['cfg_N'] = opt.cfg_N
        # gen_dir_dict['num_timesteps'] = opt.num_timesteps
        gen_dir_dict['num_stages'] = opt.num_stages
        if opt.lamda != 0:
            gen_dir_dict['gptype'] = opt.gptype
            gen_dir_dict['lamda'] = opt.lamda
            gen_dir_dict['scale'] = opt.scale
        if opt.model is not None:
            gen_dir_dict['model'] = opt.model
        gen_dir = add_sep_for_dict('_', **gen_dir_dict)
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        opt.gen = gen_dir + os.path.sep+opt.dataset
   if opt.save =='':
      if opt.save_interval is None or opt.save_interval <= 0:
         opt.save_interval = 1000
      save = 'mod'
      if not os.path.exists(save):
          os.makedirs(save)
      gen_dir = add_sep_for_dict('-', **gen_dir_dict)
      print(gen_dir)
      save = save+os.path.sep+gen_dir
      opt.save = save   
   opt.logstr = opt.gen+'-'+opt.dataset+'-'+'opt.cfg_eta'+str(opt.cfg_eta)+'-'+'opt.lr'+str(opt.lr)
   #***  Display arguments 
   show_args(opt, ['dataset','dataroot','num_workers','logstr'])
   raise_if_nonpositive_any(opt, ['d_dim','g_dim','z_dim','z_std'])
   show_args(opt, ['d_model','d_dim','d_depth','g_model','g_dim','g_depth','norm_type','z_dim','z_std'], 'Net definitions ----')
   raise_if_nonpositive_any(opt, ['cfg_T','cfg_U','cfg_N','batch_size','num_stages','cfg_eta','lr','cfg_x_epo'])
   show_args(opt, ['cfg_T','cfg_U','cfg_N','num_stages','cfg_eta','cfg_x_epo','diff_max','lamda','alpha','gptype'], 'CFG --- ')
   show_args(opt, ['optim_type','optim_eps','optim_a1','optim_a2','lr','batch_size'], 'Optimization ---')
   show_args(opt, ['seed','gen','save','save_interval','gen_interval','num_gen','gen_nrow','verbose'], 'Others ---')
   ## log file to store the param##
      ## log file to store the param##
   logfile = os.path.dirname(opt.gen)
   print('logstr {}'.format(opt.logstr))
      #dir = os.path.dirname(logfile)
   if not os.path.exists(logfile):
      os.makedirs(logfile)
   logfile = logfile+os.path.sep+'log.txt'
   with open(logfile,'w') as f:
      od = vars(opt)
      for k,v in od.items():
         f.write(str(k)+':'+str(v)+'\n')
      #logfile = dir+os.path.sep+'log.txt'
#----------------------------------------------------------
def main(args):
   parser = ArgParser_HelpWithDefaults(description='cfggan_train', formatter_class=argparse.MetavarTypeHelpFormatter)
   add_args_(parser)
   parser2=prepare_parser(parser)
   opt = parser2.parse_args(args)
   check_args_(opt)
   cfggan_train(opt)
  
if __name__ == '__main__':
   main(sys.argv[1:])   
