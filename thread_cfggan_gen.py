import argparse
import torch
import sys
from torch.backends import cudnn
from torch.nn.init import normal_
import numpy as np

from thread_cfggan_train import DCGANx, Resnet4, FCn,DCGAN4,Resnet3,Biggan
import thread_netdef
from thread_cfggan import DDG, generate
from utils.utils0 import raise_if_absent, add_if_absent_, raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults, timeLog
import utils_biggan
import os
import argparse
import torch.distributed as dist

cudnn.benchmark = True

#----------------------------------------------------------
def gen_main(rank, gpu,gen_opt):
   show_args(gen_opt, ['seed','gen','saved','num_gen'])

   torch.manual_seed(gen_opt.seed+rank)
   np.random.seed(gen_opt.seed+rank)
   if torch.cuda.is_available():
      torch.cuda.manual_seed_all(gen_opt.seed+rank)

   torch.backends.cudnn.benchmark = True
   timeLog('Reading from %s ... ' % gen_opt.saved)
   from_file = torch.load(gen_opt.saved, map_location=None if torch.cuda.is_available() else 'cpu')

   opt = from_file['opt']
   print(opt)
   print(hasattr(opt,'gptype'))
   opt.device = torch.device('cuda:{}'.format(gpu))
   if hasattr(opt,'gptype') == False:
      opt.gptype = 2
   #---  these must be in sync with cfggan_train  --------------
   def d_config(requires_grad):  # D
      if opt.d_model == DCGANx:
         return thread_netdef.Discriminator(opt.d_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.d_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.d_model == DCGAN4:
         return thread_netdef.Discriminator(opt.d_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.d_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.d_model == Resnet4:
         return thread_netdef.resnet4_D(opt.d_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad,  depth=opt.d_depth, 
                                 do_bias=not opt.do_no_bias)
      elif opt.d_model == Resnet3:
         # if opt.d_depth != 3:
            # logging('WARNING: d_depth is ignored as d_model is Resnet3.')
         return thread_netdef.resnet4_D(opt.d_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, depth=opt.d_depth,  
                                 do_bias=not opt.do_no_bias)  
      elif opt.d_model == Biggan:
         # if opt.d_depth != 3:
            # logging('WARNING: d_depth is ignored as d_model is Resnet3.')
         return thread_netdef.biggan_D(resolution=opt.resolution,D_ch=opt.g_dim,D_attn=str(opt.g_dim),n_classes=opt.n_classes)    
      else:
         raise ValueError('Unknown d_model: %s' % opt.d_model)
   def g_config(requires_grad):  # G
      if opt.g_model == DCGANx:
         return thread_netdef.Generator(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.g_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.g_model == Resnet4:   
         return thread_netdef.resnet4_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, depth=opt.g_depth,  do_bias=not opt.do_no_bias)
      elif opt.g_model == DCGAN4:
         return thread_netdef.Generator(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.g_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.g_model == Resnet3:
         # if opt.g_depth != 3:
         #    logging('WARNING: d_depth is ignored as d_model is Resnet3.')      
         return thread_netdef.resnet4_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, depth=opt.g_depth,  do_bias=not opt.do_no_bias)                                     
      elif opt.g_model == FCn:
         return thread_netdef.fcn_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, requires_grad, depth=opt.g_depth)
      elif opt.g_model == Biggan:
         # if opt.d_depth != 3:
            # logging('WARNING: d_depth is ignored as d_model is Resnet3.')
         return thread_netdef.biggan_G(resolution=opt.resolution,G_ch=opt.g_dim,G_attn=str(opt.g_dim),n_classes=opt.n_classes)
      else:
         raise ValueError('Unknown g_model: %s' % opt.g_model)
   def z_gen(num):
      return normal_(torch.Tensor(num, opt.z_dim), std=opt.z_std),torch.zeros(num).random_(0, 10)
   def z_y_gen_function_new(num,dim_z, nclasses):
      return normal_(torch.Tensor(num, opt.z_dim), std=opt.z_std),torch.zeros(num).random_(0, opt.n_classes)
   #-------------------------------------------------------------
   def z_y_gen(num,dim=0,n_classes=0):
      if opt.n_classes is not None:
         # y_ = Distribution(torch.zeros(num, requires_grad=False))
         # y_.init_distribution('categorical',num_categories=n_classes)
         # return z_,y_
         return z_y_gen_function_new(num,opt.z_dim, opt.n_classes)
         # return utils_biggan.prepare_z_y(num,opt.z_dim,opt.n_classes)
      else:
         return z_gen(num)
   ddg = DDG(opt, d_config, g_config, z_y_gen, optim_config=None, from_file=from_file)
   for i in range(200):
      # gen_opt["num_gen"]=40*i
      generate(gen_opt, ddg,l=str(i))

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, ['saved','num_gen','gen','gen_nrow'], 'cfggan_gen')
   add_if_absent_(opt, ['seed'], 1)

   raise_if_nonpositive_any(opt, ['num_gen'])  

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    args.gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, args.gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()          
#----------------------------------------------------------
def main():
   parser = ArgParser_HelpWithDefaults(description='cfggan_gen', formatter_class=argparse.MetavarTypeHelpFormatter)
   parser.add_argument('--seed', type=int, default=1, help='Random seed.')   
   parser.add_argument('--gen', type=str, required=True, help='Pathname for writing generated images.')   
   parser.add_argument('--saved', type=str, required=True, help='Pathname for the saved model.') 
   parser.add_argument('--num_gen', type=int, default=40, help='Number of images to be generated.') 
   parser.add_argument('--gen_nrow', type=int, default=8, help='Number of images in each row when making a collage. -1: No collage.')
   parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
   parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
   parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
   parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
   parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
   parser.add_argument('--master_port', type=str, default='6030',
                        help='address for master')
                        
   opt = parser.parse_args() 
   opt.world_size = opt.num_proc_node * opt.num_process_per_node
   size = opt.num_process_per_node

   print('starting in debug mode')
        
   init_processes(0, size, gen_main, opt)  
#----------------------------------------------------------
if __name__ == '__main__':
   main() 