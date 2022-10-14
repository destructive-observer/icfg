from this import s
import torch
from torch.nn.init import normal_
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.nn import Parameter as P

import utils.utils as utils
import biggan_layers as layers
import functools
# from visdom import Visdom
# import visdom
'''
单条追踪曲线设置
'''
# print(dir(visdom))
# vizG = visdom.Visdom(env='G')  # 初始化visdom类
# vizD = visdom.Visdom(env='D')

#-------------------------------------------------------------
def conv2d_params(ni, no, k, do_bias, std=0.01):
   return {'w': normal_(torch.Tensor(no, ni, k, k), std=std), 
           'b': torch.zeros(no) if do_bias else None }

def conv2dT_params(ni, no, k, do_bias, std=0.01):
   return {'w': normal_(torch.Tensor(ni, no, k, k), std=std), 
           'b': torch.zeros(no) if do_bias else None }


##-------use nn module G and D-------------#

class Generator(nn.Module):
  def __init__(self, input_dim,n0g, imgsz,
             channels,    # 1: gray-scale, 3: color
             norm_type,   # 'bn', 'none'
             requires_grad, 
             depth=3, nodemul=2, do_bias=True,init_type='NO2'):
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.input_dim = input_dim
    self.n0g = n0g
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.imgsz = imgsz
    # Resolution
    self.channels = channels
    # Kernel size
    self.norm_type = norm_type
    # Attention?
    self.requires_grad = requires_grad
    # Number of classes
    self.depth = depth
    # Activation
    # Initialization style
    self.nodemul = nodemul
    # Parameterization style
    self.do_bias = do_bias
    # Epsilon for Spectral Norm?
    self.init = init_type
    self.kernel = 5
    self.padding=2
    self.output_padding=1
    self.skip_init = False
    self.nn0 = self.n0g*(self.nodemul**(self.depth-1))*4
    self.nn = self.nn0
    self.stride = 2
    self.count = 1
    self.blocks = []
    
   
    self.act=nn.ReLU(inplace=True)
    for index in range(self.depth-1):
      for i in range(self.count):
        self.blocks+=[[nn.ReLU(inplace=True)]]
        self.blocks += [[nn.ConvTranspose2d(self.nn if i == 0 else self.nn//nodemul,self.nn//nodemul,self.kernel,bias=self.do_bias,stride=self.stride,padding=self.padding,output_padding=self.output_padding)]]
        if self.norm_type == 'bn':
          self.blocks += [[nn.BatchNorm2d(self.nn//nodemul)]]   
      # If attention on this block, attach it to the end
        self.blocks+=[[nn.ReLU(inplace=True)]]
        self.blocks+=[[nn.Conv2d(self.nn//nodemul,self.nn//nodemul,1,bias=self.do_bias,stride=1,padding=0)]]
        if self.norm_type == 'bn':
          self.blocks += [[nn.BatchNorm2d(self.nn//nodemul)] ]
        self.nn = self.nn//self.nodemul
      
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.sz = imgsz // (2**depth)
    self.linear = nn.Linear(self.input_dim,self.sz*self.sz*self.nn0)    # Embedding for projection discrimination
    self.lastconv =nn.ConvTranspose2d(self.nn,self.channels,self.kernel,stride=self.stride,padding=self.padding,bias=self.do_bias, output_padding=self.output_padding)
    # Initialize weights
    if not self.skip_init:
      self.init_weights()
  # Initialize
  def init_weights(self):
      self.param_count = 0
      for module in self.modules():
        if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
          if self.init == 'ortho':
            init.orthogonal_(module.weight)
          elif self.init == 'N02':
            init.normal_(module.weight, 0, 0.02)
          elif self.init in ['glorot', 'xavier']:
            init.xavier_uniform_(module.weight)
          else:
            print('Init style not recognized...')
          self.param_count += sum([p.data.nelement() for p in module.parameters()])
      print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
      h = x
      # print(h.device)
      h  = self.linear(h)
      h = h.view(h.size(0),self.nn0, self.sz, self.sz)
    # Loop over blocks
      for index, blocklist in enumerate(self.blocks):
        for block in blocklist:
          h = block(h)
      h = self.act(h)
    # Apply global sum pooling as in SN-GAN
   #  h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
      out = self.lastconv(h)
      # print(out.device)
      out = torch.tanh(out)
    # Get projection of final featureset onto class vectors and add to evidence
   #  out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
      return out



class Discriminator(nn.Module):

  def __init__(self, nn0, imgsz,
             channels,    # 1: gray-scale, 3: color
             norm_type,   # 'bn', 'none'
             requires_grad, 
             depth=3, leaky_slope=0.2, nodemul=2, do_bias=True,init_type='NO2'):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.nn0 = nn0
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.imgsz = imgsz
    # Resolution
    self.channels = channels
    # Kernel size
    self.norm_type = norm_type
    # Attention?
    self.requires_grad = requires_grad
    # Number of classes
    self.depth = depth
    # Activation
    self.leaky_slope = leaky_slope
    # Initialization style
    self.nodemul = nodemul
    # Parameterization style
    self.do_bias = do_bias
    # Epsilon for Spectral Norm?
    self.init = init_type
    self.kernel = 5
    self.padding=2
    self.count = 1
    self.skip_init = False
    self.stride = 2
    self.nn = self.nn0
    # Fp16?
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    self.firstconv =nn.Conv2d(self.channels,self.nn0,self.kernel,stride=self.stride,padding=self.padding,bias=self.do_bias)
   
    self.firstact=nn.LeakyReLU(self.leaky_slope,inplace=False)
    for index in range(self.depth-1):
      for i in range(self.count):
        self.blocks += [[nn.Conv2d(self.nn if i == 0 else self.nn*nodemul,self.nn*nodemul,self.kernel,bias=self.do_bias,stride=self.stride,padding=self.padding)]]
        if self.norm_type == 'bn':
          self.blocks += [[nn.BatchNorm2d(self.nn*nodemul)]]
      # If attention on this block, attach it to the end
        self.blocks+=[[nn.LeakyReLU(self.leaky_slope,inplace=False)]]
        self.blocks+=[[nn.Conv2d(self.nn*nodemul,self.nn*nodemul,1,bias=self.do_bias,stride=1,padding=0)]]
        if self.norm_type == 'bn':
          self.blocks += [[nn.BatchNorm2d(self.nn*nodemul)]]
        self.blocks+=[[nn.LeakyReLU(self.leaky_slope,inplace=False)]] 
        self.nn = self.nn*self.nodemul
      
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.sz = imgsz // (2**depth)
    self.linear = nn.Linear(self.sz*self.sz*self.nn,1)    # Embedding for projection discrimination
   #  print(self.sz*self.sz*self.nn)
    # Initialize weights
    if not self.skip_init:
      self.init_weights()
  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    h = self.firstconv(h)
    h = self.firstact(h)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    h = h.view(h.size(0),-1)
   #  print(h.shape)
    # Apply global sum pooling as in SN-GAN
   #  h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
    out = self.linear(h)
    # Get projection of final featureset onto class vectors and add to evidence
   #  out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
    return out



##--------use nn module G and D-------------------#

#-------------------------------------------------------------
def dcganx_D(nn0, imgsz,
             channels,    # 1: gray-scale, 3: color
             norm_type,   # 'bn', 'none'
             requires_grad, 
             depth=3, leaky_slope=0.2, nodemul=2, do_bias=True):
              
   ker=5; padding=2
   
   def gen_block_params(ni, no, k):
      return {
         'conv0': conv2d_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, 1, do_bias), 
         'bn0': utils.bnparams(no) if norm_type == 'bn' else None, 
         'bn1': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no, count):
       return {'block%d' % i: gen_block_params(ni if i == 0 else no, no, ker) for i in range(count)}

   count = 1
   sz = imgsz // (2**depth)
   nn = nn0
   p = { 'conv0': conv2d_params(channels, nn0, ker, do_bias) }
   for d in range(depth-1):
      p['group%d'%d] = gen_group_params(nn, nn*nodemul, count)
      nn = nn*nodemul
   p['fc'] = utils.linear_params(sz*sz*nn, 1)
   flat_params = utils.cast(utils.flatten(p))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, stride):
      o = F.conv2d(x, params[base+'.conv0.w'], params.get(base+'conv0.b'), stride=stride, padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn0', mode)
      o = F.leaky_relu(o, negative_slope=leaky_slope, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'conv1.b'), stride=1, padding=0)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn1', mode)
      o = F.leaky_relu(o, negative_slope=leaky_slope, inplace=True)
      return o

   def group(o, params, base, mode, stride=2):
      n = 1
      for i in range(n):
         o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
      return o

   def f(input, params, mode):
      o = F.conv2d(input, params['conv0.w'], params.get('conv0.b'), stride=2, padding=padding)
      o = F.leaky_relu(o, negative_slope=leaky_slope, inplace=True)
      for d in range(depth-1):
         o = group(o, params, 'group%d'%d, mode)
      o = o.view(o.size(0), -1)
      o = F.linear(o, params['fc.weight'], params['fc.bias'])
      return o

   return f, flat_params

#-------------------------------------------------------------
def dcganx_G(input_dim, n0g, imgsz, channels,
             norm_type,  # 'bn', 'none'
             requires_grad, depth=3, 
             nodemul=2, do_bias=True):
              
   ker=5; padding=2; output_padding=1

   def gen_block_T_params(ni, no, k):
      return {
         'convT0': conv2dT_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, 1, do_bias), 
         'bn0': utils.bnparams(no) if norm_type == 'bn' else None, 
         'bn1': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_T_params(ni, no, count):
       return {'block%d' % i: gen_block_T_params(ni if i == 0 else no, no, ker) for i in range(count)}

   count = 1
   nn0 = n0g * (nodemul**(depth-1))*4
   sz = imgsz // (2**depth)  
   p = { 'proj': utils.linear_params(input_dim, nn0*sz*sz) }
   nn = nn0
   for d in range(depth-1):
      p['group%d'%d] = gen_group_T_params(nn, nn//nodemul, count)
      nn = nn//nodemul
   p['last_convT'] = conv2dT_params(nn, channels, ker, do_bias)
   flat_params = utils.cast(utils.flatten(p))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, stride):
      o = F.relu(x, inplace=True)
      o = F.conv_transpose2d(o, params[base+'.convT0.w'], params.get(base+'.convT0.b'),
                             stride=stride, padding=padding, output_padding=output_padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn0', mode)

      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'.conv1.b'),
                   stride=1, padding=0)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn1', mode)
      return o

   def group(o, params, base, mode, stride=2):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
      return o

   def f(input, params, mode):
      o = F.linear(input, params['proj.weight'], params['proj.bias'])
      o = o.view(input.size(0), nn0, sz, sz)
      for d in range(depth-1):
        o = group(o, params, 'group%d'%d, mode)
      o = F.relu(o, inplace=True)
      o = F.conv_transpose2d(o, params['last_convT.w'], params.get('last_convT.b'), stride=2,
                             padding=padding, output_padding=output_padding)
      o = torch.tanh(o)
      return o

   return f, flat_params

#-------------------------------------------------------------
def fcn_G(input_dim, nn, imgsz, channels, requires_grad, depth=2):
   def gen_block_params(ni, no):
      return {'fc': utils.linear_params(ni, no),}

   def gen_group_params(ni, no, count):
      return {'block%d' % i: gen_block_params(ni if i == 0 else no, no) for i in range(count)}

   flat_params = utils.cast(utils.flatten({
        'group0': gen_group_params(input_dim, nn, depth),
        'last_proj': utils.linear_params(nn, imgsz*imgsz*channels),
   }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode):
      return F.relu(F.linear(x, params[base+'.fc.weight'], params[base+'.fc.bias']), inplace=True)

   def group(o, params, base, mode):
      for i in range(depth):
         o = block(o, params, '%s.block%d' % (base,i), mode)
      return o

   def f(input, params, mode):
      o = group(input, params, 'group0', mode)
      o = F.linear(o, params['last_proj.weight'], params['last_proj.bias'])
      o = torch.tanh(o)
#      o = o.view(o.size(0), channels, imgsz, imgsz)
      o = o.reshape(o.size(0), channels, imgsz, imgsz)      
      return o

   return f, flat_params



# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

class biggan_G(nn.Module):
  def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn',
               **kwargs):
    super(biggan_G, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Kernel size?
    self.kernel_size = G_kernel_size
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    # Use shared embeddings?
    self.G_shared = G_shared
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    # Hierarchical latent space?
    self.hier = hier
    # Cross replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.G_param = G_param
    # Normalization style
    self.norm_style = norm_style
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # fp16?
    self.fp16 = G_fp16
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[resolution]

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

   #  # Set up optimizer
   #  # If this is an EMA copy, no need for an optim, so just return now
   #  if no_optim:
   #    return
   #  self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
   #  if G_mixed_precision:
   #    print('Using fp16 adam in G...')
   #    import utils
   #    self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
   #                         betas=(self.B1, self.B2), weight_decay=0,
   #                         eps=self.adam_eps)
   #  else:
   #    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
   #                         betas=(self.B1, self.B2), weight_decay=0,
   #                         eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y):
    # print(y.shape)
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    # print(z.shape)
    # print(self.linear.weight.shape)
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class biggan_D(nn.Module):

  def __init__(self, D_ch=64, D_wide=True, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(biggan_D, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Fp16?
    self.fp16 = D_fp16
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
    # self.linear = self.which_linear(1024*16, output_dim)
    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
    self.output_download = nn.AvgPool2d(16)
    self.output_conv  =self.which_conv(3,1024)
    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
   #  self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
   #  if D_mixed_precision:
   #    print('Using fp16 adam in D...')
   #    import utils
   #    self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
   #                           betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
   #  else:
   #    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
   #                           betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    # print(x.shape)
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        # print(h.shape)    
        h = block(h)
        # print(h.shape) 
    # print(h.shape)
    # Apply global sum pooling as in SN-GAN
    # print(x.shape)
    # x1 = self.output_conv(x)
    # print(x1.shape)
    # x1 = self.output_download(x1)
    # print(x1.shape)
    # h = h+x1
    h = torch.sum(self.activation(h), [2, 3])
    # print(h.shape)
    # h = h.view(h.size(0),-1)
    # Get initial class-unconditional output
    out = self.linear(h)
    # Get projection of final featureset onto class vectors and add to evidence
    out = out 
   #  + torch.sum(self.embed(y) * h, 1, keepdim=True)
    return out
