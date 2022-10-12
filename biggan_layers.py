''' Layers
    This file contains various layers for the BigGAN models.
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import utils.utils as utils
from utils.utils import flatten

# from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d


# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    # print(device)
    u = u.cuda()
    # W = W.to('cpu')
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input
 

# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]
      # Compute the spectrally-normalized weight

# Spectral normalization base class 
class SN_ICFG(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    self.num_outputs = num_outputs
    self.u_list= None
    self.sv_list =None
    # Register a singular vector for each sv
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  def W_N(self,weight,training):
    W_mat = weight.view(weight.size(0), -1)
    device = W_mat.device
    self.u_list = [torch.randn(1, self.num_outputs)]
    self.sv_list = [torch.ones(1)]

    self.u_list = self.u_list
    self.sv_list = self.sv_list

    # for i in range(self.num_svs):
    #   # self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
    #   # self.register_buffer('sv%d' % i, torch.ones(1))
    #   # print(i)
    #   # print(self.num_outputs)
    #   setattr(self, 'u%d' % i,torch.randn(1, self.num_outputs))
    #   setattr(self, 'sv%d' % i, torch.ones(1))
    # print(W_mat.device)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat,self.u_list, update=training, eps=self.eps) 
    # Update the svs
    if training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          # self.sv[i][:] = sv   
          self.sv_list[i][:] =sv  
    return weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, 
                          sparse, _weight)
    SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
  def forward(self, x):
    return F.embedding(x, self.W_())

def SN_conv2d_icfg(ni, no, k,sn,padding,do_bias=True, 
                          num_svs=1, num_itrs=1, eps=1e-12,stride=1,requires_grad=True):
  def get_sn_conv2d_params(ni, no, k, do_bias):
    return {'w': init.orthogonal_(torch.Tensor(no, ni, k, k)), 
           'b': torch.zeros(no) if do_bias else None }
  p=get_sn_conv2d_params(ni, no, k, do_bias)
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def forward_sn_conv2d_params(o,params,base,training=True,stride=stride,padding=padding,out_channels=no,sn=sn,num_svs=num_svs, num_itrs=num_itrs, eps=eps):
    if sn:
      sn_1 = SN_ICFG(num_svs, num_itrs, out_channels, eps=eps)
      o = F.conv2d(o, sn_1.W_N(params[base+'.w'],training=training), params.get(base+'.b'), stride=stride, padding=padding)
    else:
      o = F.conv2d(o, params[base+'.w'], params.get(base+'.b'), stride=stride, padding=padding)
    return o
  return forward_sn_conv2d_params,flat_params
def SN_linear_icfg(ni, no,sn,do_bias=True,num_svs=1, num_itrs=1, eps=1e-12,requires_grad=True):
  def get_sn_linear_params(ni, no, do_bias):
    return {'w': init.orthogonal_(torch.Tensor(no, ni)), 'b': torch.zeros(no) if do_bias else None }
  p=get_sn_linear_params(ni, no, do_bias)
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)  
  def forward_sn_linear_params(o,params,base,training=True,out_channels=no,sn=sn,num_svs=num_svs, num_itrs=num_itrs, eps=eps):
    if sn:
      sn_1 = SN_ICFG(num_svs, num_itrs, out_channels, eps=eps)

      o = F.linear(o, sn_1.W_N(params[base+'.w'],training=training), params.get(base+'.b'))
    else:
      o = F.linear(o,params[base+'.w'], params.get(base+'.b'))
    return o
  return forward_sn_linear_params,flat_params
def SN_embedding_icfg(num_embeddings, embedding_dim,sn,num_svs=1, num_itrs=1, eps=1e-12,requires_grad=True):

  def get_embedding__params(num_embeddings, embedding_dim):
    return {'w': init.orthogonal_(torch.Tensor(num_embeddings, embedding_dim))}
  p=get_embedding__params(num_embeddings, embedding_dim)
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)  
  def forward_get_embedding__params(o,params,base,training=True,num_embeddings=num_embeddings,sn=sn,num_svs=num_svs, num_itrs=num_itrs, eps=eps):
    if sn:
      sn_1 = SN_ICFG(num_svs, num_itrs, num_embeddings, eps=eps)
      o = F.embedding(o.long(), sn_1.W_N(params[base+'.w'],training=training))
    else:
      # print('o1{}'.format(o.shape))
      # print('111{}'.format(params[base+'.w'].shape))
      o = F.embedding(o.long(), params[base+'.w'])
    return o
      # print('o2{}'.format(o))
  return forward_get_embedding__params,flat_params
# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
  def __init__(self, ch, which_conv=SNConv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])    
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x

def SN_Attention_icfg(ch, which_conv=SNConv2d, name='attention', requires_grad=True):
  f={'theta':which_conv(ch,ch // 8, k=1,do_bias=False, padding=0)[0]}
  p={'theta':which_conv(ch,ch // 8, k=1,do_bias=False, padding=0)[1]}

  f['phi']=which_conv(ch,ch // 8, k=1,do_bias=False, padding=0)[0]
  p['phi']=which_conv(ch,ch // 8, k=1,do_bias=False, padding=0)[1]

  f['g']=which_conv(ch,ch // 2, k=1,do_bias=False, padding=0)[0]
  p['g']=which_conv(ch,ch // 2, k=1,do_bias=False, padding=0)[1]

  f['o']=which_conv(ch//2,ch, k=1,do_bias=False, padding=0)[0]
  p['o']=which_conv(ch//2,ch, k=1,do_bias=False, padding=0)[1]

  p['gamma'] = torch.tensor(0.)
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params) 
  def forward_sn_attention_params(x,params,base,training=True,y=None):
    theta =  f['theta'](x,params,base+'.theta',training)
    phi = F.max_pool2d(f['phi'](x,params,base+'.phi',training), [2,2])
    g = F.max_pool2d(f['g'](x,params,base+'.g',training), [2,2])
    theta = theta.view(-1, ch// 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, ch// 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o =  f['o'](torch.bmm(g, beta.transpose(1,2)).view(-1,ch // 2, x.shape[2], x.shape[3]),params,base+'.o',training)
    return params[base+'.gamma']*o+x
  
  return forward_sn_attention_params,flat_params
# def forward_normal_attention_params(x,params,base,channels,y=None):
#   theta =  F.conv2d(x,params[base+'theta.w'],params[base+'theta.b'],padding=0)
#   phi = F.max_pool2d(F.conv2d(x,params[base+'phi.w'],params[base+'phi.b'],padding=0))
#   g = F.max_pool2d(F.conv2d(x,params[base+'g.w'],params[base+'g.b'],padding=0))
#   theta = theta.view(-1, channels// 8, x.shape[2] * x.shape[3])
#   phi = phi.view(-1, channels // 8, x.shape[2] * x.shape[3] // 4)
#   g = g.view(-1, channels// 2, x.shape[2] * x.shape[3] // 4)
#     # Matmul and softmax to get attention maps
#   beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
#     # Attention map times g path
#   o =  F.conv2d(torch.bmm(g, beta.transpose(1,2)).view(-1,channels // 2, x.shape[2], x.shape[3]),params[base+'o.w'],params[base+'o.b'],padding=0)
#   return params[base+'p']*o+x

# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
  # Apply scale and shift--if gain and bias are provided, fuse them here
  # Prepare scale
  scale = torch.rsqrt(var + eps)
  # If a gain is provided, use it
  if gain is not None:
    scale = scale * gain
  # Prepare shift
  shift = mean * scale
  # If bias is provided, use it
  if bias is not None:
    shift = shift - bias
  return x * scale - shift
  #return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
  # Cast x to float32 if necessary
  float_x = x.float()
  # Calculate expected value of x (m) and expected value of x**2 (m2)  
  # Mean of x
  m = torch.mean(float_x, [0, 2, 3], keepdim=True)
  # Mean of x squared
  m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
  # Calculate variance as mean of squared minus mean squared.
  var = (m2 - m **2)
  # Cast back to float 16 if necessary
  var = var.type(x.type())
  m = m.type(x.type())
  # Return mean and variance for updating stored mean/var if requested  
  if return_mean_var:
    return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
  else:
    return fused_bn(x, m, var, gain, bias, eps)

# My batchnorm, supports standing stats    
class myBN(nn.Module):
  def __init__(self, num_channels, eps=1e-5, momentum=0.1):
    super(myBN, self).__init__()
    # momentum for updating running stats
    self.momentum = momentum
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Register buffers
    self.register_buffer('stored_mean', torch.zeros(num_channels))
    self.register_buffer('stored_var',  torch.ones(num_channels))
    self.register_buffer('accumulation_counter', torch.zeros(1))
    # Accumulate running means and vars
    self.accumulate_standing = False
    
  # reset standing stats
  def reset_stats(self):
    self.stored_mean[:] = 0
    self.stored_var[:] = 0
    self.accumulation_counter[:] = 0
    
  def forward(self, x, gain, bias):
    if self.training:
      out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
      # If accumulating standing stats, increment them
      if self.accumulate_standing:
        self.stored_mean[:] = self.stored_mean + mean.data
        self.stored_var[:] = self.stored_var + var.data
        self.accumulation_counter += 1.0
      # If not accumulating standing stats, take running averages
      else:
        self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
        self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
      return out
    # If not in training mode, use the stored statistics
    else:         
      mean = self.stored_mean.view(1, -1, 1, 1)
      var = self.stored_var.view(1, -1, 1, 1)
      # If using standing stats, divide them by the accumulation counter   
      if self.accumulate_standing:
        mean = mean / self.accumulation_counter
        var = var / self.accumulation_counter
      return fused_bn(x, mean, var, gain, bias, self.eps)
def myBN_ICFG(num_channels,eps=1e-5, momentum=0.1,requires_grad=True):
  def get_mybn_params(num_channels):
    return{'stored_mean':torch.zeros(num_channels),'stored_var':torch.ones(num_channels),'accumulation_counter':torch.zeros(1)}
  p=get_mybn_params(num_channels)
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params) 
  def forward_mybn_params(x,params,base,training,gain, bias,eps=eps, momentum=momentum,accumulate_standing=False):
    if training:
      out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=eps)
      # If accumulating standing stats, increment them
      if accumulate_standing:
        params[base+'.stored_mean'][:] = params[base+'.stored_mean'] + mean.data
        params[base+'.stored_var'][:] = params[base+'.stored_var'] + var.data
        params[base+'.accumulation_counter'] += 1.0
      # If not accumulating standing stats, take running averages
      else:
        params[base+'.stored_mean'][:] = params[base+'.stored_mean'] * (1 - momentum) + mean * momentum
        params[base+'.stored_var'][:] = params[base+'.stored_var'] * (1 - momentum) + var * momentum
      return out
    # If not in training mode, use the stored statistics
    else:         
      mean = params[base+'.stored_mean'].view(1, -1, 1, 1)
      var = params[base+'.stored_var'].view(1, -1, 1, 1)
      # If using standing stats, divide them by the accumulation counter   
      if accumulate_standing:
        mean = mean / params[base+'.accumulation_counter']
        var = var / params[base+'.accumulation_counter']
      return fused_bn(x, mean, var, gain, bias, eps)
  return forward_mybn_params,flat_params
# Simple function to handle groupnorm norm stylization                      
def groupnorm(x, norm_style):
  # If number of channels specified in norm_style:
  if 'ch' in norm_style:
    ch = int(norm_style.split('_')[-1])
    groups = max(int(x.shape[1]) // ch, 1)
  # If number of groups specified in norm style
  elif 'grp' in norm_style:
    groups = int(norm_style.split('_')[-1])
  # If neither, default to groups = 16
  else:
    groups = 16
  return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable). 
class ccbn(nn.Module):
  def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',):
    super(ccbn, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # Norm style?
    self.norm_style = norm_style
    
    if self.cross_replica:
      return
      # self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif self.mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size)) 
    
    
  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    # print(x.shape)
    # print(y.shape)
    # gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    # bias = self.bias(y).view(y.size(0), -1, 1, 1)
    # If using my batchnorm
    if self.mybn or self.cross_replica:
        return
      # return self.bn(x, gain=gain, bias=bias)
    # else:
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = groupnorm(x, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x
      return out
      # return out * gain + bias
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)

def ccbn_icfg(output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',requires_grad=True):
  f={'gain':which_linear(input_size,output_size)[0]}
  p={'gain':which_linear(input_size,output_size)[1]}

  f['bias']=which_linear(input_size,output_size)[0]
  p['bias']=which_linear(input_size,output_size)[1]

  p['running_mean']= torch.zeros(output_size)
  p['running_var']= torch.ones(output_size)

  # p['stored_var']=torch.ones(output_size)
  # p['stored_mean']=torch.zeros(output_size)
  # f['mybn'],p['mybn'] = myBN_ICFG(output_size,eps,momentum)

  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params) 
  def forward_ccbn_params(x,y,params,base,training,eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn'):
    # print(y.shape)
    # print('y.shape{}'.format(y.shape))
    # gain = (1 + f['gain'](y,params,base+'.gain')).view(y.size(0), -1, 1, 1)
    # bias =f['bias'](y,params,base+'.bias').view(y.size(0), -1, 1, 1)
    # If using my batchnorm
    # print('mybn{}'.format(mybn))
    # print('cross_replica{}'.format(cross_replica))
    if mybn or cross_replica:
      return
      # myBN_ICFG
      # return f['mybn'](x,params,base+'.mybn',True,gain=gain, bias=bias)
      # return self.bn(x, gain=gain, bias=bias)
    # else:
    else:
      if norm_style == 'bn':
        out = F.batch_norm(x, params[base + '.running_mean'],params[base + '.running_var'],None,None, training, 0.1, eps)
      elif norm_style == 'in':
        out = F.instance_norm(x,params[base + '.running_mean'],params[base + '.running_var'],None,None,
                          training, 0.1, eps)
      elif norm_style == 'gn':
        out = groupnorm(x, norm_style)
      elif norm_style == 'nonorm':
        out = x
      # print('out.shape{}'.format(out.shape))
      # print('gain.shape{}'.format(gain.shape))
      # print('gain.w{}'.format(params[base+'.gain.w'].shape))
      # print(bias.shape)
      # print(params[base+'.gain.b'].shape)
      return out
      # return out * gain + bias
  return forward_ccbn_params,flat_params

# Normal, non-class-conditional BN
class bn(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False):
    super(bn, self).__init__()
    self.output_size= output_size
    # Prepare gain and bias layers
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    
    if self.cross_replica:
      # self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)    
      return
    elif mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
     # Register buffers if neither of the above
    else:     
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))
    
  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)
def BN_icfg(output_size, eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False,requires_grad=True):
  def get_bn_params(output_size):
    return{'gain':torch.ones(output_size),'bias':torch.zeros(output_size),
    'running_mean':torch.zeros(output_size),'running_var':torch.ones(output_size)}
  p= get_bn_params(output_size)
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params) 
  def forward_bn_params(x,params,base,training):
    return F.batch_norm(x, params[base+'.running_mean'], params[base+'.running_var'],  params[base+'.gain'],  params[base+'.bias'],
                          training, momentum, eps)
  return forward_bn_params,flat_params

def GBblock_icfg(in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None, 
               upsample=None,requires_grad=True):
  
  
  learnable_sc = in_channels != out_channels or upsample
  p={'conv1':which_conv(in_channels, out_channels)[1]}
  f={'conv1':which_conv(in_channels, out_channels)[0]}
  p['conv2']=which_conv(out_channels, out_channels)[1]
  f['conv2']=which_conv(out_channels, out_channels)[0]
  if learnable_sc:
    p['conv_sc'] = which_conv(in_channels, out_channels, k=1,padding=0)[1]
    f['conv_sc'] = which_conv(in_channels, out_channels, k=1,padding=0)[0]
  p['bn1']=which_bn(output_size=in_channels)[1]
  f['bn1']=which_bn(output_size=in_channels)[0]
  p['bn2']=which_bn(output_size=out_channels)[1]
  f['bn2']=which_bn(output_size=out_channels)[0]
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f_g(x,y,params,base,training,eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn'):
    h = activation(f['bn1'](x, y,params,base+'.bn1',training))
    if upsample:
      h = upsample(h)
      x = upsample(x)
    h = f['conv1'](h,params,base+'.conv1',training)
    h = activation(f['bn2'](h, y,params,base+'.bn2',training))
    h = f['conv2'](h,params,base+'.conv2',training)
    if learnable_sc:       
      x = f['conv_sc'](x,params,base+'.conv_sc',training)
    return h + x
  return f_g,flat_params

# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must 
# be preselected)
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None, 
               upsample=None):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x, y):
    h = self.activation(self.bn1(x, y))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    h = self.activation(self.bn2(h, y))
    h = self.conv2(h)
    if self.learnable_sc:       
      x = self.conv_sc(x)
    return h + x
    
    
# Residual block for the discriminator
class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x
    
  def forward(self, x):
    if self.preactivation:
      # h = self.activation(x) # NOT TODAY SATAN
      # Andy's note: This line *must* be an out-of-place ReLU or it 
      #              will negatively affect the shortcut connection.
      h = F.relu(x)
    else:
      h = x    
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)     
        
    return h + self.shortcut(x)
def DBblock_icfg(in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,requires_grad=True):
  
  
  hidden_channels = out_channels if wide else in_channels
  learnable_sc = True if (in_channels != out_channels) or downsample else False
  p={'conv1':which_conv(in_channels, hidden_channels)[1]}
  f={'conv1':which_conv(in_channels, hidden_channels)[0]}
  p['conv2']=which_conv(hidden_channels, out_channels)[1]
  f['conv2']=which_conv(hidden_channels, out_channels)[0]
  if learnable_sc:
    p['conv_sc'] = which_conv(in_channels, out_channels, k=1,padding=0)[1]
    f['conv_sc'] = which_conv(in_channels, out_channels, k=1,padding=0)[0]
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def shortcut(x,params,base):
    if preactivation:
      if learnable_sc:
        x = f['conv_sc'](x,params,base+'.conv_sc')
      if downsample:
        x = downsample(x)
    else:
      if downsample:
        x = downsample(x)
      if learnable_sc:
        x = f['conv_sc'](x,params,base+'.conv_sc')
    return x
  def f_d(x,params,base):
    if preactivation:
      # h = self.activation(x) # NOT TODAY SATAN
      # Andy's note: This line *must* be an out-of-place ReLU or it 
      #              will negatively affect the shortcut connection.
      h = F.relu(x)
    else:
      h = x    
    h =  f['conv1'](h,params,base+'.conv1')
    h =  f['conv2'](activation(h),params,base+'.conv2')
    if downsample:
      h = downsample(h)     
        
    return h + shortcut(x,params,base)
  return f_d,flat_params
# dogball