"""
   From https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py
"""
import torch
from torch.nn.init import kaiming_normal_,xavier_normal_,_calculate_fan_in_and_fan_out
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from nested_dict import nested_dict
import math

def cast(params, dtype='float'):
   if isinstance(params, dict):
      return {k: cast(v, dtype) for k,v in params.items()}
   else:
      return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()

def conv_params(ni, no, k=1):
   return kaiming_normal_(torch.Tensor(no, ni, k, k),nonlinearity='relu')

def conv_params_Xavier(ni, no, k=1):
   return xavier_normal_(torch.Tensor(no, ni, k, k))

def linear_params(ni, no):
       return {'weight': kaiming_normal_(torch.Tensor(no, ni),nonlinearity='relu'), 'bias': torch.zeros(no)}
def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out
def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
     
def linear_params_kaiming(ni, no,scale=0.):
       return {'weight': kaiming_uniform_(torch.Tensor(no, ni), gain=1e-10 if scale == 0 else scale, mode='fan_avg'), 'bias': torch.zeros(no)}

def bnparams(n):
   return {'weight': torch.rand(n),
           'bias': torch.zeros(n),
           'running_mean': torch.zeros(n),
           'running_var': torch.ones(n)}

def data_parallel(f, input, params, mode, device_ids, output_device=None):
   assert isinstance(device_ids, list)
   if output_device is None:
      output_device = device_ids[0]

   if len(device_ids) == 1:
      return f(input, params, mode)

   params_all = Broadcast.apply(device_ids, *params.values())
   params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                      for j in range(len(device_ids))]

   replicas = [partial(f, params=p, mode=mode) for p in params_replicas]
   inputs = scatter([input], device_ids)
   outputs = parallel_apply(replicas, inputs)
   return gather(outputs, output_device)

def flatten(params):
   return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
   return F.batch_norm(x, weight=params[base + '.weight'],
                       bias=params[base + '.bias'],
                       running_mean=params[base + '.running_mean'],
                       running_var=params[base + '.running_var'],
                       training=mode)

def set_requires_grad_except_bn_(params):
   for k, v in params.items():
      if not k.endswith('running_mean') and not k.endswith('running_var'):
         v.requires_grad = True

# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)    
    # return self.variable
    
  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)    
    return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', 
                fp16=False,z_var=1.0):
  z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device,torch.float16 if fp16 else torch.float32)   
  
  if fp16:
    z_ = z_.half()

  y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
  y_.init_distribution('categorical',num_categories=nclasses)
  y_ = y_.to(device, torch.int64)
  return z_, y_


def initiate_standing_stats(net):
  for module in net.modules():
    if hasattr(module, 'accumulate_standing'):
      module.reset_stats()
      module.accumulate_standing = True


def accumulate_standing_stats(net, z, y, nclasses, num_accumulations=16):
  initiate_standing_stats(net)
  net.train()
  for i in range(num_accumulations):
    with torch.no_grad():
      z.normal_()
      y.random_(0, nclasses)
      x = net(z, net.shared(y)) # No need to parallelize here unless using syncbn
  # Set to eval mode
  net.eval() 

