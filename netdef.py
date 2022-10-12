from this import s
import torch
from torch.nn.init import normal_
import torch.nn.functional as F
import utils.utils as utils
import biggan_layers
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

##-----add conditional for minst here-------------------###
def dcganx_C_D(nn0, imgsz,
             channels,    # 1: gray-scale, 3: color
             norm_type,   # 'bn', 'none'
             requires_grad, 
             depth=3, leaky_slope=0.2, nodemul=2, do_bias=True,label_dim=10,fc_label_dim=1000):
              
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
   p['fc'] = utils.linear_params(sz*sz*nn+fc_label_dim, 1)
   p['fc_label']=utils.linear_params(label_dim, fc_label_dim)
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

   def f(input,label, params, mode):
      o = F.conv2d(input, params['conv0.w'], params.get('conv0.b'), stride=2, padding=padding)
      o = F.leaky_relu(o, negative_slope=leaky_slope, inplace=True)
      for d in range(depth-1):
         o = group(o, params, 'group%d'%d, mode)
      o = o.view(o.size(0), -1)
      y_ = F.linear(label, params['fc_label.weight'], params['fc_label.bias'])
      y_ = F.leaky_relu(y_, negative_slope=leaky_slope, inplace=True)
      o = torch.cat([o, y_], 1)
      o = F.linear(o, params['fc.weight'], params['fc.bias'])
      return o

   return f, flat_params

#-------------------------------------------------------------
def dcganx_C_G(input_dim, n0g, imgsz, channels,
             norm_type,  # 'bn', 'none'
             requires_grad, depth=3, 
             nodemul=2, do_bias=True,label_dim=10,fc_label_dim=1000):
              
   ker=5; padding=2; output_padding=1

   def gen_block_T_params(ni, no, k):
      return {
         'convT0': conv2dT_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, 1, do_bias), 
         'bn0': utils.bnparams(no) if norm_type == 'bn' else None, 
         'bn1': utils.bnparams(no) if norm_type == 'bn' else None,
         'fc_label':utils.linear_params(label_dim, ni),
         'fc_label1':utils.linear_params(label_dim, ni)
      }

   def gen_group_T_params(ni, no, count):
       return {'block%d' % i: gen_block_T_params(ni if i == 0 else no, no, ker) for i in range(count)}

   count = 1
   nn0 = n0g * (nodemul**(depth-1))*4
   sz = imgsz // (2**depth)  
   p = { 'proj': utils.linear_params(input_dim, nn0*sz*sz) }
   # p['fc_label']=utils.linear_params(label_dim, nn0)
   # p['fc_label1']=utils.linear_params(label_dim, nn0)
   nn = nn0
   for d in range(depth-1):
      p['group%d'%d] = gen_group_T_params(nn, nn//nodemul, count)
      nn = nn//nodemul
   p['last_convT'] = conv2dT_params(nn, channels, ker, do_bias)
   p['bnn']=utils.bnparams(nn0)
   flat_params = utils.cast(utils.flatten(p))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, stride,label):
      o = F.relu(x, inplace=True)
      gain = F.linear(label, params[base+'.fc_label.weight'], params[base+'.fc_label.bias']).view(label.size(0), -1, 1, 1)
      bias = F.linear(label, params[base+'.fc_label1.weight'], params[base+'.fc_label1.bias']).view(label.size(0), -1, 1, 1)
      o = o*gain+bias 
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

   def group(o, params, base, mode, stride=2,label=None):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1,label)
      return o

   def f(input,label, params, mode):
      # print(label.shape)
      # print( params['fc_label.weight'].shape)
      # print( params['fc_label.bias'].shape)
      # gain = self.fc2(labels).view(labels.size(0), -1, 1, 1)
      # # bias = self.fc3(labels).view(labels.size(0), -1, 1, 1)
      # gain = F.linear(label, params['fc_label.weight'], params['fc_label.bias']).view(label.size(0), -1, 1, 1)
      # bias = F.linear(label, params['fc_label1.weight'], params['fc_label1.bias']).view(label.size(0), -1, 1, 1) 
      # y_ =  F.leaky_relu(y_, negative_slope=0.2, inplace=True)
      # input = torch.cat([input, y_], 1)
      o = F.linear(input, params['proj.weight'], params['proj.bias'])
      o = o.view(input.size(0), nn0, sz, sz)
      # o = utils.batch_norm(o, params,'bnn', mode)
      # o = F.relu(o, inplace=True)
      
      # print(o.shape)
      # o = o*gain+bias
      for d in range(depth-1):
        o = group(o, params, 'group%d'%d, mode,label=label)      
      o = F.relu(o, inplace=True)
      o = F.conv_transpose2d(o, params['last_convT.w'], params.get('last_convT.b'), stride=2,
                             padding=padding, output_padding=output_padding)
      o = torch.tanh(o)
      return o

   return f, flat_params



##-------add conditional G and D------------------###

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

def simple_mlp(input_dim,output_dim,requires_grad):
   p = { 'fc': utils.linear_params(input_dim, 100) }
   p['fc1'] =  utils.linear_params(100, 1)
   flat_params = utils.cast(utils.flatten(p))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
   def f(input, params, mode):
      o = F.linear(input, params['fc.weight'], params['fc.bias'])
      o = F.relu(o, inplace=True)
      o = F.linear(o, params['fc1.weight'], params['fc1.bias'])    
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

#-------------------------------------------------------------
def resnet4_D(nn, imgsz,
              channels,    # 1: gray-scale, 3: color
              norm_type,  # 'bn', 'none'
              requires_grad,
              depth,
              do_bias=True):             
   # depth =4
   # depth = 4
   ker = 3
   padding = (ker-1)//2
   count = 1

   def gen_group0_params(no):
      ni = channels
      return { 'block0' : {
         'conv0': conv2d_params(ni, no, ker, do_bias), 
         'conv1': conv2d_params(no, no, ker, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }}

   def gen_resnet_D_block_params(ni, no, k, norm_type, do_bias):
      return {
         'conv0': conv2d_params(ni, ni, k, do_bias), 
         'conv1': conv2d_params(ni, no, k, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no):
       return {'block%d' % i: gen_resnet_D_block_params(ni if i == 0 else no, no, ker, norm_type, do_bias) for i in range(count)}
   # depth1=depth-1
   # print(imgsz)
   sz = imgsz // (2**depth)
   p = { 'fc': utils.linear_params(sz*sz*nn*(2**(depth-1)), 1) }
   nn1 = nn
   p['group0']=gen_group0_params(nn)
   for d in range(depth-1):
      p['group%d'%(d+1)] = gen_group_params(nn1,  nn1*2)
      nn1 = nn1*2
      
   flat_params = utils.cast(utils.flatten(p))
   # print(sz*sz*nn*(2**(depth-1)))

   
   # flat_params = utils.cast(utils.flatten({
   #      'group0': gen_group0_params(nn),
   #      'group1': gen_group_params(nn,   nn*4),
   #      'group2': gen_group_params(nn*4, nn*8),
   #    #   'group1': gen_group_params(nn,   nn*2),
   #    #   'group2': gen_group_params(nn*2, nn*4),
   #    #   'group3': gen_group_params(nn*16, nn*16),        
   #      'fc': utils.linear_params(sz*sz*nn*8, 1),
   # }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_downsample, is_first):
      o = x
      if not is_first:
         o = F.relu(o, inplace=True)   
      o = F.conv2d(x, params[base+'.conv0.w'], params.get(base+'conv0.b'), padding=padding)
      o = F.relu(o, inplace=True)      
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'conv1.b'), padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn', mode)
 
      if do_downsample:
         o = F.avg_pool2d(o,2)
         x = F.avg_pool2d(x,2)
      
      if base + '.convdim' in params:
         return o + F.conv2d(x, params[base + '.convdim'])
      else:
         return o + x


   def group(o, params, base, mode, do_downsample, is_first=False):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, 
                   do_downsample=(do_downsample and i == count-1), 
                   is_first=(is_first and i == 0))                   
      return o

   def f(input, params, mode):
      # print('input shape{}'.format(input.shape))
      o = group(input, params, 'group0', mode, do_downsample=True, is_first=True)
      for d in range(depth-1):
         o = group(o, params, 'group%d'%(d+1), mode, do_downsample=True)
      # o = group(o, params, 'group1', mode, do_downsample=True)
      # o = group(o, params, 'group2', mode, do_downsample=True)
      # o = group(o, params, 'group3', mode, do_downsample=True)      
      # print(o.shape)
      o = F.relu(o, inplace=True)
      o = o.view(o.size(0), -1)
      # print(o.shape)
      
      o = F.linear(o, params['fc.weight'], params['fc.bias'])
      return o

   return f, flat_params   
   
#-------------------------------------------------------------
def resnet4_G(input_dim, n0g, imgsz, channels,
             norm_type,  # 'bn', 'none'
             requires_grad,
             depth,
             do_bias=True):         
   # depth = 4
   # depth = 3
   ker = 3
   padding = (ker-1)//2
   count = 1

   def gen_resnet_G_block_params(ni, no, k, norm_type, do_bias):
      return {
         'conv0': conv2d_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, k, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no):
       return {'block%d' % i: gen_resnet_G_block_params(ni if i == 0 else no, no, ker, norm_type, do_bias) for i in range(count)}
   depth1 = depth-1
   nn0 = n0g * (2**depth1)
   nn = n0g * (2**depth1); sz = imgsz // (2**depth1)
   p = { 'proj': utils.linear_params(input_dim, nn*sz*sz) }
   for d in range(depth1):
      p['group%d'%d] = gen_group_params(nn,  nn//2)
      nn = nn//2
   # p['group%d'%depth1] = gen_group_params(nn,  nn)
   p['last_conv']= conv2d_params(nn, channels, ker, do_bias)
   flat_params = utils.cast(utils.flatten(p))
   # print('nn {}'.format(nn))
   # print('nn0 {}'.format(nn0))
   # flat_params = utils.cast(utils.flatten({
   #      'proj': utils.linear_params(input_dim, nn*sz*sz),
   #      'group0': gen_group_params(nn,    nn//2),
   #      'group1': gen_group_params(nn//2, nn//4),
   #      'group2': gen_group_params(nn//4, nn//8),
   #    #   'group3': gen_group_params(nn//8, nn//8),        
   #      'last_conv': conv2d_params(nn//8, channels, ker, do_bias),
   # }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_upsample):
      o = F.relu(x, inplace=True)
      if do_upsample:
        o = F.interpolate(o, scale_factor=2, mode='nearest')
            
      o = F.conv2d(o, params[base+'.conv0.w'], params.get(base+'.conv0.b'), padding=padding)
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'.conv1.b'), padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn', mode)
         
      xo = F.conv2d(x, params[base + '.convdim']) 
      if do_upsample:
         return o + F.interpolate(xo, scale_factor=2, mode='nearest')
      else:
         return o + xo
 
   def group(o, params, base, mode, do_upsample):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, do_upsample if i == 0 else False)
      return o

   def show_shape(o, msg=''):
      print(o.size(), msg)

   def f(input, params, mode):
      o = F.linear(input, params['proj.weight'], params['proj.bias'])
      o = o.view(input.size(0), nn0, sz, sz)
      for d in range(depth1):
         o = group(o, params, 'group%d'%d, mode, do_upsample=True)
      # o = group(o, params, 'group%d'%depth1, mode, do_upsample=True)
      # # print('o shape is {}'.format(o.shape))
      # o = group(o, params, 'group1', mode, do_upsample=True)
      # o = group(o, params, 'group2', mode, do_upsample=True)
      # o = group(o, params, 'group3', mode, do_upsample=True)
      # if mode:
      #    o1 = o[0]
      #    o2= o1.unsqueeze(1)
      #    vizG.images(o2,opts=dict(title='Random images', caption='How random.'))
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params['last_conv.w'], params.get('last_conv.b'), padding=padding)
      o = torch.tanh(o)
      # print('G o shape{}'.format(o.shape))
      return o

   return f, flat_params   
   
def resnet1024_G(input_dim, n0g, imgsz, channels,
             norm_type,  # 'bn', 'none'
             requires_grad,
             depth,
             do_bias=True):         
   # depth = 4
   # depth = 3
   ker = 3
   padding = (ker-1)//2
   count = 1

   def gen_resnet_G_block_params(ni, no, k, norm_type, do_bias):
      return {
         'conv0': conv2d_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, k, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no):
       return {'block%d' % i: gen_resnet_G_block_params(ni if i == 0 else no, no, ker, norm_type, do_bias) for i in range(count)}
   depth1 = depth-1
   nn0 = n0g * (2**depth1)
   nn = n0g * (2**depth1); sz = imgsz // (2**depth)
   p = { 'proj': utils.linear_params(input_dim, nn*sz*sz) }
   for d in range(depth1):
      p['group%d'%d] = gen_group_params(nn,  nn//2)
      nn = nn//2
   p['group%d'%depth1] = gen_group_params(nn,  nn)
   p['last_conv']= conv2d_params(nn, channels, ker, do_bias)
   flat_params = utils.cast(utils.flatten(p))
   # print('nn {}'.format(nn))
   # print('nn0 {}'.format(nn0))
   # flat_params = utils.cast(utils.flatten({
   #      'proj': utils.linear_params(input_dim, nn*sz*sz),
   #      'group0': gen_group_params(nn,    nn//2),
   #      'group1': gen_group_params(nn//2, nn//4),
   #      'group2': gen_group_params(nn//4, nn//8),
   #    #   'group3': gen_group_params(nn//8, nn//8),        
   #      'last_conv': conv2d_params(nn//8, channels, ker, do_bias),
   # }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_upsample):
      o = F.relu(x, inplace=True)
      if do_upsample:
        o = F.interpolate(o, scale_factor=2, mode='nearest')
            
      o = F.conv2d(o, params[base+'.conv0.w'], params.get(base+'.conv0.b'), padding=padding)
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'.conv1.b'), padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn', mode)
         
      xo = F.conv2d(x, params[base + '.convdim']) 
      if do_upsample:
         return o + F.interpolate(xo, scale_factor=2, mode='nearest')
      else:
         return o + xo
 
   def group(o, params, base, mode, do_upsample):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, do_upsample if i == 0 else False)
      return o

   def show_shape(o, msg=''):
      print(o.size(), msg)

   def f(input, params, mode):
      o = F.linear(input, params['proj.weight'], params['proj.bias'])
      o = o.view(input.size(0), nn0, sz, sz)
      for d in range(depth1):
         o = group(o, params, 'group%d'%d, mode, do_upsample=True)
      o = group(o, params, 'group%d'%depth1, mode, do_upsample=True)
      # # print('o shape is {}'.format(o.shape))
      # o = group(o, params, 'group1', mode, do_upsample=True)
      # o = group(o, params, 'group2', mode, do_upsample=True)
      # o = group(o, params, 'group3', mode, do_upsample=True)
      # if mode:
      #    o1 = o[0]
      #    o2= o1.unsqueeze(1)
      #    vizG.images(o2,opts=dict(title='Random images', caption='How random.'))
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params['last_conv.w'], params.get('last_conv.b'), padding=padding)
      o = torch.tanh(o)
      # print('G o shape{}'.format(o.shape))
      return o

   return f, flat_params   
   

#-------------------------------------------------------------
def resnet4_D_large(nn, imgsz,
              channels,    # 1: gray-scale, 3: color
              norm_type,  # 'bn', 'none'
              requires_grad,
              depth,
              do_bias=True):             
   # depth =4
   # depth = 4
   ker = 3
   padding = (ker-1)//2
   count = 1

   def gen_group0_params(no):
      ni = channels
      return { 'block0' : {
         'conv0': conv2d_params(ni, no, ker, do_bias), 
         'conv1': conv2d_params(no, no, ker, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }}

   def gen_resnet_D_block_params(ni, no, k, norm_type, do_bias):
      return {
         'conv0': conv2d_params(ni, ni, k, do_bias), 
         'conv1': conv2d_params(ni, no, k, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no):
       return {'block%d' % i: gen_resnet_D_block_params(ni if i == 0 else no, no, ker, norm_type, do_bias) for i in range(count)}
   # depth1=depth-1
   sz = imgsz // (2**depth)
   p = { 'fc': utils.linear_params(sz*sz*nn*(2**(depth-1)), 1) }
   nn1 = nn
   p['group0']=gen_group0_params(nn)
   for d in range(depth-1):
      p['group%d'%(d+1)] = gen_group_params(nn1,  nn1*2)
      nn1 = nn1*2
      
   flat_params = utils.cast(utils.flatten(p))
   # print(sz*sz*nn*(2**(depth-1)))

   
   # flat_params = utils.cast(utils.flatten({
   #      'group0': gen_group0_params(nn),
   #      'group1': gen_group_params(nn,   nn*4),
   #      'group2': gen_group_params(nn*4, nn*8),
   #    #   'group1': gen_group_params(nn,   nn*2),
   #    #   'group2': gen_group_params(nn*2, nn*4),
   #    #   'group3': gen_group_params(nn*16, nn*16),        
   #      'fc': utils.linear_params(sz*sz*nn*8, 1),
   # }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_downsample, is_first):
      o = x
      if not is_first:
         o = F.relu(o, inplace=True)   
      o = F.conv2d(x, params[base+'.conv0.w'], params.get(base+'conv0.b'), padding=padding)
      o = F.relu(o, inplace=True)      
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'conv1.b'), padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn', mode)
 
      if do_downsample:
         o = F.avg_pool2d(o,2)
         x = F.avg_pool2d(x,2)
      
      if base + '.convdim' in params:
         return o + F.conv2d(x, params[base + '.convdim'])
      else:
         return o + x


   def group(o, params, base, mode, do_downsample, is_first=False):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, 
                   do_downsample=(do_downsample and i == count-1), 
                   is_first=(is_first and i == 0))                   
      return o

   def f(input, params, mode):
      # print('input shape{}'.format(input.shape))
      o = group(input, params, 'group0', mode, do_downsample=True, is_first=True)
      for d in range(depth-1):
         o = group(o, params, 'group%d'%(d+1), mode, do_downsample=True)
      # o = group(o, params, 'group1', mode, do_downsample=True)
      # o = group(o, params, 'group2', mode, do_downsample=True)
      # o = group(o, params, 'group3', mode, do_downsample=True)      
      # print(o.shape)
      o = F.relu(o, inplace=True)
      o = o.view(o.size(0), -1)
      # print(o.shape)
      
      o = F.linear(o, params['fc.weight'], params['fc.bias'])
      return o

   return f, flat_params   
   
#-------------------------------------------------------------
def resnet4_G_large(input_dim, n0g, imgsz, channels,
             norm_type,  # 'bn', 'none'
             requires_grad,
             depth,
             do_bias=True):         
   # depth = 4
   # depth = 3
   ker = 3
   padding = (ker-1)//2
   count = 1

   def gen_resnet_G_block_params(ni, no, k, norm_type, do_bias):
      return {
         'conv0': conv2d_params(ni, no, k, do_bias), 
         'conv1': conv2d_params(no, no, k, do_bias), 
         'convdim': utils.conv_params(ni, no, 1), 
         'bn': utils.bnparams(no) if norm_type == 'bn' else None
      }

   def gen_group_params(ni, no):
       return {'block%d' % i: gen_resnet_G_block_params(ni if i == 0 else no, no, ker, norm_type, do_bias) for i in range(count)}
   depth1 = depth-1
   nn0 = n0g * (2**depth1)
   nn = n0g * (2**depth1); sz = imgsz // (2**depth1)
   p = { 'proj': utils.linear_params(input_dim, nn*sz*sz) }
   for d in range(depth1):
      p['group%d'%d] = gen_group_params(nn,  nn//2)
      nn = nn//2
   # p['group%d'%depth1] = gen_group_params(nn,  nn)
   p['last_conv']= conv2d_params(nn, channels, ker, do_bias)
   flat_params = utils.cast(utils.flatten(p))
   # print('nn {}'.format(nn))
   # print('nn0 {}'.format(nn0))
   # flat_params = utils.cast(utils.flatten({
   #      'proj': utils.linear_params(input_dim, nn*sz*sz),
   #      'group0': gen_group_params(nn,    nn//2),
   #      'group1': gen_group_params(nn//2, nn//4),
   #      'group2': gen_group_params(nn//4, nn//8),
   #    #   'group3': gen_group_params(nn//8, nn//8),        
   #      'last_conv': conv2d_params(nn//8, channels, ker, do_bias),
   # }))

   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_upsample):
      o = F.relu(x, inplace=True)
      if do_upsample:
        o = F.interpolate(o, scale_factor=2, mode='nearest')
            
      o = F.conv2d(o, params[base+'.conv0.w'], params.get(base+'.conv0.b'), padding=padding)
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params[base+'.conv1.w'], params.get(base+'.conv1.b'), padding=padding)
      if norm_type == 'bn':
         o = utils.batch_norm(o, params, base + '.bn', mode)
         
      xo = F.conv2d(x, params[base + '.convdim']) 
      if do_upsample:
         return o + F.interpolate(xo, scale_factor=2, mode='nearest')
      else:
         return o + xo
 
   def group(o, params, base, mode, do_upsample):
      for i in range(count):
         o = block(o, params, '%s.block%d' % (base,i), mode, do_upsample if i == 0 else False)
      return o

   def show_shape(o, msg=''):
      print(o.size(), msg)

   def f(input, params, mode):
      o = F.linear(input, params['proj.weight'], params['proj.bias'])
      o = o.view(input.size(0), nn0, sz, sz)
      for d in range(depth1):
         o = group(o, params, 'group%d'%d, mode, do_upsample=True)
      # o = group(o, params, 'group%d'%depth1, mode, do_upsample=True)
      # # print('o shape is {}'.format(o.shape))
      # o = group(o, params, 'group1', mode, do_upsample=True)
      # o = group(o, params, 'group2', mode, do_upsample=True)
      # o = group(o, params, 'group3', mode, do_upsample=True)
      # if mode:
      #    o1 = o[0]
      #    o2= o1.unsqueeze(1)
      #    vizG.images(o2,opts=dict(title='Random images', caption='How random.'))
      o = F.relu(o, inplace=True)
      o = F.conv2d(o, params['last_conv.w'], params.get('last_conv.b'), padding=padding)
      o = torch.tanh(o)
      # print('G o shape{}'.format(o.shape))
      return o

   return f, flat_params   
   

def biggan_G(G_ch=64, dim_z=100, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=128, hier=False,
               cross_replica=False, mybn=False,
               G_activation=torch.nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn',requires_grad=True,
               **kwargs):
   attention = G_attn
   ch = G_ch
   count=0            
   arch = G_arch(ch, attention)[resolution]
   if hier:
      # Number of places z slots into
      num_slots = len(arch['in_channels']) + 1
      z_chunk_size = (dim_z // num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      dim_z = z_chunk_size *  num_slots
   else:
      num_slots = 1
      z_chunk_size = 0
   
   if G_param == 'SN':
      which_conv = functools.partial(biggan_layers.SN_conv2d_icfg,sn=True,k=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=SN_eps)
      which_linear = functools.partial(biggan_layers.SN_linear_icfg,sn=True,num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=SN_eps)
   else:
      which_conv = functools.partial(biggan_layers.SN_conv2d_icfg,k=3,poadding=1,sn=False)
      which_linear = functools.partial(biggan_layers.SN_linear_icfg,sn=False)

   which_embedding = functools.partial(biggan_layers.SN_embedding_icfg,sn=False)
   bn_linear = (functools.partial(which_linear, do_bias=False) if G_shared
                 else which_embedding)
   which_bn = functools.partial(biggan_layers.ccbn_icfg,input_size=(shared_dim + z_chunk_size if G_shared
                                      else n_classes),
                          which_linear=bn_linear,
                          cross_replica=cross_replica,
                          mybn=mybn,
                          norm_style=norm_style,
                          eps=BN_eps)
                          


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
   # shared = (which_embedding(n_classes, shared_dim) if G_shared 
   #                  else None)
   p={'shared':which_embedding(n_classes, shared_dim)[1]}         
   f={'shared':which_embedding(n_classes, shared_dim)[0]}
    # First linear layer
   # linear = which_linear(dim_z // num_slots,arch['in_channels'][0] * (bottom_width **2))
   p['first_linear']=which_linear(dim_z // num_slots,arch['in_channels'][0] * (bottom_width **2))[1]
   f['first_linear']=which_linear(dim_z // num_slots,arch['in_channels'][0] * (bottom_width **2))[0]

   for index in range(len(arch['out_channels'])):
      f['block%d'%(index)],p['block%d'%(index)]=biggan_layers.GBblock_icfg(in_channels=arch['in_channels'][index],
                             out_channels=arch['out_channels'][index],
                             which_conv=which_conv,
                             which_bn=which_bn,
                             activation=G_activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if arch['upsample'][index] else None))
      # If attention on this block, attach it to the end
      count=count+1
      if arch['attention'][arch['resolution'][index]]:
      #   print('attention')
      #   print(arch)
        f['attention%d'%(index)],p['attention%d'%(index)]=biggan_layers.SN_Attention_icfg(arch['out_channels'][index],which_conv)
      #   count=count+1

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here

   p['output_bn']=biggan_layers.BN_icfg(arch['out_channels'][-1],
                                                cross_replica=cross_replica,
                                                mybn=mybn)[1]
   p['output_conv']=which_conv(arch['out_channels'][-1], 3)[1]
   f['output_bn']=biggan_layers.BN_icfg(arch['out_channels'][-1],
                                                cross_replica=cross_replica,
                                                mybn=mybn)[0] 
   f['output_conv']=which_conv(arch['out_channels'][-1], 3)[0]



   flat_params = utils.cast(utils.flatten(p))
   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
   def f_G(z,y,params,training):
   #   print(y)
     y = f['shared'](y,params,'shared',training=training)
   #   print(y)
     if hier:
        zs = torch.split(z, z_chunk_size, 1)
        z = zs[0]
        ys = [torch.cat([y, item], 1) for item in zs[1:]]
     else:
      #   print(y)
        ys = [y] * count
      #   print(ys)
      
    # First linear layer
   #   print(z.shape)
   #   print(params['first_linear.w'].shape)
     h = f['first_linear'](z,params,'first_linear',training=training)
    # Reshape
     h = h.view(h.size(0), -1, bottom_width, bottom_width)
    
    # Loop over blocks
     for index in range(count):
      #   print(ys[index].shape)
        h = f['block%d'%(index)](h, ys[index],params,'block%d'%(index),training)
        if arch['attention'][arch['resolution'][index]]:
          h = f['attention%d'%(index)](h,params,'attention%d'%(index),training)
        
    # Apply batchnorm-relu-conv-tanh at output
     h = f['output_bn'](h,params,'output_bn',training=training)
     h = G_activation(h)
     h = f['output_conv'](h,params,'output_conv',training=training)
     return torch.tanh(h)

   return f_G, flat_params

def biggan_D( D_ch=64, D_wide=True, resolution=64,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=torch.nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN',requires_grad=True,**kwargs):
   ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    # Resolution
   resolution = resolution
   attention = D_attn
   arch = D_arch(ch, attention)[resolution]
   if D_param == 'SN':
      which_conv = functools.partial(biggan_layers.SN_conv2d_icfg,
                          k=3, padding=1,sn=True,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=SN_eps)
      which_linear = functools.partial(biggan_layers.SN_linear_icfg,sn=True,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=SN_eps)
      which_embedding = functools.partial(biggan_layers.SN_embedding_icfg,sn=True,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=SN_eps)
   
   
   p={'outer_linear':which_linear(arch['out_channels'][-1], output_dim)[1]}
   f={'outer_linear':which_linear(arch['out_channels'][-1], output_dim)[0]}
   p['outer_emb']=which_embedding(n_classes, arch['out_channels'][-1])[1]
   f['outer_emb']=which_embedding(n_classes, arch['out_channels'][-1])[0]

    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
   count = 0
   for index in range(len(arch['out_channels'])):
      f['block%d'%(index)],p['block%d'%(index)]=biggan_layers.DBblock_icfg(in_channels=arch['in_channels'][index],
                       out_channels=arch['out_channels'][index],
                       which_conv=which_conv,
                       wide=D_wide,
                       activation=D_activation,
                       preactivation=(index > 0),
                       downsample=(torch.nn.AvgPool2d(2) if arch['downsample'][index] else None))
      # If attention on this block, attach it to the end
      count=count+1
      if arch['attention'][arch['resolution'][index]]:
        f['attention%d'%(index)],p['attention%d'%(index)]=biggan_layers.SN_Attention_icfg(arch['out_channels'][index],
                                             which_conv)
    # Turn self.blocks into a ModuleList so that it's all properly registered.

   flat_params = utils.cast(utils.flatten(p))
   if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
   def f_D(x,params,y=None):
      # Stick x into h for cleaner for loops without flow control
     h = x
   #   print(h.shape)
    # Loop over blocks
     for index in range(count):
       h = f['block%d'%(index)](h,params,'block%d'%(index))
       if arch['attention'][arch['resolution'][index]]:
         h = f['attention%d'%(index)](h,params,'attention%d'%(index))
    
    # Apply global sum pooling as in SN-GAN
     h = torch.sum(D_activation(h), [2, 3])
    # Get initial class-unconditional output
     out = h = f['outer_linear'](h,params,'outer_linear')
    # Get projection of final featureset onto class vectors and add to evidence
     out = out 
   #   +torch.sum(f['outer_emb'](y,params,'outer_emb') * h, 1, keepdim=True)
     return out
   return f_D,flat_params

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