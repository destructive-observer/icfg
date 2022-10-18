import torch.distributions as D
import numpy as np
import torch

def sample_mog(batch_size, n_mixture=8, std=0.01, radius=1.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    # xs = torch.from_numpy(xs)
    # ys = torch.from_numpy(ys)
    # print(xs)
    # print(torch.ones(n_mixture)/8)
    cat = D.Categorical(torch.ones(n_mixture))
    # print(cat.sample([8]))
    # m = D.Categorical(torch.tensor([ 0.1, 0.1, 0.1, 0.1 ]))
    # print(m.sample())  # equal probability of 0, 1, 2, 3


    diag_a = torch.diag_embed(torch.tensor([std,std]))
    # print(diag_a)
    stack_a = [[xi, yi] for xi, yi in zip(xs.ravel(), ys.ravel())]
    # print(stack_a)
    stack_b = torch.Tensor(stack_a)
    # print(stack_b.shape)
    comps = D.MultivariateNormal(stack_b, diag_a)
    # comps = [D.MultivariateNormal(torch.tensor([xi, yi]), diag_a) for xi, yi in zip(xs.ravel(), ys.ravel())]
    # print(comps)
    data = D.MixtureSameFamily(cat, comps)
    # print(data)


    # mix = D.Categorical(torch.rand(3,5))
    # comp1 = D.Independent(D.Normal(
    #         torch.randn(3,5,2), torch.rand(3,5,2)), 1)
    # gmm = D.MixtureSameFamily(mix, comp1)    
    # print(gmm.sample([3,5,2]))
    return data.sample([batch_size])

# data = sample_mog(512)
# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(5,5))
# print(data.shape)
# plt.scatter(data[:,0],data[:,1], c='g', edgecolor='none')
# # plt.xlim((-10,10))
# # plt.ylim((-10,10))
# # plt.xlabel([])
# # plt.ylabel([])
# plt.axis('off')
# plt.title('sine wave')
# #使用show展示图像
# plt.show()