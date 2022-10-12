import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size,shuffle=True,)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='nearest').type(dtype)
    def get_pred(x):
        # print(x)
        if resize:
            x = up(x)
            # print(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        # print(batchv)
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            # print(self.orig[index][0])
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as T

    # train_Data = dset.CIFAR10(root='../', download=True,
    #                          transform=T.Compose([
    #                              T.Scale(32),
    #                              T.ToTensor(),
    #                              T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )
    #  datasets.SVHN(dataroot, split='train', transform=tr, download=do_download)
    # train_Data = dset.SVHN(root='../', download=True,split='train',
    #                          transform=T.Compose([
    #                              T.Scale(32),
    #                              T.ToTensor()
    #                          ])
    # )
    
    def to_pm1(input):
        return input*2-1
    # train_Data = dset.LSUN('../LSUN/', classes=['tower_train'], transform=T.Compose([ T.Resize(64), T.CenterCrop(64),T.ToTensor(),to_pm1 ]))
    # train_Data = dset.MNIST('../', train=True, transform=T.Compose([                                      
    #                              T.ToTensor()
    #                              ,T.Lambda(lambda x:x.repeat(3,1,1)),to_pm1]), download=True)
                                #  T.Lambda(lambda x:x.repeat(3,1,1)),
                                 
   
    # def get_tr(dataset, is_train, do_augment):
    #     if dataset == 'ImageNet':  
    #         tr = T.Compose([ T.Resize(256), T.CenterCrop(224) ]) 
    #     elif dataset == 'MNIST':
    #         tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32
    #     elif dataset.endswith('64'):
    #         tr = T.Compose([ T.Resize(64), T.CenterCrop(64) ])
    #     else:
    #         tr = T.Compose([ ])
    #     if do_augment:
    #         tr = T.Compose([
    #           tr, 
    #           T.Pad(4, padding_mode='reflect'),
    #           T.RandomHorizontalFlip(),
    #           T.RandomCrop(32),
    #     ])
    #     return T.Compose([tr, T.ToTensor(), to_pm1 ])
    # tr = get_tr('SVHN',False,do_augment=True)
    train_Data=dset.ImageFolder(root='./data',transform=T.Compose([
                                #  T.Pad(2),

                                #  T.Resize(64), T.CenterCrop(64) ,
                                 T.ToTensor(),
                                #  T.Lambda(lambda x:x.repeat(3,1,1)),
                                 to_pm1
                                #  T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))

    # IgnoreLabelDataset(train_Data)
    # print(train_Data)
    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(train_Data), cuda=True, batch_size=32, resize=True, splits=10))
