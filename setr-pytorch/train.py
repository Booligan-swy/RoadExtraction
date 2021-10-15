import torch
import torch.nn as nn
import os
import torch.utils.data as Data
from data_preprocessing import ImageFolder
from time import time
# from loss import dice_bce_loss
from framework import MyFrame
from tqdm import tqdm
from setr.SETR import SETR_Naive

SHAPE = (512,512)
ROOT = 'data/datasets512/train/'

##取名字中包含sat的所有样本名
imagelist = filter(lambda x:x.find('sat') != -1,os.listdir(ROOT))


##得到样本图片的索引序列号，例如99,998,999....
trainlist = []
for item in imagelist:
    x = item
    y = x[:-8]
    trainlist.append(y)


NAME = 'SETR_m_v1'
BATCHSIZE_PER_CARD = 4


batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
print("batchsize:",batchsize)

dataset_img = ImageFolder(trainlist=trainlist,root=ROOT)
# print(dataset_img.__len__())

data_loader = Data.DataLoader(
    dataset=dataset_img,
    batch_size=batchsize,
    shuffle=False,
    # drop_last=True,
    num_workers=0
)
print(len(data_loader))
model = SETR_Naive(img_dim = 512,
        patch_dim = 16,
        num_channels = 3,
        num_classes = 2,
        embedding_dim = 1024,
        num_heads = 16,
        num_layers = 2,
        hidden_dim = 1024,
        dropout_rate=0.1,
        attn_dropout_rate=0.1)
solver = MyFrame(model,nn.MSELoss(),2e-4)
mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.

for epoch in range(1,total_epoch+1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img,mask in tqdm(data_loader_iter,ncols=100,total=len(data_loader_iter)):
        # print("image.shape:",img.shape," mask.shape",mask.shape)
        solver.set_input(img,mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)


    print('epoch:',epoch)
    print('********',file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic),file=mylog)
    print('train_loss:', train_epoch_loss,file=mylog)
    print('SHAPE:', SHAPE,file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch,file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

print('Finish!',file=mylog)
print('Finish!')
mylog.close()







