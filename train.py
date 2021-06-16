from fracnet_dataset import FracNetTrainDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from model import ResUNet
from utils import logger, weights_init, metrics, common, loss
from transforms import Mask, ToTensor, Window, MinMaxNorm, RandomResize, RandomFlip_LR, RandomFlip_UD
import os
import numpy as np
from collections import OrderedDict

def val(model, val_loader, loss_func, n_labels):
    model.training = False
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice': val_dice.avg[1]})
    return val_log

def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    # model.is_training = True
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
        # print(target)
        target = common.to_one_hot_3d(target,n_labels)
        # target_ndarray=np.array(target[0])
        # target_index0=np.where(target_ndarray==0)
        # target_index1=np.where(target_ndarray==1)
        # print(target_index0,target_index1)
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss0 = loss_func(output[0], target)
        loss1 = loss_func(output[1], target)
        loss2 = loss_func(output[2], target)
        loss3 = loss_func(output[3], target)

        loss = loss3  +  alpha * (loss0 + loss1 + loss2)
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss3.item(),data.size(0))
        train_dice.update(output[3], target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice': train_dice.avg[1]})
    return val_log

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    batch_size = 4
    num_workers = 32
    train_image_dir ='data/train/image'
    train_label_dir ='data/train/label'
    val_image_dir ='data/val/image'
    val_label_dir ='data/val/label'
    
    transforms_train =[
                Mask(0.1),
                Window(-200, 1000),
                MinMaxNorm(-200, 1000),
                RandomResize([0.8,1.2],[0.8,1.2],[0.8,1.2]),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                ]

    transforms_val=[
                Window(-200, 1000),
                MinMaxNorm(-200, 1000),
    ]
                
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
                                   transforms=transforms_train)
    train_loader = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
                                                  num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
                                 transforms=transforms_val)
    val_loader = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
                                                num_workers)
    
    
    model = ResUNet(in_channel=1, out_channel=args.n_labels).to(device)
    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
 
    loss = loss.TverskyLoss()

    log = logger.Train_Logger(save_path,"train_log")

    best = [0,0] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4 # 深监督衰减系数初始值
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()    

