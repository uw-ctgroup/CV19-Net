import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import time 
from Dataset.CXRDataset import ChestXrayDataSet
from Model.Network import DenseNet121

#########################################################################
def train(BATCH_SIZE = 150, num_iterations = 100, index = '001'):
    """
    index: the index for ensemble model and its corresponding data list 
    """

    N_CLASSES = 2 
    CLASS_NAMES = ['COVID', 'NON-COVID']
    DATA_DIR = './Data/images'
    CKPT_PATH = './Model_weights/model_pretrain.pth.tar' # the model pretrained on NIH dataset 
    TRAIN_IMAGE_LIST = './Data/labels/labels_' + index + '/labels_covid_train.txt'
    TEST_IMAGE_LIST = './Data/labels/labels_' + index + '/labels_covid_val.txt'
    
    modelname = 'model_train_' + index   
    savefig_name = 'Loss_' + modelname + '.png'
    train_loss_name = 'TrainLoss_' + modelname + '.npy'
    val_loss_name = 'ValLoss_' + modelname + '.npy'

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
    
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        state_dict = removekey(checkpoint['state_dict'], \
            ['module.densenet121.classifier.0.weight', 'module.densenet121.classifier.0.bias'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    # Build the traininig and validation Dataloader 
    transResize = 256
    transCrop = 224
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    transformList = []
    transformList.append(transforms.RandomResizedCrop(transCrop, scale=(0.8, 1.0))) #0.08
    transformList.append(transforms.RandomHorizontalFlip(p=0.5))
    transformList.append(transforms.RandomRotation(30, resample=2))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence = transforms.Compose(transformList)
 
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_IMAGE_LIST,
                                    transform=transformSequence)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=6, pin_memory=True, drop_last=True)

    transformList_test = []
    transformList_test.append(transforms.Resize(transCrop))
    transformList_test.append(transforms.ToTensor())
    transformList_test.append(normalize)      
    transformSequence_test=transforms.Compose(transformList_test)
    
    val_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform = transformSequence_test)
   
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=6, pin_memory=True, drop_last=False)
    
    # Define the optimizer 
    optimizer = optim.Adam([
        {'params': model.module.densenet121.features.parameters()}, \
            {'params': model.module.densenet121.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}], \
                lr=6e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, mode = 'min', min_lr = 1e-5)
         
    # Define the LOSS 
    loss = torch.nn.BCELoss(reduction='mean')
    
    # Training mode 
    losstrain_list = []
    lossVal_list = []
    loss_min = 1000
    save_epoch = 0
    nonsave_epoch = 0
    
    for epochID in range(num_iterations):     
        loss_train = epochTrain(model, train_loader, optimizer, loss)
        lossVal, losstensor, auroc_score = epochVal(model, val_loader, optimizer, loss)
        losstrain_list.append(loss_train)
        lossVal_list.append(lossVal)
        
        scheduler.step(losstensor.item())

        if lossVal < loss_min:
            loss_min = lossVal
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossVal}, './Model_weights/' + modelname + '.pth.tar')
            print ('Epoch [' + str(epochID + 1) + '] [save] Train loss= ' + str(losstrain_list[-1]))
            print ('Epoch [' + str(epochID + 1) + '] [save] Val loss= ' + str(lossVal))     
            save_epoch = epochID + 1
            
        else:
            print ('Epoch [' + str(epochID + 1) + '] [----] Train loss= ' + str(losstrain_list[-1]))
            print ('Epoch [' + str(epochID + 1) + '] [----] Val loss= ' + str(lossVal))
            nonsave_epoch = epochID + 1 

        if nonsave_epoch - save_epoch >= 10:
            # For early stopping 
            break
            
        torch.cuda.empty_cache()
        print('----------------------------------------------------------------------')

        if epochID % 10 == 9:
            fig = plt.subplots(1, 1, figsize=(5, 5))
            plt.plot(losstrain_list, label='Training Loss')
            plt.plot(lossVal_list, label='Validation Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
            plt.close()

    np.save(train_loss_name, losstrain_list)
    np.save(val_loss_name, lossVal_list)

#-------------------------------------------------------------------------------- 
def epochTrain (model, dataLoader, optimizer, loss):
    model.train()
    losstrain = 0
    losstrainNorm = 0  
    for batchID, (inp, target) in enumerate (dataLoader):
        target  = target.cuda(async = True)
        input_var = Variable(inp)
        vartarget = Variable(target)     
        output = model(input_var)
            
        lossvalue = loss(output, vartarget)
                       
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()
        losstrain += lossvalue.data
        losstrainNorm += 1 
        
    outLoss = losstrain / losstrainNorm
    return outLoss.cpu().detach().numpy()
            
    #-------------------------------------------------------------------------------- 
def epochVal (model, dataLoader, optimizer, loss):
    model.eval()
    lossVal = 0
    lossValNorm = 0    
    losstensorMean = 0
    gt = torch.zeros(1, N_CLASSES)
    gt = gt.cuda()
    pred = torch.zeros(1, N_CLASSES)
    pred = pred.cuda()
    
    # L2 regularization 
    lambda_l2 = 0.01
    l2 = 0 
    for p in model.module.densenet121.classifier.parameters():
        l2 = l2 + (p**2).sum()
        
    with torch.no_grad():
        for i, (inp, target) in enumerate (dataLoader):
            target = target.cuda(async=True)
            input_var = Variable(inp)
            vartarget = Variable(target)    
        
            varOutput = model(input_var)
            pred = torch.cat((pred, varOutput.data), 0)
            target = target.cuda()
            gt = torch.cat((gt, target), 0)    
          
            losstensor = loss(varOutput, vartarget) + lambda_l2/2*l2
            losstensorMean += losstensor
            lossVal += losstensor.data
            lossValNorm += 1
        torch.cuda.empty_cache()
            
    outLoss = lossVal / lossValNorm
    losstensorMean = losstensorMean / lossValNorm
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    del gt, pred
    gt_np = gt_np[1: gt_np.shape[0],:]
    pred_np = pred_np[1: pred_np.shape[0],:]
    AUROCs = compute_AUCs(gt_np, pred_np)
    print('The AUROC of {} is {}'.format(CLASS_NAMES[0], AUROCs[0]))
        
    return outLoss.data.cpu().detach().numpy(), losstensorMean, AUROCs[0]


if __name__ == '__main__':
    
    ensemble_index = ['001', '002', '003', '004', '005', '006',\
        '007', '008', '009', '010', '011', '012', '013', '014',\
        '015', '016', '017', '018', '019', '020'] 
    for i, index_ in enumerate(ensemble_index):
        train(BATCH_SIZE = 150, num_iterations = 100, index = index_)
       
