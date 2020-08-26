import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch import optim
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
from torch.autograd import Variable
import scipy
from PIL import Image
from Dataset.ChestXrayDataSet import ChestXrayDataSet
from utils.utils import mkdir, removekey, compute_AUCs, acu_curve, delong_roc_variance

def evaluate(BATCH_SIZE = 150, thres = 0.5, vendor = 'base', index = '001'): 
    """
    thres: the high-sensitivity or high-specificity threshold
    vendor: evaluation on a specific vendor. 
    index: the index for ensemble model and its corresponding data list 
    """
    N_CLASSES = 2
    CLASS_NAMES = ['Covid', 'Non-covid']

    CKPT_PATH = './Model_weights/model_train_' + index + '.pth.tar'
    DATA_DIR = './Data/images'
    
    if vendor == 'base':
        TEST_IMAGE_LIST = './Data/labels/labels_covid_test.txt'    
    else:
        TEST_IMAGE_LIST = './Data/labels/labels_covid_test_' + vendor + '.txt'
            
    cudnn.benchmark = True
    

    # initialize and load the model
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        print("=> loading checkpoint")
    else:
        print("=> no checkpoint found")
        
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transResize = 256
    transCrop = 224

    transformList_test = []
    transformList_test.append(transforms.Resize(transCrop)) 
    transformList_test.append(transforms.ToTensor())
    transformList_test.append(normalize)      
    transformSequence_test=transforms.Compose(transformList_test)
    
    val_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform = transformSequence_test)
                                    
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=6, pin_memory=True)
    
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
  
    
    # Define LOSS  
    loss = torch.nn.BCELoss(reduction='mean')

    # Testing mode 
    epochVal(model, val_loader, optimizer, loss, thres, index, vendor)
 

    #-------------------------------------------------------------------------------- 
def epochVal (model, dataLoader, optimizer, loss, thres, index, vendor):
    model.eval()
    lossVal = 0
    lossValNorm = 0    
    losstensorMean = 0
    gt = torch.zeros(1, 2)
    gt = gt.cuda()
    pred = torch.zeros(1, 2)
    pred = pred.cuda()
    with torch.no_grad():
        for i, (inp, target) in enumerate (dataLoader):
            target = target.cuda()
            input_var = Variable(inp)
            vartarget = Variable(target)    
        
            varOutput = model(input_var)
            pred = torch.cat((pred, varOutput.data), 0)
            target = target.cuda()
            gt = torch.cat((gt, target), 0)    
            losstensor = loss(varOutput, vartarget)
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
    
    file = "./Results_" + vendor
    mkdir(file) # build a new folder to save the results             
    np.save('./Results_'  + vendor + '/Prediction_Results_' + index + '.npy', pred_np)
    np.save('./Results_'  + vendor + '/GroundTruth.npy' , gt_np)
    
    COVID_predict_score = pred_np[:,0] # column 0 is COVID prediction score
    scores_covid = COVID_predict_score[gt_np[:,0]==1] # these are the covid+ cases
    scores_noncovid = COVID_predict_score[gt_np[:,0]==0] # these are the covid- cases
    bins = np.linspace(0, 1, 100)
    plt.hist(scores_covid, bins, alpha=0.5, label='Positive cases')
    plt.hist(scores_noncovid, bins, alpha=0.5, label='Negative cases')
    plt.legend(loc='upper right')
    plt.xlabel('Probability score')
    plt.savefig('./Results_' + vendor + '/Probability_score_' + index +'.png', dpi=300, bbox_inches='tight')
    #plot.show()
    plt.close()
    
    AUROCs = compute_AUCs(gt_np, pred_np)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))       
    
    acu_curve(gt_np[:,0], pred_np[:,0], vendor, index)
    
    TP = 0 # True positive
    TN = 0 # True negative 
    FP = 0 # False positive
    FN = 0 # False negative 

    for ij in range(len(gt_np)):
        if gt_np[ij,0] == 1: # covid
            if pred_np[ij, 0] > thres:
                TP = TP + 1
            else:
                FN = FN +1 
                
        else:  # non-covid 
            if pred_np[ij,0] < thres:
                TN = TN+1
            else:
                FP = FP +1 
                
    print('True Positive', TP)
    print('False Negative', FN)
    print('True Negative', TN)
    print('False Positive', FP)
    print('sensitivity', TP/(TP+FN))
    print('specificity', TN/(TN+FP))
    
    alpha = 0.95
    auc_result, auc_cov = delong_roc_variance(gt_np[:,0], pred_np[:,0])
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    
    ci = scipy.stats.norm.ppf(
    lower_upper_q,
    loc=auc_result,
    scale=auc_std)
    ci[ci > 1] = 1
    
    print('AUC:', auc_result)
    print('95% AUC CI:', ci)


def decisionmake(ensemble_index, thres=0.5, vendor='base'):
    # build a new folder to save the ensemble results   
    file = "./Results_" + vendor + "/Results_final"
    mkdir(file) 
    
    # Calculate prediction values / GT 
    gt_np = np.load('./Results_' + vendor + '/GroundTruth.npy')
    pred_np = np.zeros(gt_np.shape)
    pred_np = np.expand_dims(pred_np, axis=2)
   
    for i, index_ in enumerate(ensemble_index):    
        pred_np_path = './Results_'  + vendor + '/Prediction_Results_' + index_ + '.npy'
        if os.path.isfile(pred_np_path):
            pred_np_ = np.load(pred_np_path)
            pred_np_ = np.expand_dims(pred_np_, axis=2)
            pred_np = np.concatenate((pred_np, pred_np_), axis=2)
            
    pred_np = pred_np[:, :, 1: pred_np.shape[2]]
    print('# of models:', pred_np.shape[2])
    np.save('./Results_'  + vendor + '/Results_final/Prediction_Results_' + vendor + '_total.npy', pred_np)

    pred_np = np.sqrt(np.mean(pred_np**2, axis=2))
   
    np.save('./Results_' + vendor + '/Results_final/GroundTruth_' + vendor + '.npy', gt_np)   
    np.save('./Results_' + vendor + '/Results_final/Prediction_Results_' + vendor + '.npy', pred_np)
    
    COVID_predict_score = pred_np[:,0] # column 0 is COVID prediction score
    scores_covid = COVID_predict_score[gt_np[:,0]==1] # these are the covid+ cases
    scores_noncovid = COVID_predict_score[gt_np[:,0]==0] # these are the covid- cases
    bins = np.linspace(0, 1, 100)
    plt.hist(scores_covid, bins, alpha=0.5, label='Positive cases')
    plt.hist(scores_noncovid, bins, alpha=0.5, label='Negative cases')
    plt.legend(loc='upper right')
    plt.xlabel('Probability score')
    plt.savefig('./Results_' + vendor +'/Results_final/Probability_score.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    AUROCs = compute_AUCs(gt_np, pred_np)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))       
    
    acu_curve(gt_np[:,0], pred_np[:,0], vendor)
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for ij in range(len(gt_np)):
        if gt_np[ij,0] == 1: # covid
            if pred_np[ij, 0] > thres:
                TP = TP + 1
            else:
                FN = FN +1 
                
        else:  # non-covid 
            if pred_np[ij,0] < thres:
                TN = TN+1
            else:
                FP = FP +1 
                
    print('True Positive', TP)
    print('False Negative', FN)
    print('True Negative', TN)
    print('False Positive', FP)
    print('sensitivity', TP/(TP+FN))
    print('spercificty', TN/(TN+FP))
    
    alpha = 0.95
    auc_result, auc_cov = delong_roc_variance(gt_np[:,0], pred_np[:,0])
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    
    ci = scipy.stats.norm.ppf(
    lower_upper_q,
    loc=auc_result,
    scale=auc_std)
    ci[ci > 1] = 1
    
    print('AUC:', auc_result)
    print('95% AUC CI:', ci)


if __name__ == '__main__':
    
    ensemble_index = ['001', '002', '003', '004', '005', '006',\
        '007', '008', '009', '010', '011', '012', '013', '014',\
        '015', '016', '017', '018', '019', '020'] 

    for i, index_ in enumerate(ensemble_index): 
        evaluate(BATCH_SIZE = 150, thres = 0.4, vendor='base', index = index_)
      

    decisionmake(ensemble_index = ensemble_index, thres=0.4, vendor='base')   
