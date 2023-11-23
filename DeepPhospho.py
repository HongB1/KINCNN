import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append("/home/hb/python/")
sys.path.append("/home/hb/python/efficientnet_kincnn/code")
import kincnn
from phospho_preprocessing import prepare_dataset, AttrDict
import random
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch import nn
from torch.autograd import Variable
from Radam import RAdam
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
from precision_recall import precision_recall
from EarlyStopping import EarlyStopping
import wandb
from matplotlib import pyplot as plt
import numpy as np
# from utils import CosineAnnealingWarmUpRestarts
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

if __name__ == "__main__":
    config = AttrDict()
    config.gpu_num = sys.argv[1]
    config.batch_size = int(sys.argv[2])
    config.n_epoch = int(sys.argv[3])
    config.defalut_learning_rate = float(sys.argv[4])
    config.fold_num = int(sys.argv[5])
    config.scheduler_patience, config.scheduler_factor = int(sys.argv[6]), float(sys.argv[7])
    config.erls_patience = int(sys.argv[8])
    config.dataset = sys.argv[9]
    config.pretrain_fold_num = sys.argv[10]
    config.model = f'KINCNN'
    config.project = sys.argv[11]
    config.save_dir = f'/home/hb/python/efficientnet_kincnn/saved_model/{datetime.today().strftime("%m%d")}/{config.project}/{datetime.today().strftime("%H%M")}_weight{config.pretrain_fold_num}'
    # os.makedirs(f'{}')
    os.makedirs(f'{config.save_dir}', exist_ok=True)

    import yaml
    with open(f'{config.save_dir}/config.yaml', 'w') as f:
        yaml.dump(config, f)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

seed_everything(42)        
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_num  # Set the GPU number to use
device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print(f'Using CUDA_VISIBLE_DEVICES {config.gpu_num}')
print('Count of using GPUs:', torch.cuda.device_count())

'''prepare dataset'''
dtset = prepare_dataset(dataset_mode=config.dataset)
train_set = dtset[0]
valid_set = dtset[1]
print(train_set, valid_set)

# train_set = data_utils.TensorDataset(torch.tensor(xTrain), torch.tensor(yTrain))
# valid_set = data_utils.TensorDataset(torch.tensor(xTest), torch.tensor(yTest))
train_loader = data_utils.DataLoader(train_set, batch_size=config.batch_size, pin_memory=True, shuffle=True)
valid_loader = data_utils.DataLoader(valid_set, batch_size=config.batch_size, )

dataloaders = {'train':train_loader,'valid':valid_loader}
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'valid']}
dataset = ConcatDataset([train_set, valid_set])

def train_model_5cv():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=5, shuffle=True)

    wandb.init(project='phospho', entity="jeguring", reinit=True, config=config)
    print(config)
    project_name = f'{datetime.today().strftime("%m%d%H%M")}_{config.project}'
    wandb.run.name = project_name 
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):

        globals()[f'{fold}_train_loss'] = []
        globals()[f'{fold}_train_precision'] = []
        globals()[f'{fold}_train_recall'] = []
        globals()[f'{fold}_train_f1'] = []
        globals()[f'{fold}_train_acc'] = []

        globals()[f'{fold}_valid_loss'] = []
        globals()[f'{fold}_valid_precision'] = []
        globals()[f'{fold}_valid_recall'] = []
        globals()[f'{fold}_valid_f1'] = []
        globals()[f'{fold}_valid_acc'] = []
        globals()[f'{fold}_lr'] = []

        globals()[f'{fold}_result'] = []
        print(f'FOLD {fold}')
        print('--------------------------------')

        '''model compile'''
        model = kincnn.EfficientNet.from_name(f'{config.model}')

        '''optimizer & loss'''

        optimizer = RAdam(model.parameters(), lr=0)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience, threshold=0.0001, cooldown=0, min_lr=0, verbose=1)
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=50, cycle_mult=2, max_lr=0.1, min_lr=0.000001, warmup_steps=20, gamma=0.5)
        # if 'softlabel' in config.dataset :
            # criterion = nn.MSELoss()
        # else:
        criterion = nn.BCELoss()


        # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("lr: ", optimizer.param_groups[0]['lr'])
        
        state_dict = torch.load(f'/home/hb/python/efficientnet_kincnn/saved_model/AE_pretrain_4_1143_weight0/{config.pretrain_fold_num}fold_best_model.pth')
        print(state_dict['state_dict'])
        # model.load_state_dict(state_dict['state_dict']) 
        encoder_state_dict = {_layer_name: _weight for _layer_name, _weight in state_dict['state_dict'].items() if 'decoder' not in _layer_name}
        model.load_state_dict(encoder_state_dict) 
        model = model.to(device)
        criterion.to(device)
                
        best_model_weights = model.state_dict()
        best_loss = 1000000.0
        
        # Define data loaders for training and testing data in this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=config.batch_size, sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.batch_size, sampler=test_subsampler)
        
        
        early_stopping = EarlyStopping(patience = config.erls_patience, verbose = True)

        for epoch in tqdm(range(config.n_epoch), position=0, leave=True):
            print('-' * 60)
            print('Epoch {}/{}'.format(epoch+1, config.n_epoch))

            train_corrects = 0.0      
            train_loss = 0.0
            train_precision, train_recall, train_f1 = 0.0, 0.0, 0.0
            # rus = RandomUnderSampler(random_state=epoch)
            # undersampling_idx, _ = rus.fit_resample(train_idx, y_train)
            # undersampling_idx = undersampling_idx.flatten()
            
            # X_urs = X_train[undersampling_idx]
            # y_urs = y_train[undersampling_idx]
            # print(y_urs.bincount())
            # train_dataset = data_utils.TensorDataset(torch.tensor(X_urs), torch.tensor(y_urs))
            
            # trainloader = torch.utils.data.DataLoader(
            #                 train_dataset,
            #                 batch_size=config.batch_size,)
            # validloader = torch.utils.data.DataLoader(
            #                 valid_dataset,
            #                 batch_size=config.batch_size,)
            for _, (inputs, labels) in enumerate(tqdm(trainloader, position=1, leave=True)): 
                model.train(True)
                # inputs, labels = data
                inputs = Variable(inputs.to(device, dtype=torch.float), requires_grad=True)
                # print(labels.shape)
                # print(labels)
                # labels = labels.unsqueeze(1).to(device, dtype=torch.float)
                # print(labels)
                # print(labels.shape)
                labels = Variable(labels.to(device))
                # print(labels)

                pred = model(inputs) # forward
                # print('pred:', pred)
                # print('labels:', labels)
                # precision, recall, f1 = precision_recall(labels.float().view(-1,1), pred) # outputs = net(inputs)
                loss = criterion(pred, labels.float().view(-1,1)).to(device)
                # loss = criterion(pred, labels).to(device)
                # print(labels)
                preds = (pred>0.5).float()

                '''backward'''
                optimizer.zero_grad() # zero the parameter gradients
                loss.backward()
                optimizer.step()

                '''train record'''
                train_loss += loss.item()
                train_preds = (pred>=0.5).float()
                # train_corrects += accuracy_score(labels.cpu(), train_preds.cpu())
                # train_precision += precision.detach()
                # train_recall += recall.detach()
                # train_f1 += f1.detach()

            '''epoch train record'''
            epoch_train_loss = train_loss / len(trainloader)
            # epoch_train_precision = train_precision / len(trainloader)
            # epoch_train_recall = train_recall / len(trainloader)
            # epoch_train_f1 = train_f1 / len(trainloader)
            # epoch_train_acc = train_corrects / len(trainloader)
            # # ---train 1 epoch 끝---

                # ---valid 1 epoch 
            with torch.no_grad():
                model.eval()

                valid_corrects = 0.0         
                valid_loss = 0.0
                valid_precision, valid_recall, valid_f1 = 0.0, 0.0, 0.0

                for i, (inputs, labels) in enumerate(tqdm(validloader, position=1, leave=True)):
                    # model.train(False)
                    inputs = Variable(inputs.to(device, dtype=torch.float), requires_grad=True)
                    labels = Variable(labels.to(device))

                    pred = model(inputs) 
                    
                    # precision, recall, f1 = precision_recall(labels.float().view(-1,1), pred) # outputs = net(inputs)
                    loss = criterion(pred, labels.float().view(-1,1)).to(device)
                    # loss = criterion(pred, labels.float().view(-1,1)).to(device)   

                    '''valid record'''
                    valid_loss += loss.item()
                    valid_preds = (pred>=0.5).float()
                    # valid_corrects += accuracy_score(labels.cpu(), valid_preds.cpu())
                    # valid_precision += precision.detach()
                    # valid_recall += recall.detach()
                    # valid_f1 += f1.detach()
            
            '''epoch valid record'''
            epoch_valid_loss = valid_loss / len(validloader) 
            # epoch_valid_precision = valid_precision / len(validloader)
            # epoch_valid_recall = valid_recall / len(validloader)
            # epoch_valid_f1 = valid_f1 / len(validloader)
            # epoch_valid_acc = valid_corrects / len(validloader)

            globals()[f'{fold}_train_loss'].append(epoch_train_loss)
            # globals()[f'{fold}_train_precision'].append(epoch_train_precision)
            # globals()[f'{fold}_train_recall'].append(epoch_train_recall)
            # globals()[f'{fold}_train_f1'].append(epoch_train_f1)
            # globals()[f'{fold}_train_acc'].append(epoch_train_acc) 

            globals()[f'{fold}_valid_loss'].append(epoch_valid_loss)
            # globals()[f'{fold}_valid_precision'].append(epoch_valid_precision)
            # globals()[f'{fold}_valid_recall'].append(epoch_valid_recall)
            # globals()[f'{fold}_valid_f1'].append(epoch_valid_f1)
            # globals()[f'{fold}_valid_acc'].append(epoch_valid_acc) 

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_model_weights = model.state_dict()
            # valiid 1 epoch end
            # 가장 최근 모델 저장
            checkpoint = {'epoch':epoch, 
            'loss':epoch_valid_loss,
                'model': model,
                        #'state_dict': model.module.state_dict(),
                            'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, f"{config.save_dir}/{fold}fold_latest_epoch.pth")

            # Earlystopping & best 모델 저장
            savePath = "{}/{}fold_best_model.pth".format(wandb.config.save_dir, fold) 
            early_stopping(epoch_valid_loss, model, optimizer, savePath)
            if early_stopping.early_stop:
                print(f'Early stopping... fold:{fold} epoch:{epoch} loss:{epoch_valid_loss}')
                break
            # wandb.log({f'{config.data} : {fold}fold Validation loss': epoch_valid_loss, f'{fold}fold Learning_rate':optimizer.param_groups[0]['lr']})

            wandb.log({f"{fold} fold train" : {"loss":epoch_train_loss}, f"{fold} fold val":{"loss":epoch_valid_loss} ,f"{fold} fold learning_rate":optimizer.param_groups[0]['lr']})
            globals()[f'{fold}_lr'].append(optimizer.param_groups[0]['lr'])
            scheduler.step(epoch_valid_loss) # reduced는 무조건 epoch에서 backward
            print("lr: ", optimizer.param_groups[0]['lr'])
            print('-' * 60)
            print()
            # globals()[f'{fold}_result'].append(epoch_valid_loss)

        torch.cuda.empty_cache()

    plt.plot(globals()['0_valid_loss'], label="0fold")
    plt.plot(globals()['1_valid_loss'], label='1fold')
    plt.plot(globals()['2_valid_loss'], label='2fold')
    plt.plot(globals()['3_valid_loss'], label='3fold')
    plt.plot(globals()['4_valid_loss'], label='4fold')
    plt.title('Validation loss')
    plt.xlabel("epoch")
    plt.ylabel("Validation loss")
    plt.legend()
    plt.show()
    plt.savefig(config.save_dir + "/fig_saved.png")
    # wandb.log({f'{config.data}': plt})
    wandb.run.save()
    wandb.finish()

    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

train_model_5cv()