import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append("/home/hb/python/")
sys.path.append("/home/hb/python/efficientnet_kincnn/code")
import kincnn_AE as kincnn
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
from utils import PhosphoDataset
from matplotlib import pyplot as plt
import numpy as np
# from utils import CosineAnnealingWarmUpRestarts
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Subset

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
    config.save_dir = f'/home/hb/python/efficientnet_kincnn/saved_model/{datetime.today().strftime("%m%d")}/{config.dataset}_{datetime.today().strftime("%H%M")}_bs{config.batch_size}_weight{config.pretrain_fold_num}'

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # Set the GPU number to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print(f'Using CUDA_VISIBLE_DEVICES {config.gpu_num}')
print('Count of using GPUs:', torch.cuda.device_count())

'''prepare dataset'''
atlas_dataset = PhosphoDataset(pkl_file='/home/hb/python/efficientnet_kincnn/data/pretrain/atlas0595.pkl', root_dir='/home/hb/python/efficientnet_kincnn/data/pretrain/0595instance')
print(len(atlas_dataset))
import pandas as pd
df = atlas_dataset.dataset_frame
y = df['answer']
cv = StratifiedGroupKFold(n_splits=5, shuffle=False)#, random_state=1114)
def train_model_5cv():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=5, shuffle=True)

    wandb.init(project='phospho', entity="jeguring", reinit=True, config=config)
    print(config)
    project_name = f'{datetime.today().strftime("%m%d%H%M")}'
    wandb.run.name = project_name 
    
    # for fold, (train_idx, valid_idx) in enumerate(cv.split(np.arange(len(df)), y=y, groups=df['stratify'])):
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(np.arange(len(df)))):
        train_dataset = Subset(atlas_dataset, train_idx)
        valid_dataset = Subset(atlas_dataset, valid_idx)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
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
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=50, cycle_mult=2, max_lr=0.1, min_lr=0.000001, warmup_steps=20, gamma=0.5)
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("lr: ", optimizer.param_groups[0]['lr'])
        # state_dict = torch.load(f'/home/hb/python/phospho/saved_model/0224/DeepPP_pretrain_1090_1708_bs1024_weight0/{config.pretrain_fold_num}fold_best_model.pth')
        # model.load_state_dict(state_dict['state_dict']) 
        model = model.to(device)
        criterion.to(device)
                
        best_model_weights = model.state_dict()
        best_loss = 1000000.0
        
        #             batch_size=config.batch_size, sampler=test_subsampler)
        
        
        early_stopping = EarlyStopping(patience = config.erls_patience, verbose = True)

        for epoch in tqdm(range(config.n_epoch), position=0, leave=True):
            print('-' * 60)
            print('Epoch {}/{}'.format(epoch+1, config.n_epoch))

            train_loss = 0.0

            for _, batch_data in enumerate(tqdm(train_loader, position=1, leave=True)): 
                inputs = batch_data['matrix']
                y = batch_data['matrix'].to(device)
                # labels = batch_data['label']
                model.train(True)
                inputs = Variable(inputs.to(device, dtype=torch.float), requires_grad=True)
                # labels = Variable(labels.to(device))
                # print(labels)
                decoded = model(inputs)
                loss = criterion(decoded, y).to(device)
                pred = model(inputs) # forward
                # loss = criterion(pred, labels.float().view(-1,1)).to(device)
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

            '''epoch train record'''
            epoch_train_loss = train_loss / len(train_loader)
            # # ---train 1 epoch 끝---

                # ---valid 1 epoch 
            with torch.no_grad():
                model.eval()

                valid_corrects = 0.0         
                valid_loss = 0.0
                valid_precision, valid_recall, valid_f1 = 0.0, 0.0, 0.0

                for i, batch_data in enumerate(tqdm(valid_loader, position=1, leave=True)):
                    # model.train(False)
                    inputs = batch_data['matrix']
                    # labels = batch_data['label']
                    y = batch_data['matrix'].to(device)
                    inputs = Variable(inputs.to(device, dtype=torch.float), requires_grad=True)
                    # labels = Variable(labels.to(device))
                    decoded = model(inputs)
                    loss = criterion(decoded, y).to(device)

                    '''valid record'''
                    valid_loss += loss.item()
                    valid_preds = (pred>=0.5).float()
            
            '''epoch valid record'''
            epoch_valid_loss = valid_loss / len(valid_loader) 
            globals()[f'{fold}_train_loss'].append(epoch_train_loss)
            globals()[f'{fold}_valid_loss'].append(epoch_valid_loss)


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
    wandb.run.save()
    wandb.finish()

    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

train_model_5cv()