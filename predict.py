import os
import argparse
import pandas as pd
import numpy as np
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  torch.utils.data as Data
from    sklearn.model_selection import KFold
import  pickle as pkl
from model import Transformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--file', help='input patches',  default = 'converted_protein')
parser.add_argument('--task', help='binary task or multi-class task', default = 'binary')
parser.add_argument('--midfolder', help='pth to the midfolder foder', default = 'midfolder/')
parser.add_argument('--out', help='pth to the output foder', default = 'out/')
parser.add_argument('--toolpth', help='pth to the PhaVP foder', default = 'PhaVP/')
parser.add_argument('--outfile', help='name of the output file', default = 'final_prediction.csv')
inputs = parser.parse_args()


file_fn   = inputs.file 
mid_fn    = inputs.midfolder
task      = inputs.task
out_fn    = inputs.out
tool_fn   = inputs.toolpth
outfile   = inputs.outfile


converted_test = pkl.load(open(f"{mid_fn}/{file_fn}", 'rb'))
test_labels = np.array(np.zeros(converted_test.shape[0]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if task == 'binary':
    out_dim = 2
else:
    out_dim = 8

def reset_model():
    model = Transformer(
                src_vocab_size = converted_test.shape[2],
                src_pad_idx = 0,
                device=device,
                max_length=converted_test.shape[1],
                dropout=0.1,
                out_dim = out_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    return model, optimizer, loss_func

def return_softmax(all_score):
    result = []
    for item in all_score:
       result.append(softmax(item))
    return np.array(result)


def return_batch(train_sentence, label, flag, drop):
    X_train = torch.from_numpy(train_sentence).float()
    y_train = torch.from_numpy(label).long()
    train_dataset = Data.TensorDataset(X_train, y_train)
    training_loader = Data.DataLoader(
        dataset=train_dataset,    
        batch_size=256,
        shuffle=flag,               
        num_workers=0,
        drop_last=drop              
    )
    return training_loader

def return_tensor(var, device):
    return torch.from_numpy(var).to(device)



model, optimizer, loss_func = reset_model()
try:
    if task == 'binary':
        pretrained_dict=torch.load(f'{tool_fn}/model/transformer_binary.pth', map_location=device)
    else:
        pretrained_dict=torch.load(f'{tool_fn}/model/transformer_multi.pth', map_location=device)
    model.load_state_dict(pretrained_dict)
except:
    print('cannot find pre-trained model')
    exit(1)


int2label = {0:'minor_capsid',1:'tail_fiber', 2:'major_tail', 3:'portal', 4:'minor_tail', 5:'baseplate', 6:'major_capsid', 7:'other'}


if task == 'binary':
    test_loader = return_batch(converted_test, test_labels, flag = False, drop=False)
    model = model.eval()
    with torch.no_grad():
        all_pred = []
        all_score = []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            logit = model(batch_x.to(device))
            pred  = np.argmax(logit.squeeze(1).cpu().detach().numpy(), axis=1).tolist()
            all_pred += pred
            pred  = logit.squeeze(1).cpu().detach().numpy()
            all_score.append(pred)
        all_score = np.concatenate(all_score)
        all_score = return_softmax(all_score)
        name = file_fn.split('converted_')[1]
        df = pd.read_csv(f'{mid_fn}/{name}.txt')
        all_pred = ['PVP' if item > 0.5 else 'non-PVP' for item in all_pred]
        pred_df = pd.DataFrame({"accession":df['accession'].values, "pred":all_pred, "score":all_score[:, 1]})    
        pred_df.to_csv(f'{out_fn}/{outfile}', index=False)
else:
    test_loader = return_batch(converted_test, test_labels, flag = False, drop=False)
    model = model.eval()
    with torch.no_grad():
        all_pred = []
        all_score = []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            logit = model(batch_x.to(device))
            pred  = np.argmax(logit.squeeze(1).cpu().detach().numpy(), axis=1).tolist()
            all_pred += pred
            pred  = logit.squeeze(1).cpu().detach().numpy()
            all_score.append(pred)
        all_score = np.concatenate(all_score)
        all_score = return_softmax(all_score)
        name = file_fn.split('converted_')[1]
        df = pd.read_csv(f'{mid_fn}/{name}.txt')
        all_pred = [int2label[item] for item in np.argmax(all_score, axis=1)]
        pred_df = pd.DataFrame({"accession":df['accession'].values, "pred":all_pred, "score":np.max(all_score, axis=1)})    
        pred_df.to_csv(f'{out_fn}/{outfile}', index=False)
