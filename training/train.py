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
from sklearn.utils.class_weight import compute_class_weight




parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--trainfile', help='patches for training',  default = 'converted_train')
parser.add_argument('--valfile', help='patches for validation',  default = 'converted_val')
parser.add_argument('--trainlabel', help='patches for training',  default = 'converted_train_label')
parser.add_argument('--vallabel', help='patches for validation',  default = 'converted_val_label')
parser.add_argument('--nepoch', help='number of epoch for training', type=int)
parser.add_argument('--midfolder', help='pth to the midfolder foder', default = './')
parser.add_argument('--out', help='pth to the output foder', default = 'out/')
parser.add_argument('--task', help='trianing model on binary task or multi-class task', default = 'binary')
inputs = parser.parse_args()


mid_fn = inputs.midfolder
out_fn = inputs.out
task   = inputs.task
nepoch = inputs.nepoch
converted_train = inputs.trainfile
converted_val   = inputs.valfile
converted_train_label = inputs.trainlabel
converted_val_label   = inputs.vallabel



converted_train = pkl.load(open(f"{mid_fn}/{converted_train}", 'rb'))
converted_test  = pkl.load(open(f"{mid_fn}/{converted_val}", 'rb'))

train_labels = pkl.load(open(f"{mid_fn}/{converted_train_label}", 'rb'))
test_labels  = pkl.load(open(f"{mid_fn}/{converted_val_label}", 'rb'))



num_of_class = len(set(test_labels))
if task == 'binary':
    label2int = {'PVP':1, 'non-PVP': 0}
    train_labels = np.array([label2int[item] for item in train_labels])
    test_labels  = np.array([label2int[item] for item in test_labels])
    df = pd.DataFrame({'strlabel':label2int.keys(), 'intlabel': label2int.values})
    df.to_csv(f'{out_fn}/label2int.csv', index=False)
else:
    label2int = {item:idx for idx, item in enumerate(set(test_labels))}
    train_labels = np.array([label2int[item] for item in train_labels])
    test_labels  = np.array([label2int[item] for item in test_labels])
    df = pd.DataFrame({'strlabel':label2int.keys(), 'intlabel': label2int.values})
    df.to_csv(f'{out_fn}/label2int.csv', index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def reset_model():
    model = Transformer(
                src_vocab_size = converted_train.shape[2],
                src_pad_idx = 0,
                device=device,
                max_length=converted_train.shape[1],
                dropout=0.2,
                out_dim=num_of_class
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    return model, optimizer, loss_func




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

if torch.cuda.device_count() > 1:
    print(f'Use {torch.cuda.device_count()} GPUs!\n')
    model = nn.DataParallel(model)

model.to(device)





training_loader = return_batch(converted_train, train_labels, flag = True, drop=True)
test_loader = return_batch(converted_test, test_labels, flag = False, drop=False)
max_f1 = 0
for epoch in range(nepoch):
    _ = model.train()
    for step, (batch_x, batch_y) in enumerate(training_loader): 
        prediction = model(batch_x.to(device))
        loss = loss_func(prediction.squeeze(1), batch_y.to(device))
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    _ = model.eval()
    with torch.no_grad():
        all_pred = []
        for step, (batch_x, batch_y) in enumerate(test_loader): 
            logit = model(batch_x.to(device))
            pred  = np.argmax(logit.squeeze(1).cpu().detach().numpy(), axis=1).tolist()
            all_pred += pred
        f1 = accuracy_score(test_labels, all_pred)
        if max_f1 < f1:
            max_f1 = f1 
            if task == 'binary':
                torch.save(model.state_dict(), f'{out_fn}/transformer_binary.pth')
            else:
                torch.save(model.state_dict(), f'{out_fn}/transformer_multi.pth')
            print("testing:")
            print(classification_report(test_labels, all_pred))
            all_pred = []
            all_label = []
            for step, (batch_x, batch_y) in enumerate(training_loader):
                logit = model(batch_x.to(device))
                pred  = np.argmax(logit.squeeze(1).cpu().detach().numpy(), axis=1).tolist()
                all_pred += pred
                all_label.append(batch_y.cpu().detach().numpy())
            all_label = np.array(all_label).reshape(-1)
            print("training:")
            print(classification_report(all_label, all_pred))
           