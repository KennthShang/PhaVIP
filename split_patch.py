import argparse
import os
import pandas as pd
import numpy as np
from    sklearn.model_selection import KFold
import  pickle as pkl
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--infile', help='FASTA file of contigs',  default = 'protein')
parser.add_argument('--res', help='resolution', type= int,  default = 64)
parser.add_argument('--midfolder', help='pth of midfolder',  default = 'midfolder')
parser.add_argument('--outfile', help='name of output',  default = 'converted_protein')
inputs = parser.parse_args()

infile   = inputs.infile
res      = inputs.res
outfile  = inputs.outfile
midfolder = inputs.midfolder


train_data_df = pd.read_csv(f"{midfolder}/{infile}.txt")
train_labels  = train_data_df['label'].values
train_feature = train_data_df['figure'].values
train_feature = np.array([item.split(" ") for item in train_feature]).astype(float)


def return_img(feature):
    converted_figure = []
    for item in feature:
        fig = item.reshape(res, res)
        patch_figure = []
        for i in range(0, res, 16):
            for j in range(0, res, 16):
                patch = fig[i:i+16, j:j+16]
                patch = patch.reshape(-1)
                assert len(patch) == 256
                patch_figure.append(patch)
        patch_figure = np.array(patch_figure)
        converted_figure.append(patch_figure)

    converted_figure = np.array(converted_figure)
    return converted_figure


converted_train = return_img(train_feature)

def normalize(feature):
    norm_feature = []
    for item in feature:
        max_ = np.max(item)
        min_ = np.min(item)
        norm_feature.append((item-min_)/(max_-min_))
    return np.array(norm_feature) 


converted_train = normalize(converted_train)
pkl.dump(converted_train, open(f"{midfolder}/{outfile}", "wb"))
pkl.dump(train_labels, open(f'{midfolder}/label_{outfile}', 'wb'))
