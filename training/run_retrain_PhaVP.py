import os
import argparse
import pandas as pd
import numpy as np
import subprocess
import shutil
from shutil import which
from Bio import SeqIO


parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--trainin', help='the input file', default = 'train.csv')
parser.add_argument('--valin', help='the input file', default = 'val.csv')
parser.add_argument('--task', help='(binary) or (multi)-class task', default = 'binary')
parser.add_argument('--mid', help='pth to the midfolder foder (under output folder)', default = 'midfolder/')
parser.add_argument('--out', help='pth to the store the final prediction (under output folder)', default = 'out/')
parser.add_argument('--nepoch', help='number of epoch for training', type=int, default= 150)
inputs = parser.parse_args()



task      = inputs.task
mid_fn    = inputs.mid
out_fn    = inputs.out
trainin   = inputs.trainin
valin     = inputs.valin
nepoch    = inputs.nepoch

if not os.path.isdir(out_fn):
    os.makedirs(out_fn)



# generate CGR images
trainin = trainin.split('.')[0]
#cgr_cmd = f'Rscript ../generate_cgr.R {trainin} AMINO'
#print("Running Rscript to generate CGR images...")
#_ = subprocess.check_call(cgr_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

valin = valin.split('.')[0]
#cgr_cmd = f'Rscript ../generate_cgr.R {valin} AMINO'
#print("Running Rscript to generate CGR images...")
#_ = subprocess.check_call(cgr_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Split patches
#split_cmd = f'python ../split_patch.py --infile {trainin} --res 64 --midfolder ./ --outfile converted_{trainin}'
#print("Splitting CGR images into patches...")
#_ = subprocess.check_call(split_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#split_cmd = f'python ../split_patch.py --infile {valin} --res 64 --midfolder ./ --outfile converted_{valin}'
#print("Splitting CGR images into patches...")
#_ = subprocess.check_call(split_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



# retrain ViT
ViT_cmd = f'python train.py --task {task} --midfolder ./ --out {out_fn} --nepoch {nepoch} --trainfile converted_{trainin} --valfile converted_{valin} --trainlabel label_converted_{trainin} --vallabel label_converted_{valin}'
print(f"training ViT for {task} prediction...")
_ = subprocess.check_call(ViT_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


