import os
import argparse
import pandas as pd
import numpy as np
import subprocess
import shutil
from shutil import which
from Bio import SeqIO


parser = argparse.ArgumentParser(description="""Main script of PhaSUIT.""")
parser.add_argument('--filein', help='the input file', default = 'input.fa')
parser.add_argument('--type', help='The type of the input (protein or dna)',  default = 'protein')
parser.add_argument('--task', help='(binary) or (multi)-class task', default = 'binary')
parser.add_argument('--toolpth', help='pth to the PhaVP foder', default = 'PhaVP/')
parser.add_argument('--root', help='pth to the output foder', default = 'user_0/')
parser.add_argument('--mid', help='pth to the midfolder foder (under output folder)', default = 'midfolder/')
parser.add_argument('--out', help='pth to the store the final prediction (under output folder)', default = 'out/')
parser.add_argument('--threads', help='threads for speed up', type= int, default = 4)
inputs = parser.parse_args()


file_type = inputs.type 
task      = inputs.task
mid_fn    = inputs.mid
out_fn    = inputs.out 
tool_fn   = inputs.toolpth
root_fn   = inputs.root
threads   = inputs.threads
filein    = inputs.filein


if not os.path.isdir(f'{root_fn}/{out_fn}'):
    os.makedirs(f'{root_fn}/{out_fn}')

if not os.path.isdir(f'{root_fn}/{mid_fn}'):
    os.makedirs(f'{root_fn}/{mid_fn}')


# check the type of input
if file_type != 'protein':
    prodigal = "prodigal"
    # check if pprodigal is available
    if which("pprodigal") is not None:
        print("Using parallelized prodigal...")
        prodigal = f'pprodigal -T {threads}'

    prodigal_cmd = f'{prodigal} -i {filein} -a {root_fn}/{mid_fn}/prodigal.fa -f gff -p meta'
    print("Running prodigal...")
    _ = subprocess.check_call(prodigal_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(f'{root_fn}/{mid_fn}/prodigal.fa', 'r') as fin:
        with open(f'{root_fn}/{mid_fn}/test_protein.fa', 'w') as fout:
            raw = fin.read()
            raw = raw.replace('*', '')
            fout.write(raw)
    os.system(f'rm {root_fn}/{mid_fn}/prodigal.fa')
else:
    shutil.copyfile(f'{filein}', f'{root_fn}/{mid_fn}/test_protein.fa')

# generate CGR images
acc_list   = []
seq_list   = []
label_list = []
for record in SeqIO.parse(f'{root_fn}/{mid_fn}/test_protein.fa', 'fasta'):
    acc_list.append(record.id)
    seq_list.append(str(record.seq))
    label_list.append('unknow')

df = pd.DataFrame({'accession': acc_list, 'label': label_list, 'sequence': seq_list})
df.to_csv(f'{root_fn}/{mid_fn}/test_protein.csv', index=False)


cgr_cmd = f'Rscript {tool_fn}/generate_cgr.R {root_fn}/{mid_fn}/test_protein AMINO'
print("Running Rscript to generate CGR images...")
_ = subprocess.check_call(cgr_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Split patches
split_cmd = f'python {tool_fn}/split_patch.py --infile test_protein --res 64 --midfolder {root_fn}/{mid_fn}/ --outfile converted_test_protein'
print("Splitting CGR images into patches...")
_ = subprocess.check_call(split_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Run ViT for prediction
if task == 'multi':
    ViT_cmd = f'python {tool_fn}/predict.py --file converted_test_protein --task {task} --midfolder {root_fn}/{mid_fn}/ --out {root_fn}/{out_fn} --toolpth {tool_fn} --outfile multi_class_prediction.csv'
    print("Running ViT for multi-class prediction...")
    _ = subprocess.check_call(ViT_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    ViT_cmd = f'python {tool_fn}/predict.py --file converted_test_protein --task binary --midfolder {root_fn}/{mid_fn}/ --out {root_fn}/{out_fn} --toolpth {tool_fn} --outfile binary_prediction.csv'
    print("Running ViT for binary prediction...")
    _ = subprocess.check_call(ViT_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ViT_cmd = f'python {tool_fn}/predict.py --file converted_test_protein --task multi --midfolder {root_fn}/{mid_fn}/ --out {root_fn}/{mid_fn} --toolpth {tool_fn} --outfile multi_tmp.csv'
    print("Running ViT for multi-class prediction...")
    _ = subprocess.check_call(ViT_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



    df = pd.read_csv(f'{root_fn}/{out_fn}/binary_prediction.csv')
    recruit_PVP = {acc: 1 for acc, pred in zip(df['accession'].values, df['pred'].values) if pred == 'PVP'}
    PVP_list = []
    non_PVP_list = []
    for record in SeqIO.parse(f'{root_fn}/{mid_fn}/test_protein.fa', 'fasta'):
        try:
            if recruit_PVP[record.id]:
                PVP_list.append(record)
        except:
            non_PVP_list.append(record)
    SeqIO.write(PVP_list, f'{root_fn}/{out_fn}/pvp.fa', 'fasta')
    SeqIO.write(non_PVP_list, f'{root_fn}/{out_fn}/non_pvp.fa', 'fasta')

    df = pd.read_csv(f'{root_fn}/{mid_fn}/multi_tmp.csv')
    df_list = []
    for acc in recruit_PVP:
        df_list.append(df[df['accession'] == acc])
    try:
        df = pd.concat(df_list)
        df.to_csv(f'{root_fn}/{out_fn}/multi_class_prediction.csv', index=False)
    except:
        pass























