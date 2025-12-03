from decimal import DecimalException
import numpy as np
from sklearn.metrics import confusion_matrix as cm
#from tensorflow.keras.preprocessing import sequence
import torch
from performer_pytorch import Performer
from einops import rearrange, repeat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader,Dataset
import pandas as pd
import os
from itertools import groupby
from tqdm import tqdm
import argparse
import datetime
import ast
import pdb
import math
import random
from scipy.interpolate import interp1d
from random import sample
import multiprocessing
from scipy import stats
import time

def sampler(distr):
    newSampleCount=1000
    distr = np.array(distr)
    newSampleCount = min(newSampleCount,1000)
    distr = distr[distr!=(-1)]
    distSampleCount = len(distr)
    if(distSampleCount<2):
        return np.array([-1])
    interp_func = interp1d(np.arange(distSampleCount), distr)
    scale = distSampleCount/newSampleCount
    ind = 0
    i = np.linspace(0, (distSampleCount-1), num = 1000)
    result = np.round(interp_func(i))
    result=result.astype(int)
    return(list(result))

def applyAvgPooling(reads,WindowSize=None,numberOfSection=None): #chunk_size = # of exons used in calculating each averaged read depth value
    if(numberOfSection==None):
        numberOfSection = len(reads)/WindowSize
    windows = np.array_split(reads,math.ceil(numberOfSection))
    result = np.round(list(map(np.mean,windows))).astype(int)
    return result


def pad_sequences(sequences, maxlen, dtype=np.int32, value=-1):
    padded = np.full((len(sequences), maxlen), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        trunc = seq[-maxlen:]  # keep last maxlen elements if too long
        padded[i, -len(trunc):] = trunc  # left pad: fill from the right
    return padded


parser = argparse.ArgumentParser(description="Generates Parquet File")

required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-l", "--label", help="Path to the groundtruth label (Optional)")
required_args.add_argument("-i", "--input", help="Path to the input directory containing *_W50.txt files.", required=True)
required_args.add_argument("-g", "--gene-lookup", help="Path to the gene index file (.npy).", required=True)
required_args.add_argument("-t", "--target-gene-regions", help="Path to the target gene regions TXT file.", required=True)
required_args.add_argument("-o", "--output-dict", help="Path to the output directory.", required=True)
required_args.add_argument("-c", "--cohort-name", help="Name of the cohort.", required=True)

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

data_path = args.input 

sample_files = [f for f in os.listdir(data_path) if f.endswith('_W50.txt')]
all_samples_names = [item.split("_")[0] for item in sample_files]

#GROUND_TRUTH_PATH = "/mnt/alperyilmaz/ECOLE_SUPERIEURE/DRAGEN_Dataset_Groundtruth/Groundtruth_fullPASS/gene_wise/"

GENE_LOOKUP = np.load(args.gene_lookup,allow_pickle='TRUE').item()

GENE_REGION_INFO = pd.read_csv(args.target_gene_regions,sep="\t")
GENE_REGION_INFO = GENE_REGION_INFO.rename(columns={"gene_name":"GeneName", "seqname": "Chr", "start": "gene_start", "end":"gene_end"})
GENE_REGION_INFO["centerLoc"] = ((GENE_REGION_INFO["gene_start"]+GENE_REGION_INFO["gene_end"])/2).astype(int)
LABELS_INFO = None

result_df = pd.DataFrame(columns=['sample_name','gene_ind',"gene_name",'chr', 'start', 'end', 'read_depth', 'label', ]) 

if(args.label!=None):
    LABELS_INFO = pd.read_csv(args.label,sep="\t")
       
    
def find_samples(directory):
    samples = []
    for filename in os.listdir(directory):
        if filename.endswith("_W50.txt"):
            sample = filename.replace("_W50.txt", "")
            samples.append(sample)
    return samples

pool = multiprocessing.Pool(processes=6)
print(f"creating the train dataset")
ctr=0

TARGET_SAMPLES = find_samples(data_path)

for sample_name in tqdm(TARGET_SAMPLES):

    reads = pd.read_csv(os.path.join(data_path, sample_name + "_W50.txt"), sep="\t")
    #reads["ReadList"]=reads["ReadList"].map(lambda a : ast.literal_eval(a))
    reads["ReadList"] = pool.map(ast.literal_eval,reads["ReadList"])

    #ground_truth = pd.read_csv(GROUND_TRUTH_PATH+"Groundtruth_"+item+".csv", header=None)
    reads["centerLoc"] = ((reads["Start"]+reads["End"])/2).astype(int)

    gtcalls = GENE_REGION_INFO.merge(reads,on=["GeneName","Chr","centerLoc"])

    gtcalls["center_ind"] = (gtcalls["ReadList"].map(len)/2).astype(int) 

    print("processing sample: "+sample_name)

    readdepths_data = []
    wgscalls_data = []
    
    if(args.label!=None):
        gtcalls["label"] = gtcalls.merge(LABELS_INFO[["gene_name", "seqname", "start", "end", sample_name]],left_on=["GeneName","Chr","gene_start","gene_end"], right_on=["gene_name","seqname","start","end"],how="left")[sample_name]

    gtcalls = gtcalls.rename(columns={'GeneName': 'gene_name',
                             'Chr': 'chr',
                             'Start': 'start',
                             'End': 'end'
                              })

    chrom_filter = ((gtcalls['chr']!="chrX") & (gtcalls['chr']!="chrY"))
    gtcalls = gtcalls.loc[chrom_filter].reset_index(drop=True)

    temp_readdepths = gtcalls["ReadList"]

    gtcalls["gene_ind"] = gtcalls.apply(lambda x : GENE_LOOKUP[x["gene_name"]+"-"+x["chr"]],axis=1)

    empty_list_filter = (temp_readdepths.map(len) == 0)

    temp_readdepths = temp_readdepths[~empty_list_filter].reset_index(drop=True)
    gtcalls = gtcalls[~empty_list_filter].reset_index(drop=True)

    longer_gene_len_filter = ((gtcalls["gene_end"]-gtcalls["gene_start"]) > 50*1000) #for those whose reads within gene regions are more than WindowSize * 1000, just insert read depth in gene region
    if(longer_gene_len_filter.any()):
        temp_readdepths.loc[longer_gene_len_filter] = gtcalls.loc[longer_gene_len_filter].apply(lambda x : applyAvgPooling(x["ReadList"][300:(-300)],numberOfSection=1000),axis=1)

    shorter_gene_len_filter = ((gtcalls["gene_end"]-gtcalls["gene_start"]) <= 50*1000)  #for those whose reads within gene regions are less than WindowSize * 1000, insert the context
    if(shorter_gene_len_filter.any()):
        temp_readdepths.loc[shorter_gene_len_filter] = gtcalls.loc[shorter_gene_len_filter].apply(lambda x : x["ReadList"][max((x["center_ind"]-500),0):x["center_ind"]+500],axis=1)
    temp_readdepths = np.asarray([np.asarray(k).astype(int) for k in temp_readdepths],dtype="object")
    temp_readdepths = pad_sequences(temp_readdepths, maxlen=1000,dtype=np.int32,value=-1)
    gtcalls["read_depth"] = list(temp_readdepths)
    gtcalls["sample_name"] = sample_name

    gtcalls = gtcalls.drop(columns=['gene_start','gene_end','centerLoc', 'ReadList', 'center_ind'])
    #result_df = result_df.append(gtcalls, ignore_index=True)
    result_df = pd.concat([result_df, gtcalls], ignore_index=True)
    
if(args.label==None):
    result_df[["sample_name",'gene_ind',"gene_name",'chr', 'start', 'end', 'read_depth' ]].to_parquet(os.path.join(args.output_dict,args.cohort_name+".parquet"),engine='pyarrow')
else:
    result_df[["sample_name",'gene_ind',"gene_name",'chr', 'start', 'end', 'read_depth','label' ]].to_parquet(os.path.join(args.output_dict,args.cohort_name+".parquet"),engine='pyarrow')
pool.close()