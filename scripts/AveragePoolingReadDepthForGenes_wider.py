import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import math
import sys
import argparse
import os

parser = argparse.ArgumentParser(description='CHALLENGER Preprocessor')

parser.add_argument("-s",'--sample-name', type=str, help='Sample name (string identifier)', required=True)
parser.add_argument("-t",'--target-gene-regions', type=str, help='Path to the target gene regions TXT file', required=True)
parser.add_argument("-r",'--read-depth-path', default='./', type=str, help='Path to the read-depth directory', required=True)
parser.add_argument("-o",'--output-dict', default='./', type=str, help="Path to the output directory.")

args = parser.parse_args() 

WINDOW_SIZE=50
TARGET_FILE=args.target_gene_regions
SAMPLE_READ_PATH = args.read_depth_path
SAMPLE_NAME = args.sample_name


def applyAvgPooling(i,gene_reads,reads):
    windows = np.array_split(reads,math.ceil(len(reads)/WINDOW_SIZE))
    result = np.round(list(map(np.mean,windows))).astype(int)
    gene_reads.iloc[i]["ReadList"].extend(result)

def main():
    gene_reads = pd.read_csv(TARGET_FILE,sep="\t")
    gene_reads[4] = [list() for x in range(len(gene_reads.index))]
    gene_reads.columns =['GeneName','Chr', 'Start', 'End', 'ReadList']    
    i=0
    ####### BEGIN NOTE : Remove the following code if you have "chr" suffix in your read depth
    #gene_reads["Chr"] = gene_reads["Chr"].apply(lambda x : x.split("chr")[1])
    ###### END NOTE
    GENE_START = gene_reads.iloc[i]["Start"]
    GENE_END = gene_reads.iloc[i]["End"]
    GENE_CHR = gene_reads.iloc[i]["Chr"]

    reads = np.zeros(GENE_END-GENE_START+1)
    for k,chunk in tqdm(enumerate(pd.read_csv(os.path.join(SAMPLE_READ_PATH, SAMPLE_NAME + ".txt"),sep="\t" ,chunksize=1000000,engine="c",header=None))):
        while(True):
            cnd = ((chunk[0].astype(str) ==GENE_CHR) & (chunk[1]>=GENE_START) & ((chunk[1]<=GENE_END)))
            if(cnd.any()): 
                reads[chunk[cnd][1]-GENE_START] = chunk[cnd][2]
            if(cnd.iloc[-1] == True): # meaning we processed last element of the chunk, do not change the query gene
                break            
            else:  #pass to the next gene
                applyAvgPooling(i,gene_reads,reads)
                i+=1
                GENE_START = gene_reads.iloc[i]["Start"]
                GENE_END = gene_reads.iloc[i]["End"]
                GENE_CHR = gene_reads.iloc[i]["Chr"]
                reads = np.zeros(GENE_END-GENE_START+1)  

    applyAvgPooling(i,gene_reads,reads) # for last chunk
    ####### BEGIN NOTE : Remove the following code if you have "chr" suffix in your read depth
    #gene_reads["Chr"] = gene_reads["Chr"].apply(lambda a : "chr"+a)
    ###### END NOTE    
    gene_reads.to_csv(os.path.join(args.output_dict, f"{SAMPLE_NAME}_W{WINDOW_SIZE}.txt"),index=False,sep="\t")

if __name__=="__main__": 
    print("[{0}] Calculating average read depths (window size = {1}) ".format(SAMPLE_NAME,WINDOW_SIZE))  
    main()
    print("[{0}] Completed ".format(SAMPLE_NAME))  


