# -*- coding: utf-8 -*-


"""# Setting Arguments"""

import os,sys
import subprocess
import threading
import argparse
from tqdm import tqdm
from pathlib import Path
import requests
import pdb
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast,RobertaTokenizerFast
import json
from torchvision.datasets import DatasetFolder
from datasets import load_dataset,DatasetDict,load_from_disk
from transformers import TrainerCallback,Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import evaluate
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers,AddedToken
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__),"..","CHALLENGER_model")))
from data_collator import DataCollatorForLanguageModeling

from challenger_roberta_configurations import CHALLENGER_roberta_config
from challenger_modeling_roberta import *

parser = argparse.ArgumentParser(description='CHALLENGER - RoBERTa')

# Fine-Tuning Configuration
parser.add_argument("-g", "--gpu", help="GPU device(s) to use (e.g., '0' or '0,1').")
parser.add_argument("-bs", "--batch-size", default=32, type=int, help="Mini-batch size for fine-tuning.")
parser.add_argument("-lr", "--lr-scheduler-type", default="linear", type=str, help="Learning rate scheduler type (e.g., 'linear', 'cosine').")
parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input Parquet file.")
parser.add_argument("-ep", "--num-epoch", default=10, type=int, help="Number of training epochs.")

# Model Configurations
parser.add_argument("-n", "--normalize", type=str, help="Path to mean/std normalization file (.txt).")
parser.add_argument("-b", "--baseline-coverages-path", default="", type=str, help="Path to baseline gene coverage dictionary (.pt).")
parser.add_argument("-t", "--tokenizer-path", type=str, required=True, help="Path to the tokenizer configuration file (.json).")

# Utils
parser.add_argument("-w", "--init-weight", default="outputs", type=str, metavar="PATH", help="Path to initial model weights.")
parser.add_argument("-o", "--output-dir", default="outputs", type=str, metavar="PATH", help="Path where the trained model weights will be saved.")
parser.add_argument("-r", "--run-name", type=str, required=True, help="Name of this run for logging and organization.")


args = parser.parse_args()  # running in command line

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


args.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

os.makedirs(args.output_dir,exist_ok=True)

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def get_mean_std(file_name):
    file1 = open(file_name, 'r')
    line = file1.readline()
    mean_cov_val = float(line.split(",")[0])
    std_cov_val = float(line.split(",")[1])
    return (mean_cov_val,std_cov_val)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CLS_tp_tn_fp_fn = 0
        self.CLS_tp_nocall = 0
        self.CLS_tp_duplication = 0
        self.CLS_tp_plus_fp_nocall = 0
        self.CLS_tp_plus_fp_duplication = 0 
        self.CLS_tp_plus_fp_deletion = 0
        self.CLS_tp_plus_fn_nocall = 0
        self.CLS_tp_plus_fn_duplication = 0
        self.CLS_tp_plus_fn_deletion = 0
        self.CLS_tp_deletion = 0

        self.latest_cls_loss = torch.tensor(0).to("cuda")
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=-1):
        labels = inputs.pop("labels")
        # forward pass
        model_output = model(**inputs)
        seg_logits = model_output["seg_logits"]
        cls_logits = model_output["cls_logits"]

        _, predicted_seg = torch.max(seg_logits, 2)
        _, predicted_cls = torch.max(cls_logits, 1)
        
        #seg_logits = outputs.get("seg_logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0])).to(args.device)
        positive_flag = ((labels)>=0) & (inputs["input_ids"]>=0)
        labels[:,1:-1] = tokenizer.encode("[NOCALL]")[0]-labels[:,1:-1]
        masked_labels = labels[positive_flag].to(args.device)
        masked_predicted_seg = predicted_seg[positive_flag].to(args.device)

        sample_labels = torch.zeros(cls_logits.shape[0]).to(int)
        dup_samples = (labels==1).any(axis=1)
        del_samples = (labels==2).any(axis=1)
        sample_labels[dup_samples] = 1
        sample_labels[del_samples] = 2
        sample_labels = sample_labels.to(args.device)

        ####### Classification Loss ###################
        cls_ce_loss_val = loss_fct(cls_logits.view(-1, 3), sample_labels.view(-1))

        ####### Total Loss ##############
        loss = (2*cls_ce_loss_val)

        ####### Compute CLS Scores ########
        self.CLS_tp_nocall += (torch.logical_and(predicted_cls == sample_labels,predicted_cls == 0)).sum().item() 
        self.CLS_tp_duplication += (torch.logical_and(predicted_cls == sample_labels,predicted_cls == 1)).sum().item() 
        self.CLS_tp_deletion += (torch.logical_and(predicted_cls == sample_labels,predicted_cls == 2)).sum().item()

        self.CLS_tp_plus_fp_nocall += (predicted_cls == 0).sum().item()
        self.CLS_tp_plus_fp_duplication += (predicted_cls == 1).sum().item()
        self.CLS_tp_plus_fp_deletion += (predicted_cls == 2).sum().item()

        self.CLS_tp_plus_fn_nocall += (sample_labels == 0).sum().item()
        self.CLS_tp_plus_fn_duplication += (sample_labels == 1).sum().item()
        self.CLS_tp_plus_fn_deletion += (sample_labels == 2).sum().item()
        self.CLS_tp_tn_fp_fn += sample_labels.size(0)
        
        self.latest_cls_loss = cls_ce_loss_val    

        return(loss, {"label": (seg_logits,cls_logits)}) if return_outputs else loss
        #return (loss, seg_logits) if return_outputs else loss

def calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion):
    nocall_prec = tp_nocall / (tp_plus_fp_nocall+1e-15) 
    dup_prec = tp_duplication / (tp_plus_fp_duplication+1e-15) 
    del_prec = tp_deletion / (tp_plus_fp_deletion+1e-15) 
    
    nocall_recall = tp_nocall / (tp_plus_fn_nocall+1e-15) 
    dup_recall = tp_duplication / (tp_plus_fn_duplication+1e-15) 
    del_recall = tp_deletion / (tp_plus_fn_deletion+1e-15) 
    
    nocall_f1 = (2*nocall_prec*nocall_recall) / (nocall_prec+nocall_recall+1e-15)
    dup_f1 = (2*dup_prec*dup_recall) / (dup_prec+dup_recall+1e-15)
    del_f1 = (2*del_prec*del_recall) / (del_prec+del_recall+1e-15)
    
    overall_prec = (dup_prec+del_prec)/2
    overall_recall = (dup_recall+del_recall)/2
    overall_f1 = (dup_f1+del_f1)/2

    return nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall, nocall_f1, dup_f1, del_f1, overall_prec, overall_recall, overall_f1

            

"""## Tokenization"""
def tokenization(create_vocab):
    if(create_vocab):
        #vocab_words = [f"[{str(i)}]" for i in range(250)]
        vocab_words= []
        special_token_list=[]
        
        # Initialize a Byte-Level BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Prepare the signal data for training
        
        # Train the tokenizer on the signal data
        trainer = trainers.BpeTrainer(vocab_size=12, min_frequency=10, special_tokens=["[BOS]", "[SEP]", "[PAD]", "[MASK]","[UNK]","[CLS]","[DEL]","[DUP]","[NOCALL]","[CNV_MASK]"])
        tokenizer.train_from_iterator("", trainer=trainer)
        for special_token in vocab_words:
            special_token_list += [AddedToken(special_token, single_word=True)]
        tokenizer.add_tokens(special_token_list)
        
        # Set up decoding
        tokenizer.decoder = decoders.ByteLevel()
        
        # Save the tokenizer
        tokenizer.save("challenger_tokenizer.json")

    tokenizer = RobertaTokenizerFast( 
                                tokenizer_file=args.tokenizer_path,  
                                do_lower_case=True,
                                bos_token='[BOS]',
                                eos_token='[EOS]',
                                unk_token='[UNK]',
                                sep_token='[SEP]',
                                cls_token='[CLS]',
                                pad_token='[PAD]',
                                mask_token='[MASK]',
                                padding_side = 'left',
                                model_max_length=1024,
                                max_len=1024,
                                )
    return tokenizer


"""## Data Loader"""

def convert2token_coverage(arr):
  arr = [int(x) for x in arr]
  arr.insert(0,-99)
  arr.append(-99)
  return arr

def convert2token_cnv(arr_str):
    arr_str = arr_str.replace("[","")
    arr_str = arr_str.replace("]","")
    arr_str = arr_str.replace("0","[NOCALL]")
    arr_str = arr_str.replace("1","[DUP]")
    arr_str = arr_str.replace("2","[DEL]")
    arr = arr_str.split(" ")
    arr.insert(0,"[CLS]")
    arr.append("[SEP]")
    return "".join(arr)

def data_loader(tokenizer):
    torch.manual_seed(2809)
    np.random.seed(2809)
    labelID2label = {0 : "[NOCALL]", 1 : "[DUP]", 2 : "[DEL]"}
    def preprocess_function(examples):
        read_depth =  examples['read_depth']
        model_inputs = {}
        model_inputs["input_ids"] = convert2token_coverage(read_depth)
        label = examples["label"]
        model_inputs["label_ids"] = tokenizer.encode("[CLS]"+(labelID2label[label]*1000)+"[SEP]")
        model_inputs["attention_mask"] = (np.array(model_inputs["input_ids"])!=(-1)).astype(int).tolist()
        model_inputs["position_ids"] = np.arange(1002)
        model_inputs["gene_ids"] = examples['gene_ind']
        return model_inputs

    train_dataset = load_dataset("parquet", data_files=args.input, split="train",cache_dir="../cache/")
    train_dataset = train_dataset.map(preprocess_function)
    train_dataset = train_dataset.remove_columns("chr")
    train_dataset = train_dataset.remove_columns("start")
    train_dataset = train_dataset.remove_columns("end")
    #train_dataset = train_dataset.remove_columns("read_depth")
    train_dataset = train_dataset.remove_columns("gene_ind")
    train_dataset = train_dataset.remove_columns("label")

    train_testvalid = train_dataset.train_test_split(test_size=0.01)
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'valid': train_testvalid['test']})

    return dataset


##Trainer Configuration"""
def set_trainer(roberta_model, dataset):
    trainer_args = TrainingArguments(
        args.output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        do_train=True,
        do_eval=True,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        num_train_epochs = args.num_epoch,
        #max_steps=10000,
        load_best_model_at_end = True,
        save_total_limit=5
        
    )

                
    # Define the Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.0, mask_replace_prob=1.0, random_replace_prob=0.0, cnv_mlm_probability=1.0, cnv_mask_replace_prob=1.0, cnv_random_replace_prob=0.0
    )

    
    class CustomCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)   
            if state.is_local_process_zero:
                if((state.global_step!=0) & (state.global_step%50 == 0)):
                    #nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall, nocall_f1, dup_f1, del_f1, overall_prec, overall_recall, overall_f1 = calculate_metrics(self._trainer.tp_nocall, self._trainer.tp_plus_fp_nocall, self._trainer.tp_duplication, self._trainer.tp_plus_fp_duplication, self._trainer.tp_deletion, self._trainer.tp_plus_fp_deletion, self._trainer.tp_plus_fn_nocall, self._trainer.tp_plus_fn_duplication, self._trainer.tp_plus_fn_deletion)
                    #SEG_nocall_prec, SEG_dup_prec, SEG_del_prec, SEG_nocall_recall, SEG_dup_recall, SEG_del_recall, SEG_nocall_f1, SEG_dup_f1, SEG_del_f1, SEG_overall_prec, SEG_overall_recall, SEG_overall_f1 = calculate_metrics(self._trainer.SEG_tp_nocall, self._trainer.SEG_tp_plus_fp_nocall, self._trainer.SEG_tp_duplication, self._trainer.SEG_tp_plus_fp_duplication, self._trainer.SEG_tp_deletion, self._trainer.SEG_tp_plus_fp_deletion, self._trainer.SEG_tp_plus_fn_nocall, self._trainer.SEG_tp_plus_fn_duplication, self._trainer.SEG_tp_plus_fn_deletion)
                    CLS_nocall_prec, CLS_dup_prec, CLS_del_prec, CLS_nocall_recall, CLS_dup_recall, CLS_del_recall, CLS_nocall_f1, CLS_dup_f1, CLS_del_f1, CLS_overall_prec, CLS_overall_recall, CLS_overall_f1 = calculate_metrics(self._trainer.CLS_tp_nocall, self._trainer.CLS_tp_plus_fp_nocall, self._trainer.CLS_tp_duplication, self._trainer.CLS_tp_plus_fp_duplication, self._trainer.CLS_tp_deletion, self._trainer.CLS_tp_plus_fp_deletion, self._trainer.CLS_tp_plus_fn_nocall, self._trainer.CLS_tp_plus_fn_duplication, self._trainer.CLS_tp_plus_fn_deletion)
                    cls_loss = torch.clone(self._trainer.latest_cls_loss)
                    cls_perfs = {
                                    "train/CLS/NOCALL_f1":round(CLS_nocall_f1,3) , "train/CLS/NOCALL_prec" : round(CLS_nocall_prec,3) , "train/CLS/NOCALL_recall": round(CLS_nocall_recall,3) ,
                                    "train/CLS/DUP_f1":round(CLS_dup_f1,3) , "train/CLS/DUP_prec" : round(CLS_dup_prec,3) , "train/CLS/DUP_recall": round(CLS_dup_recall,3) ,
                                    "train/CLS/DEL_f1":round(CLS_del_f1,3) , "train/CLS/DEL_prec" : round(CLS_del_prec,3) , "train/CLS/DEL_recall": round(CLS_del_recall,3) ,
                                    "train/CLS/Overall_f1":round(CLS_overall_f1,3) , "train/CLS/Overall_prec" : round(CLS_overall_prec,3) , "train/CLS/Overall_recall": round(CLS_overall_recall,3) ,
                                    "train/CLS/Loss":round(cls_loss.item(),3)
                                }
                    print(cls_perfs)
                    print(logs)
                

    def compute_metrics(eval_pred):
        (seg_logits, cls_logits), labels = eval_pred
        seg_predictions = np.argmax(seg_logits, axis=-1)
        cls_predictions = np.argmax(cls_logits, axis=-1)
        
        
        sample_labels = torch.zeros(cls_logits.shape[0]).to(int)
        dup_samples = (labels[1]==1).any(axis=1)
        del_samples = (labels[1]==2).any(axis=1)
        sample_labels[dup_samples] = 1
        sample_labels[del_samples] = 2

        masked_region = labels[1]!=(-100)
        labels = labels[1][masked_region]


        cls_nocall_labels = (sample_labels==0)*1
        cls_dup_labels = (sample_labels==1)*1
        cls_del_labels = (sample_labels==2)*1
        cls_nocall_preds = (cls_predictions==0)*1
        cls_dup_preds = (cls_predictions==1)*1
        cls_del_preds = (cls_predictions==2)*1


        cls_nocall_perf = {}
        cls_dup_perf = {}
        cls_del_perf = {}
                
        #seg_nocall_preds = [seg_nocall_preds] if isinstance(seg_nocall_preds, int) else seg_nocall_preds
        #seg_nocall_labels = [seg_nocall_labels] if isinstance(seg_nocall_labels, int) else seg_nocall_labels
        for metric_name, metric in [["accuracy",accuracy_score], ["precision",precision_score], ["recall",recall_score], ["f1",f1_score]]:
            cls_nocall_perf.update({metric_name:metric(cls_nocall_labels,cls_nocall_preds)})
            cls_dup_perf.update({metric_name:metric(cls_dup_labels,cls_dup_preds)})
            cls_del_perf.update({metric_name:metric(cls_del_labels,cls_del_preds)})      
        
        cls_overall_accuracy = (cls_dup_perf["accuracy"] + cls_del_perf["accuracy"]) / 2
        cls_overall_f1 = (cls_dup_perf["f1"] + cls_del_perf["f1"]) / 2
        cls_overall_prec = (cls_dup_perf["precision"] + cls_del_perf["precision"]) / 2
        cls_overall_recall = (cls_dup_perf["recall"] + cls_del_perf["recall"]) / 2
        cls_overall_perf = {"accuracy":cls_overall_accuracy,"f1":cls_overall_f1, "precision": cls_overall_prec, "recall":cls_overall_recall}
        
        all_perfs = {                    
                        "CLS_NOCALL_acc":cls_nocall_perf["accuracy"], "CLS_NOCALL_f1":cls_nocall_perf["f1"] , "CLS_NOCALL_prec" : cls_nocall_perf["precision"] , "CLS_NOCALL_recall": cls_nocall_perf["recall"] ,
                        "CLS_DUP_acc":cls_dup_perf["accuracy"], "CLS_DUP_f1":cls_dup_perf["f1"] , "CLS_DUP_prec" : cls_dup_perf["precision"] , "CLS_DUP_recall": cls_dup_perf["recall"] ,
                        "CLS_DEL_acc":cls_del_perf["accuracy"], "CLS_DEL_f1":cls_del_perf["f1"] , "CLS_DEL_prec" : cls_del_perf["precision"] , "CLS_DEL_recall": cls_del_perf["recall"] ,
                        "CLS_Overall_acc":cls_overall_perf["accuracy"], "CLS_Overall_f1":cls_overall_perf["f1"] , "CLS_Overall_prec":cls_overall_perf["precision"] , "CLS_Overall_recall":cls_overall_perf["recall"] ,
                        "CLS_Loss": round(trainer.latest_cls_loss.item(),3)
                    }
            
        return all_perfs
    
    trainer = CustomTrainer(
        model=roberta_model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, 
    )
    
    trainer.add_callback(CustomCallback(trainer)) 
    
    return trainer


if __name__ == "__main__":
    tokenizer = tokenization(create_vocab=False)
    dataset = data_loader(tokenizer)
    roberta_model = CHALLENGER_roberta_ForMaskedLM.from_pretrained(
        args.init_weight,
        tie_word_embeddings=False
    ).to(args.device)
    trainer = set_trainer(roberta_model, dataset)
    trainer.train()
    
    

      
