# %%
from torch import nn
import torch

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type
import torch.optim as optim

from transformers import(
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoModelForCausalLM
    
)

import os
import sys

sys.path.append('/home/chris/projects/dep_syntax-with-surface_statistics')
sys.path.append('/work/gk77/k77015/projects/unsup_dep_parsing-stat_measures')

# from data import UDDatasetForPMI, collate_fn_for_pmi
from src.common.data import ModelArguments, DataArguments 
from data import UDDatasetForNEMI, collate_fn_for_nemi_training, collate_fn_for_nemi_fixz
# from model import BERTPMIWrapper

from functools import partial
from torch.utils.data import DataLoader 

from tqdm import tqdm


import h5py

from src.common.utilities import vmap_list

import numpy as np
from easydict import EasyDict as edict

from utilities import get_fullword_mask, get_upos_mask
from model import BERTfDivergenceEstimator, BERTWassersteinEstimator, GPTEnergyEstimator, CountBasedEstimator, MLMEnergyEstimator

# %%

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# %%

# %%
# device = torch.device('cpu')

# %%
# working_dir = '/home/chris/projects/dep_syntax-with-surface_statistics'
working_dir = '/work/gk77/k77015/projects/unsup_dep_parsing-stat_measures'

# %%


# %%

argparser_args = [
    "--model_str", "bert-base-uncased", 
    '--clm_model_str', 'facebook/opt-350m',
    "--dev_data_file", "{}/vinfo_data/en_ewt_extv2-ud-dev-10.conllu".format(working_dir),
    "--dev_sample_dir", "{}/cache/opt-samples/bert-base-uncased.facebook_opt-350m".format(working_dir),
    "--dev_sample_fn", "en_ewt_extv2-ud-dev-10.conllu.tar.uposmask_qt1.5_SSMH.samples_128",
    # "--dev_dataarguments", "{}/cache/opt-samples/bert-large-cased.facebook_opt-1.3b/en_ewt_extv2-ud-dev-10.conllu.tar.uposmask.samples_48.h5dir.args.pt".format(working_dir),
    "--batch_size", "128", 
    "--seed", "1", 
    '--learning_rate', '1e-4',
    '--gpu_id', '0',
]

# %%
@dataclass
class ModelArgumentsForNEMI(ModelArguments):
    model_str: str = field(default='bert-base-uncased')
    clm_model_str: str = field(default='facebook/opt-350m')
    mlm_model_str: str = field(default='xlm-roberta-base')
    energy_model:str = field(default='clm')
    flag_use_fast_sampler: bool = field(default=False)

@dataclass
class DataArgumentsForNEMI(DataArguments):
    dev_sample_dir: str = field(default="{}/cache/opt-samples/bert-base-uncased.facebook_opt-350m".format(working_dir))
    dev_sample_fn: str = field(default=None)
    dev_data_file: str = field(default="{}/vinfo_data/en_ewt_extv2-ud-dev-10.conllu".format(working_dir))

    dev_dataarguments: str = field(default=None)

@dataclass
class TrainingArguments:
    learning_rate: float = field(default=1e-4)
    seed: int = field(default=1)
    batch_size: int = field(default=32)
    gpu_id: int = field(default=0)
    flag_add_guards_to_clm_tokenizer: bool = field(default=False)
    flag_use_bf16:bool = field(default=False)
    
    
    
parser = HfArgumentParser(
    [ModelArgumentsForNEMI, DataArgumentsForNEMI, TrainingArguments]
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()#args=argparser_args)
set_seed(training_args.seed)
print(model_args)
print(data_args)
print(training_args)
# data_args.dev_data_file = "{}/vinfo_data/en_ewt_extv2-ud-dev-10.conllu".format(working_dir)

device = torch.device('cuda:{}'.format(training_args.gpu_id) if torch.cuda.is_available() else 'cpu')

def build_h5py(fn_samples):
    h5f_samples = h5py.File(fn_samples, 'r')
    return h5f_samples, None
    
# h5f_samples, _ = build_h5py(os.path.join(data_args.dev_sample_dir, data_args.dev_sample_fn))


prop_tokenizer = AutoTokenizer.from_pretrained(model_args.model_str)

filter_dependency_only = lambda sent, pair: True

dev_dataset = UDDatasetForNEMI(os.path.join(working_dir, data_args.dev_data_file), prop_tokenizer, flag_training=False)

# d_dict = edict({})
d_dict = edict({

    # 'bert_wasserstein': BERTWassersteinEstimator(device=device),
    'countkl': CountBasedEstimator()
})
if model_args.energy_model == 'clm':
    clm_model = AutoModelForCausalLM.from_pretrained(model_args.clm_model_str).to(device)
    clm_tokenizer = AutoTokenizer.from_pretrained(model_args.clm_model_str)
    clm_model.eval()
    d_dict = edict({'clm': GPTEnergyEstimator(clm_model, clm_tokenizer, training_args.flag_add_guards_to_clm_tokenizer, use_fast_sampler=model_args.flag_use_fast_sampler), **d_dict})
elif model_args.energy_model == 'mlm':
    mlm_model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base').to(device)
    mlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    d_dict = edict({'mlm': MLMEnergyEstimator(mlm_model, mlm_tokenizer), **d_dict})
else:
    raise NotImplementedError(f'energy model needs to be either clm or mlm, now is {model_args.energy_model}')


print(d_dict, flush=True)
# with torch.autocast(dtype=torch.bfloat16, enabled=training_args.flag_use_bf16, device_type='cuda'):
with torch.no_grad():
    for _, d in d_dict.items():
        h5f_samples, h5f_scores = d.build_h5(data_args.dev_sample_dir, data_args.dev_sample_fn)
        

        dev_dataset.sample_sampled_pairs(h5f=h5f_samples, filter_func=filter_dependency_only)
        collate_fn = partial(collate_fn_for_nemi_fixz, tokenizer=prop_tokenizer, device=device) 
        train_iter = DataLoader(dev_dataset.data, batch_size=1, shuffle=True, collate_fn=collate_fn)

        for inputs, _ in tqdm(train_iter):
            (raws_p, inputs_p), (raws_q, inputs_q) = inputs.iter
            
            x, y = inputs.word_idx
            energy_d = d.forward_to_h5(inputs.sid, inputs.word_idx, h5f_scores, raws_p, raws_q, device = device)

        h5f_scores.close()
        h5f_samples.close()

