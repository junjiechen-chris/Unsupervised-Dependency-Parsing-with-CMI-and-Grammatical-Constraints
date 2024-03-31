# %%
from __future__ import annotations
from json import load
import torch
from torch import nn
import sys
sys.path.append('/home/chris/projects/dep_syntax-MI-wisteria')
sys.path.append('/work/gk77/k77015/projects/unsup_dep_parsing-stat_measures')

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Type, Set
from clm_sampler import FastMHSampler
from data import collate_fn_for_clm_sampler, UDDatasetForCLMSampler
from functools import partial


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    HfArgumentParser,
)

from torch.utils.data import DataLoader 
import numpy as np
from tqdm import tqdm
import pandas as pd
# import networkx as nx
from random import random, choices
import h5py
import json
from datetime import datetime
# from multiprocessing import Pool
import matplotlib.pyplot as plt
# # import seaborn as sns
# import importlib
# from random import random

from pathlib import Path
from copy import deepcopy
from src.common.utilities import vmap_list

from src.common.data import ModelArguments, DataArguments
import tarfile

# %%
torch.backends.cudnn.allow_tf32 = True
working_dir = '/home/chris/projects/dep_syntax-MI-wisteria'

# %%
@dataclass
class ModelArgumentsForCLMSampling(ModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    clm_model_str: str = field(default='facebook/opt-350m')
    gpu_id: int = field(default = 0)
    target_toks: int = field(default = 4096)

@dataclass
class DataArgumentsForCLMSampling(DataArguments):
    fn_hdf5: str = field(default=None)
    dir_hdf5:str = field(default=None)
    
    tar_member:Optional[str] = field(default = None)
    fn_upos2word:str = field(default='corpus/ud_en_upos2word.json')
    



@dataclass
class SamplingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    batch_size: int = field(default=64)
    batch_size_tokens: int = field(default=384)
    num_samples : int = field(default=32)
    num_tries : int = field(default=4)
    num_steps_between_samples : int = field(default=4)
    num_burn_in_steps : int = field(default=32)
    q_temperature: float = field(default=1.5)
    energy_temperature: float = field(default=1.)
    flag_use_upos_mask: bool = field(default=True)
    left_context_size: int = field(default=10)
    right_context_size: int = field(default=10)
    dependency_length_limit: int = field(default=12)
    flag_exclude_puncts : bool = field(default=True)
    energy_type: str = field(default='norm') # should be raw or norm
    qtmp_decay: float = field(default=0.01)
    flag_use_tf32_for_matmul:bool = field(default=True)
    energy_model:str = field(default='clm')
    flag_allow_x_postag:bool = field(default=True)
    sample_max_seqlen: int = field(default=42)
    flag_cache_energy: bool = field(default=True)
    flag_strip_accent: bool = field(default=False)
    
parser = HfArgumentParser(
    [ModelArgumentsForCLMSampling, DataArgumentsForCLMSampling, SamplingArguments]
)
model_args, data_args, sampling_args = parser.parse_args_into_dataclasses()#args=argparse_args)

print('data args:', data_args)
print('model_args:', model_args)
print('sampling_args:', sampling_args)

if sampling_args.flag_use_tf32_for_matmul:
    torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    device='cuda:{}'.format(model_args.gpu_id)
else:
    device = 'cpu'
print('using device {}'.format(device))

# %%

prop_tokenizer = AutoTokenizer.from_pretrained(model_args.model_str)
prop_model = AutoModelForMaskedLM.from_pretrained(model_args.model_str).to(device)


def pair_filter_base(sid, x, upos, flag_exclude_puncts=False, dependency_length_limit = 8):
    sampled_upos = set(['PRON', 'AUX', 'DET', 'NOUN', 'ADP', 'PROPN', 'VERB', 'NUM', 'ADJ', 'CCONJ', 'ADV', 'PART', 'INTJ', 'SYM', 'SCONJ'])
    if sampling_args.flag_allow_x_postag:
        sampled_upos.add('X')
    return_flags = [abs(x[1] - x[0]) < dependency_length_limit]
    if flag_exclude_puncts:
        return_flags.append(all(vmap_list(lambda x: x in sampled_upos, upos)))
    return all(return_flags)

pair_filter = partial(pair_filter_base, flag_exclude_puncts=sampling_args.flag_exclude_puncts, dependency_length_limit=sampling_args.dependency_length_limit)

dev_dataset = UDDatasetForCLMSampler(os.path.join(working_dir, data_args.dev_data_file), prop_tokenizer)
dev_dataset.sample(pair_filter=pair_filter, flag_strip_accent=sampling_args.flag_strip_accent)#nonsyndep_strategy)




def get_fullword_mask():
    fragment_piece = []
    for w, id in prop_tokenizer.vocab.items():
        if not w.startswith('##'):
            fragment_piece.append(id)

    fragment_piece = torch.tensor(fragment_piece)
    
    def vocab_mask_(vocab_mask, upos):
        for _mask in vocab_mask:
            for _mask_sample in _mask:
                _mask_sample.scatter_(0, fragment_piece, True)
    
    return vocab_mask_

def get_upos_mask():
    with open(os.path.join(working_dir,data_args.fn_upos2word)) as f:
        upos2word = json.loads(f.read())
    upos_set = upos2word.keys()
    upos_mask = {k:[] for k in upos_set}
    for upos in tqdm(upos_set, disable=True):
        wordset = upos2word[upos]
        for w, id in prop_tokenizer.get_vocab().items():
            if w.lower() in wordset:
                upos_mask[upos].append(id)
    upos_mask = {k: torch.tensor(v) for k, v in upos_mask.items()}
    
    def vocab_mask_(vocab_mask, upos):
        for _upos, _mask in zip(upos, vocab_mask):
            for _upos_sample, _mask_sample in zip(_upos, _mask):
                _mask_sample.scatter_(0, upos_mask[_upos_sample], True)
    
    return vocab_mask_

import importlib
import data
import clm_sampler 
importlib.reload(data)
collate_fn_for_clm_sampler = data.collate_fn_for_clm_sampler

dev_dataset.sample(pair_filter=pair_filter, left_context = sampling_args.left_context_size, right_context = sampling_args.right_context_size, max_seqlen=sampling_args.sample_max_seqlen)

fn_mask_vocab = get_fullword_mask() if not sampling_args.flag_use_upos_mask else get_upos_mask()

dev_dataset.bucketing(collate_fn=partial(collate_fn_for_clm_sampler, device=device, tokenizer=prop_tokenizer, dataset = dev_dataset, mask_vocab_ = fn_mask_vocab ), limit_tokens=sampling_args.batch_size_tokens, tokenizer=prop_tokenizer)
dev_loader = iter(dev_dataset)
print("finished data processing")



clm_model_str = model_args.clm_model_str#'facebook/opt-1.3b'
clm_model = AutoModelForCausalLM.from_pretrained(clm_model_str).to(device)
clm_tokenizer = AutoTokenizer.from_pretrained(clm_model_str)

mh_sampler = FastMHSampler(prop_model, prop_tokenizer, clm_model, clm_tokenizer, None, None, sampling_args.energy_model)
dt = h5py.special_dtype(vlen=str)
# %%
def build_h5_storage(dataset, target_dir, target_fn, num_samples, mode = 'r'):
    # print(k_list)
    if data_args.tar_member is None:
        print(target_dir)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        fn = os.path.join(target_dir, '{}.samples_{}.h5'.format(target_fn, num_samples))
        h5f = h5py.File(fn, mode)
    else:
        dir = os.path.join(working_dir, target_dir, '{}.samples_{}.h5dir'.format(target_fn, num_samples))
        Path(dir).mkdir(parents=True, exist_ok=True)
        h5fn = os.path.normpath(os.path.join(dir, '{}'.format(data_args.tar_member)))
        print('writing the samples to {}'.format(h5fn))
        h5f = h5py.File(h5fn, mode)
    
    grp = h5f.require_group('samples')
    grp_id = h5f.require_group('sample_ids')
    grp_accetpance_rate = h5f.require_group('acceptance_rate')
    
    for example in dataset.examples:
        sid = example.meta_data['sid']
        sentence_size = len(example._raw_tokens)
        grp.create_dataset(str(sid), shape=(sentence_size, sentence_size, num_samples, 2), dtype=dt) 
        grp_id.create_dataset(str(sid), shape=(sentence_size, sentence_size, num_samples, 2), dtype='i') 
        grp_accetpance_rate.create_dataset(str(sid), shape=(sentence_size, sentence_size), dtype='f')

    return h5f

# %%
h5f = build_h5_storage(dev_dataset, data_args.dir_hdf5, data_args.fn_hdf5, sampling_args.num_samples, mode = 'a')


with torch.no_grad():
    for step_id, (sample, _) in tqdm(enumerate(dev_loader), total = dev_dataset._get_chunked_lens()):
        mh_sampler.clm_energy_scorer.clear_energy_cache()
        inputs = mh_sampler.greedy_decode(sample.inputs, sample.token_info)
        
        sampling_method = mh_sampler.MultiTry_MH_sampling
        
        with torch.autocast(dtype=torch.bfloat16, enabled=False, device_type='cuda'):
            MH_sampling_output = sampling_method(inputs, 
                                                                    sample.token_info, 
                                                                    num_samples = sampling_args.num_samples, 
                                                                    num_iterations_per_sample=sampling_args.num_steps_between_samples, 
                                                                    burn_in_steps = sampling_args.num_burn_in_steps, 
                                                                    device=device, 
                                                                    target_toks=model_args.target_toks, 
                                                                    tmp=sampling_args.q_temperature, 
                                                                    mode='norm', 
                                                                    num_tries = sampling_args.num_tries,
                                                                    energy_tmp = sampling_args.energy_temperature,
                                                                    qtmp_decay= sampling_args.qtmp_decay,
                                                                    flag_cache_energy=sampling_args.flag_cache_energy,)
                                                                    
            
        
        x_samples = vmap_list(prop_tokenizer.convert_ids_to_tokens, MH_sampling_output.x_samples.cpu())
        y_samples = vmap_list(prop_tokenizer.convert_ids_to_tokens, MH_sampling_output.y_samples.cpu())
        word_samples = [[[x, y] for x, y in zip(bx, by)] for bx, by in zip(x_samples, y_samples)]
        id_samples = np.stack([MH_sampling_output.x_samples.cpu(), MH_sampling_output.y_samples.cpu()], axis=-1)
        for _sid, _word_idx, _word_samples, _id_samples, _ar in zip(sample.sid, sample.word_idx, word_samples, id_samples, MH_sampling_output.MH_acceptance_rate.cpu()):
            h5f['samples'][str(_sid)][_word_idx[0], _word_idx[1]] = _word_samples
            h5f['sample_ids'][str(_sid)][_word_idx[0], _word_idx[1]] = _id_samples
            h5f['acceptance_rate'][str(_sid)][_word_idx[0], _word_idx[1]] = _ar#@MH_sampling_output.MH_acceptance_rate.cpu()
        

h5f.close()

