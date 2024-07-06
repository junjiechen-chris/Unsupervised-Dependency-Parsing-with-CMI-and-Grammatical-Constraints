from torch import nn
from easydict import EasyDict as edict
import torch
from torch.nn.utils.rnn import pad_sequence
from geomloss import SamplesLoss 
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, AutoModel
from clm_sampler import  VinfoModelFast
import scipy
from collections import Counter
import h5py
import os
import numpy as np


class CountBasedEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, positive_samples, negative_samples, *args, **kwargs):
        marginal_x_samples = [_[0] for _ in positive_samples]
        marginal_y_samples = [_[1] for _ in positive_samples]
        joint_count = Counter(positive_samples)
        num_samples = len(positive_samples)
        marginal_x_count = Counter(marginal_x_samples)
        marginal_y_count = Counter(marginal_y_samples)
        
        kl = 0.
        
        for joint_atom, count in joint_count.items():
            p_joint = count / num_samples
            p_x = marginal_x_count[joint_atom[0]] / num_samples
            p_y = marginal_y_count[joint_atom[1]] / num_samples
            kl += p_joint * np.log(p_joint) - np.log(p_x * p_y)
        
        return torch.tensor(kl)

    def forward_to_h5(self, sid, word_idx, h5f_scores,  positive_samples, negative_samples, device = 'cuda:0', *args, **kwargs):
        outputs = self(positive_samples, negative_samples)
        h5f_scores['kl_count'][str(sid)][word_idx[0], word_idx[1]] = outputs.cpu().numpy()
            
    def build_h5(self, dir_samples, fn_samples):
        path_samples = os.path.join(dir_samples, fn_samples)
        h5f_samples = h5py.File(path_samples, 'r')
        fn_scores = 'scores.{}'.format(fn_samples)
        path_scores = os.path.join(dir_samples, fn_scores)
        h5f_scores = h5py.File(path_scores, 'a')
        klgrp = h5f_scores.require_group('kl_count')
        for sid, samples in h5f_samples['samples'].items():
            mtx_l  = samples.shape[0] # sentence length
            klgrp.require_dataset(sid, shape=(mtx_l, mtx_l), dtype='f')
        return h5f_samples, h5f_scores
        

    def forward_js(self, x_samples, y_samples):
        raise NotImplementedError("The count-based KL-MI estimator is not yet implemented")

    
    
class BERTWassersteinEstimator(nn.Module):
    def __init__(self, base_model_str = 'bert-base-uncased', p=1, device = 'cuda:0'):
        super().__init__()
        assert base_model_str.startswith('bert'), 'This class is only implemented for BERT as base model'
        self.base_model = AutoModel.from_pretrained(base_model_str).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_str)
        self.fn_loss = SamplesLoss(loss="sinkhorn", p=p, blur=.05)
        self.device = device
   
   
    @torch.no_grad()
    def forward(self, positive_samples, negative_samples, device = 'cuda:0', *args, **kwargs):
        if all([p==n for p, n in zip(positive_samples, negative_samples)]): return torch.tensor(0.)
        inputs_p = self.base_tokenizer(positive_samples, return_tensors="pt", padding=True).to(self.device)
        inputs_n = self.base_tokenizer(negative_samples, return_tensors="pt", padding=True).to(self.device)
        positive_logits = self.base_model(**inputs_p).last_hidden_state[:, 0, :]
        negative_logits = self.base_model(**inputs_n).last_hidden_state[:, 0, :]
        
        wasserstein_distance = self.fn_loss(positive_logits, negative_logits)
        
        return wasserstein_distance

    def forward_to_h5(self, sid, word_idx, h5f_scores,  positive_samples, negative_samples, device = 'cuda:0', *args, **kwargs):
        outputs = self(positive_samples, negative_samples, device = device)
        h5f_scores['wasserstein'][str(sid)][word_idx[0], word_idx[1]] = outputs.cpu().numpy()
        

    def build_h5(self, dir_samples, fn_samples):
        path_samples = os.path.join(dir_samples, fn_samples)
        h5f_samples = h5py.File(path_samples, 'r')
        fn_scores = 'scores.{}'.format(fn_samples)
        path_scores = os.path.join(dir_samples, fn_scores)
        h5f_scores = h5py.File(path_scores, 'a')
        klgrp = h5f_scores.require_group('wasserstein')
        for sid, samples in h5f_samples['samples'].items():
            mtx_l  = samples.shape[0] # sentence length
            klgrp.require_dataset(sid, shape=(mtx_l, mtx_l), dtype='f')
        return h5f_samples, h5f_scores

    
class MLMEnergyEstimator(nn.Module):
    def __init__(self, mlm_model = None, mlm_tokenizer = None) -> None:
        super().__init__()
        assert mlm_model is not None and mlm_tokenizer is not None, 'Please provide a valid MLM model and tokenizer'
        self.mlm_model = mlm_model# if not clm_model is None else AutoModelForCausalLM.from_pretrained('facebook/opt-350m')
        self.mlm_tokenizer = mlm_tokenizer# if not clm_tokenizer is None else AutoTokenizer.from_pretrained('facebook/opt-350m')
        self.mlm_energy_scorer = VinfoModel(None, None, None, None, mlm_model, mlm_tokenizer, )

    def forward(self, positive_samples, negative_samples, mode='norm', device = 'cuda', target_toks = 2048, *args, **kwargs):
        positive_energy = self.mlm_energy_scorer.forward_energy_cached(positive_samples, None, mode=mode, target_toks=target_toks, device=device, flag_bypass_cache=True, energy_model='mlm')
        negative_energy = self.mlm_energy_scorer.forward_energy_cached(negative_samples, None, mode=mode, target_toks=target_toks, device=device, flag_bypass_cache=True, energy_model='mlm')
        return edict({
            'delta_mean_energy': torch.mean(positive_energy.energy) - torch.mean(negative_energy.energy),
            't-test': scipy.stats.ttest_ind(positive_energy.energy.cpu(), negative_energy.energy.cpu(), equal_var=False),
        })
        
    def forward_to_h5(self, sid, word_idx, h5f_scores, positive_samples, negative_samples, mode='norm', device = 'cuda', target_toks = 1024, *args, **kwargs):
        outputs = self(positive_samples, negative_samples, mode=mode, device=device, target_toks=target_toks)
        # print(outputs)
        # print(outputs.delta_mean_energy.cpu())
        h5f_scores['energy-delta'][str(sid)][word_idx[0], word_idx[1]] = outputs.delta_mean_energy.cpu()
        # print(outputs['t-test'])
        h5f_scores['energy-ttest'][str(sid)][word_idx[0], word_idx[1]] = outputs['t-test']
        
    
    def build_h5(self, dir_samples, fn_samples):
        path_samples = os.path.join(dir_samples, fn_samples)
        h5f_samples = h5py.File(path_samples, 'r')
        fn_scores = 'scores.{}'.format(fn_samples)
        path_scores = os.path.join(dir_samples, fn_scores)
        h5f_scores = h5py.File(path_scores, 'a')
        grp_delta = h5f_scores.require_group('energy-delta')
        grp_ttest = h5f_scores.require_group('energy-ttest')
        for sid, samples in h5f_samples['samples'].items():
            mtx_l  = samples.shape[0] # sentence length
            grp_delta.require_dataset(sid, shape=(mtx_l, mtx_l), dtype='f')
            grp_ttest.require_dataset(sid, shape=(mtx_l, mtx_l, 2), dtype='f')
        return h5f_samples, h5f_scores


   
class GPTEnergyEstimator(nn.Module):
    def __init__(self, clm_model = None, clm_tokenizer = None, flag_add_guards_to_clm_tokenizer = False, use_fast_sampler = False) -> None:
        super().__init__()
        assert clm_model is not None and clm_tokenizer is not None, 'Please provide a valid CLM model and tokenizer'
        self.clm_model = clm_model #if not clm_model is None else AutoModelForCausalLM.from_pretrained('facebook/opt-350m')
        self.clm_tokenizer = clm_tokenizer #if not clm_tokenizer is None else AutoTokenizer.from_pretrained('facebook/opt-350m')
        if not use_fast_sampler:
            self.clm_energy_scorer = VinfoModel(None, None, clm_model, clm_tokenizer, None, None, flag_add_guards_to_clm_tokenizer=flag_add_guards_to_clm_tokenizer)
        else:
            self.clm_energy_scorer = VinfoModelFast(None, None, clm_model, clm_tokenizer, None, None, flag_add_guards_to_clm_tokenizer=flag_add_guards_to_clm_tokenizer)

    def forward(self, positive_samples, negative_samples, mode='norm', device = 'cuda', target_toks = 8192, *args, **kwargs):
        positive_energy = self.clm_energy_scorer.forward_energy_cached(positive_samples, None, mode=mode, target_toks=target_toks, device=device, flag_bypass_cache=True, energy_model='clm')
        negative_energy = self.clm_energy_scorer.forward_energy_cached(negative_samples, None, mode=mode, target_toks=target_toks, device=device, flag_bypass_cache=True, energy_model='clm')
        # print(-(positive_energy.energy.mean()  - negative_energy.energy.mean()))
        return edict({
            'delta_mean_energy': torch.mean(positive_energy.energy) - torch.mean(negative_energy.energy),
            't-test': scipy.stats.ttest_ind(positive_energy.energy.cpu(), negative_energy.energy.cpu(), equal_var=False),
        })
        
    def forward_to_h5(self, sid, word_idx, h5f_scores, positive_samples, negative_samples, mode='norm', device = 'cuda', target_toks = 8192, *args, **kwargs):
        outputs = self(positive_samples, negative_samples, mode=mode, device=device, target_toks=target_toks)
        # print(outputs)
        # print(outputs.delta_mean_energy.cpu())
        # print(word_idx, outputs.delta_mean_energy.cpu())
        h5f_scores['energy-delta'][str(sid)][word_idx[0], word_idx[1]] = outputs.delta_mean_energy.cpu()
        # print(outputs['t-test'])
        h5f_scores['energy-ttest'][str(sid)][word_idx[0], word_idx[1]] = outputs['t-test']
        
    
    def build_h5(self, dir_samples, fn_samples):
        path_samples = os.path.join(dir_samples, fn_samples)
        h5f_samples = h5py.File(path_samples, 'r')
        fn_scores = 'scores.{}'.format(fn_samples)
        path_scores = os.path.join(dir_samples, fn_scores)
        print(f'save to {path_scores}')
        h5f_scores = h5py.File(path_scores, 'w')
        grp_delta = h5f_scores.require_group('energy-delta')
        grp_ttest = h5f_scores.require_group('energy-ttest')
        for sid, samples in h5f_samples['samples'].items():
            mtx_l  = samples.shape[0] # sentence length
            grp_delta.require_dataset(sid, shape=(mtx_l, mtx_l), dtype='f')
            # grp_delta[sid].fillvalue(-1000)
            grp_ttest.require_dataset(sid, shape=(mtx_l, mtx_l, 2), dtype='f')
        return h5f_samples, h5f_scores
   
    
class BERTfDivergenceEstimator(nn.Module):
    # This implements a neural estimator based on https://en.wikipedia.org/wiki/F-divergence#cite_note-:1-2
    # The sample is given by shuffling the (x,y) samples from the GPT
    
    # This implmentation supposes a BERT model followed by a shallow MLP to compute the $$f(x, y, z)$$ function
    ne_hfn_dict = {
        'kldiv': lambda x: x* torch.log(x),
        'jsdiv': lambda x: -(x+1) * torch.log(0.5*(x+1)) + x * torch.log(x) ,
        'squared hellinger': lambda x: (torch.sqrt(x) - 1)**2,
        'pearson chi-square': lambda x: (x-1)**2,
        'neyman chi-square': lambda x: 1/x - 1
    }
    
    def __init__(self, base_model_str = 'bert-base-cased', h_fn_str = 'kldiv'):
        super().__init__()
        assert base_model_str.startswith('bert'), 'This class is only implemented for BERT as base model'
        self.base_model = BertModel.from_pretrained(base_model_str)
        hidden_size = self.base_model.config.hidden_size
        self.out_proj = nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            nn.ELU(),
            nn.Linear(256, 1)
            )
        self.fn_h = self.ne_hfn_dict[h_fn_str]
        

    def _forward_ne_divergence(self, positive_samples, negative_samples):
        fn_h = self.fn_h
        
        positive_logits = self.base_model(**positive_samples).pooler_output
        negative_logits = self.base_model(**negative_samples).pooler_output
        
        positive_scores = self.out_proj(positive_logits).squeeze(-1)
        negative_scores = self.out_proj(negative_logits).squeeze(-1)
        
        lb_t1 = torch.mean(positive_scores)
        lb_t2 = torch.mean(fn_h(negative_scores))
        
        return lb_t1 - lb_t2
    

    def _forward_nekl(self, positive_samples, negative_samples, fn_h = 'kldiv'):
        assert fn_h == 'kldiv', "This implementation of the Donsker-Varadhan bound is for KL-divergence"
        
        with torch.no_grad():
            positive_logits = self.base_model(**positive_samples).pooler_output
            negative_logits = self.base_model(**negative_samples).pooler_output
        
        # print("F-divergence NE: positive logit 2 norm", torch.median(torch.norm(positive_logits, dim=-1)))
        # print("F-divergence NE: negative logit 2 norm", torch.median(torch.norm(negative_logits, dim=-1)))
        
        positive_scores = self.out_proj(positive_logits).squeeze(-1)
        negative_scores = self.out_proj(negative_logits).squeeze(-1)
        # print("F-divergence NE: positive score", positive_scores)
        # print("F-divergence NE: negative score", negative_scores)
        
        
        lb_t1 = positive_scores
        lb_t2 = negative_scores.exp()
        mi_lb = torch.mean(lb_t1) - torch.logsumexp(negative_scores, dim=-1) + torch.log(torch.tensor(negative_scores.shape[0]).float())
        
        # lb_t1 = torch.mean(positive_scores)
        # lb_t2 = torch.logsumexp(negative_scores, dim=-1) - torch.log(torch.tensor(negative_scores.shape[0]).float())
        # print("F-divergence NE: negative score mean", lb_t2)
        
        return mi_lb, lb_t1, lb_t2
        