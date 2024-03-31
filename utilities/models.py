# from turtle import back
import sre_parse
from typing import Any, Optional, List, Dict, Tuple, Union, Type, Set
from .data import UDSentence, CoNLLDataset

from tqdm import tqdm
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    TrainingArguments,
    AutoTokenizer,
    BatchEncoding,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from transformers.modeling_outputs import MaskedLMOutput

# from modeling_opengpt2 import OpenGPT2LMHeadModel
import numpy as np
from torch.optim import Optimizer

# import networkx as nx
from copy import copy
from datetime import datetime
import os
import json
from collections import OrderedDict
from .auxobjs import DictObj, ListObj
import math
from copy import deepcopy

bert2gpt_exclude_characters = ['[CLS]', '[SEP]', '[PAD]']

def remove_cls_and_pad(input_toks):    
    return [tok for tok in input_toks if tok not in bert2gpt_exclude_characters]

class MHSampler(nn.Module):
    def __init__(self, model, mlm_cache) -> None:
        super().__init__()
        self.model = model
        self.mlm_cache = mlm_cache
        # self.tokenizer = tokenizer
    
    
    def MultiTry_MH_sampling(self, b_encodings, tokenizer, b_x_idx, b_y_idx, b_x_idx_mask, b_y_idx_mask, b_x_conditional_mask, b_y_conditional_mask, num_iterations_per_sample=4, num_samples=4, burn_in_steps=4, device='cuda:0', target_toks = 8192, tmp = 1., mode = 'raw', energy_tmp = 1., num_tries = 5):
        model = self.model
        mlm_cache = self.mlm_cache
        b_encodings = deepcopy(b_encodings)
        b_encodings_bak = deepcopy(b_encodings)
        # print('MTM: entrance b_encodings.input_ids\n', b_encodings.input_ids)
        mlm_cache.clear_cache()
        b_bz, b_max_seqlen = b_encodings.input_ids.size()
        _, b_x_size = b_x_idx.size()
        _, b_y_size = b_y_idx.size()
        vocab_size = model.vocab_size
        b_xy_idx = torch.cat([b_x_idx, b_y_idx], dim=1)

        b_bid_mlm = torch.arange(b_bz, device = device)

        b_idx_group = [(b_x_idx, b_x_idx_mask, b_x_conditional_mask), (b_y_idx, b_y_idx_mask, b_y_conditional_mask)]


        total_iterations = num_samples * num_iterations_per_sample + burn_in_steps
        samples = []
        samples_energy = []
        samples_logprobs = []
        MH_accept_cnt = 0
        MH_decision_cnt = 0
        samples_accept_decision = []
        
        for n in tqdm(range(2*total_iterations), disable=True):
            b_idx, b_idx_mask, b_conditional_mask = b_idx_group[n % 2]

            # extract the old sample
            b_original_samples = b_encodings.input_ids.gather(1, b_idx) # (b_bz, b_x_size)
            b_proposal_o_input_ids = b_encodings.input_ids
            
            #construct the proposal distribution
            b_samples_input_ids = b_encodings.input_ids.scatter(1, b_idx, value=tokenizer.mask_token_id)
            b_encodings['input_ids'] = b_samples_input_ids
            
            b_cache_outputs = mlm_cache.search_from_cache(b_encodings, b_bid_mlm, b_xy_idx, b_idx, model.vocab_size)
            if b_cache_outputs.b_miss_count > 0:
                b_miss_proposal_target_logits = model.model(**b_cache_outputs.b_encodings).logits.gather(1, b_cache_outputs.b_idx.unsqueeze(2).expand(-1, -1, vocab_size)) #(b_bz, b_idx_size, vocab_size)
                b_proposal_target_logits = mlm_cache.rebuild_full_output_and_cache_miss_output(b_cache_outputs.b_cached_logits, b_miss_proposal_target_logits, b_cache_outputs.b_mask_miss_cache, b_cache_outputs.b_bid_xyid).rebuilt_outputs
            else:
                b_proposal_target_logits = b_cache_outputs.b_cached_logits

            # sample proposals
            assert b_proposal_target_logits.size(1) == 1
            assert (torch.log_softmax(b_proposal_target_logits/tmp, dim=2) + torch.logical_not(b_conditional_mask)).size(1) == 1
            b_proposal_target_logdists = (torch.log_softmax(b_proposal_target_logits/tmp, dim=2) + torch.logical_not(b_conditional_mask)* -1e9).squeeze(1)  #(b_bz, vocab_size)
            b_proposal_samples = torch.multinomial(b_proposal_target_logdists.exp(), num_tries, replacement=True) #(b_bz, 1 (samples))

            b_proposal_samples_logprobs = b_proposal_target_logdists.gather(1, b_proposal_samples) #(b_bz, num_tries)   #.sum(-1) #(b_bz)
            b_original_samples_logprobs = b_proposal_target_logdists.gather(1, b_original_samples) #(b_bz, 1)

            #compute energy
            b_proposal_p_input_ids = b_encodings.input_ids.repeat(1, num_tries).view(b_bz*num_tries, b_max_seqlen).scatter(1, b_idx.repeat(1, num_tries).view(b_bz*num_tries, 1), src = b_proposal_samples.view(b_bz*num_tries, 1))
            b_proposal_input_ids = torch.cat([b_proposal_p_input_ids.view(b_bz,num_tries*b_max_seqlen), b_proposal_o_input_ids], dim=1).view(b_bz*(num_tries+1), b_max_seqlen)
            b_proposal_attention_mask = b_encodings.attention_mask.repeat(1, num_tries+1).view(b_bz*(num_tries+1), b_max_seqlen)
            b_proposal_encodings = BatchEncoding({
                'attention_mask': b_proposal_attention_mask,
                'input_ids': b_proposal_input_ids
            })
            
            b_bid = torch.arange(b_bz, device=device).unsqueeze(1).repeat(1, num_tries+1).view(b_bz*(num_tries+1), 1)
            b_xy_idx_energy = torch.cat([b_x_idx, b_y_idx], dim=1).repeat(1, num_tries+1).view(b_bz*(num_tries+1), 2)

            b_proposal_energy, b_original_energy = model.forward_gpt_energy_cached(b_proposal_encodings, b_xy_idx_energy, b_bid, device=device, target_toks = 65536, mode = mode).energy.view(b_bz, num_tries+1).split([num_tries, 1], dim=1)

            MTM_log_weight = (-b_proposal_energy - b_proposal_samples_logprobs + 0) # -> assuming the \psi(x, x') =1
            
            MTM_sample_idx = torch.multinomial(MTM_log_weight.exp()+1e-9, 1).view(b_bz, 1)
            MTM_sample = b_proposal_samples.gather(1, MTM_sample_idx)
            MTM_sample_logprobs = b_proposal_samples_logprobs.gather(1, MTM_sample_idx)
            MTM_mask_selection = torch.zeros(b_bz, num_tries, device=device).scatter(1, MTM_sample_idx, 1).bool()
            MTM_original_log_weight = (-b_original_energy - b_original_samples_logprobs + 0)

            # accept/reject proposals
            proposal_accept_rate = torch.minimum(torch.ones(b_bz, device=device), ((torch.logsumexp(MTM_log_weight, dim=1))-torch.logsumexp(torch.cat([(MTM_log_weight + -1e9*MTM_mask_selection), MTM_original_log_weight], dim=1), dim=1)).exp())
            proposal_accept_decision = torch.bernoulli(proposal_accept_rate).bool() #(b_bz)
            MH_accept_cnt += proposal_accept_decision
            MH_decision_cnt += 1
            samples_accept_decision.append(proposal_accept_decision)

            current_samples = torch.where(proposal_accept_decision.unsqueeze(1), MTM_sample, b_original_samples)
            samples.append(current_samples)
            b_encodings['input_ids'] = b_encodings.input_ids.scatter(1, b_idx, src = current_samples)

            current_energy = torch.where(proposal_accept_decision, b_proposal_energy.gather(1, MTM_sample_idx).view(b_bz), b_original_energy.squeeze(1))
            samples_energy.append(current_energy)

            current_logprobs = torch.where(proposal_accept_decision, MTM_sample_logprobs.view(b_bz), b_original_samples_logprobs.view(b_bz))
            samples_logprobs.append(current_logprobs)

            # print('================')

        samples = torch.cat(samples, dim=1)
        samples_energy = torch.stack(samples_energy, dim=1)
        samples_logprobs = torch.stack(samples_logprobs, dim=1)

        # print('MH sampling: chain size:', samples.size())
        # print('MH sampling: chain logprobs:', samples_logprobs.size())

        assert len(samples.size()) == 2
        assert len(samples_energy.size()) == 2
        assert len(samples_logprobs.size()) == 2
        
        # print(samples_accept_rate)

        x_samples, y_samples = samples[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]
        x_energy, y_energy = samples_energy[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples_energy[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]
        x_logprobs, y_logprobs = samples_logprobs[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples_logprobs[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]

        return DictObj({
            'x_samples': x_samples,
            'y_samples': y_samples,
            'x_energy': x_energy,
            'y_energy': y_energy,
            'x_logprobs': x_logprobs,
            'y_logprobs': y_logprobs,
            'samples_energy': samples_energy,
            'samples_accept_decision': torch.stack(samples_accept_decision, dim=0),
            'MH_accept_rate': MH_accept_cnt/MH_decision_cnt,
            'full_x_samples': samples[:, ::2],
            'full_y_samples': samples[:, 1::2],
            'samples': samples
        })

    def MH_sampling(self, b_encodings, tokenizer, b_x_idx, b_y_idx, b_x_idx_mask, b_y_idx_mask, b_x_conditional_mask, b_y_conditional_mask, num_iterations_per_sample=4, num_samples=4, burn_in_steps=4, device='cuda:0', target_toks = 8192, mode = 'norm', tmp = 1., energy_tmp = 1.):
        model = self.model
        mlm_cache = self.mlm_cache
        b_encodings = deepcopy(b_encodings)
        mlm_cache.clear_cache()
        b_bz, b_max_seqlen = b_encodings.input_ids.size()
        _, b_x_size = b_x_idx.size()
        _, b_y_size = b_y_idx.size()
        vocab_size = model.vocab_size
        b_xy_idx = torch.cat([b_x_idx, b_y_idx], dim=1)

        b_bid_mlm = torch.arange(b_bz, device = device)
        b_idx_group = [(b_x_idx, b_x_idx_mask, b_x_conditional_mask), (b_y_idx, b_y_idx_mask, b_y_conditional_mask)]


        total_iterations = num_samples * num_iterations_per_sample + burn_in_steps
        samples = []
        samples_energy = [model.forward_gpt_energy_cached(b_encodings, b_xy_idx, device=device, target_toks = target_toks, mode=mode).energy]
        samples_logprobs = []
        for n in tqdm(range(2*total_iterations), disable=True):
            b_idx, b_idx_mask, b_conditional_mask = b_idx_group[n % 2]

            b_original_samples = b_encodings.input_ids.gather(1, b_idx) # (b_bz, b_x_size)
            b_original_energy = samples_energy[-1]#.to(device)

            # print('MH sampling: b_original_samples', b_original_samples)


            # prepare proposal distribution
            b_samples_input_ids = b_encodings.input_ids.scatter(1, b_idx, value=tokenizer.mask_token_id)
            b_encodings['input_ids'] = b_samples_input_ids

            b_cache_outputs = mlm_cache.search_from_cache(b_encodings, b_bid_mlm, b_xy_idx, b_idx, model.vocab_size)
            if b_cache_outputs.b_miss_count > 0:
                b_miss_proposal_target_logits = model.model(**b_cache_outputs.b_encodings).logits.gather(1, b_cache_outputs.b_idx.unsqueeze(2).expand(-1, -1, vocab_size)) #(b_bz, b_idx_size, vocab_size)
                b_proposal_target_logits = mlm_cache.rebuild_full_output_and_cache_miss_output(b_cache_outputs.b_cached_logits, b_miss_proposal_target_logits, b_cache_outputs.b_mask_miss_cache, b_cache_outputs.b_bid_xyid).rebuilt_outputs
            else:
                b_proposal_target_logits = b_cache_outputs.b_cached_logits

            # sample candidates
            assert b_proposal_target_logits.size(1) == 1
            assert (torch.log_softmax(b_proposal_target_logits, dim=2) + torch.logical_not(b_conditional_mask)).size(1) == 1
            b_proposal_target_logdists = (torch.log_softmax(b_proposal_target_logits/tmp, dim=2) + torch.logical_not(b_conditional_mask)* -1e9).squeeze(1)  #(b_bz, vocab_size)
            b_proposal_samples = torch.multinomial(b_proposal_target_logdists.exp(), 1) #(b_bz, 1 (samples))

            b_proposal_samples_logprobs = b_proposal_target_logdists.gather(1, b_proposal_samples).sum(-1) #(b_bz)
            b_original_samples_logprobs = b_proposal_target_logdists.gather(1, b_original_samples).sum(-1)



            #compute energy
            b_proposal_encodings = BatchEncoding({
                **b_encodings,
                'input_ids': b_encodings.input_ids.scatter(1, b_idx, src=b_proposal_samples)
            })

            b_proposal_energy = model.forward_gpt_energy_cached(b_proposal_encodings, b_xy_idx, device=device, target_toks = target_toks, mode=mode).energy

            proposal_accept_rate = torch.minimum(torch.ones_like(-b_proposal_energy, device=device), ((-b_proposal_energy/energy_tmp + b_original_samples_logprobs) - (-b_original_energy/energy_tmp + b_proposal_samples_logprobs)).exp())
            proposal_accept_decision = torch.bernoulli(proposal_accept_rate).bool() #(b_bz)

            current_samples = torch.where(proposal_accept_decision.unsqueeze(1), b_proposal_samples, b_original_samples)
            samples.append(current_samples)
            b_encodings['input_ids'] = b_encodings.input_ids.scatter(1, b_idx, src = current_samples)

            current_energy = torch.where(proposal_accept_decision, b_proposal_energy, b_original_energy)
            samples_energy.append(current_energy)

            current_logprobs = torch.where(proposal_accept_decision, b_proposal_samples_logprobs, b_original_samples_logprobs)
            samples_logprobs.append(current_logprobs)

            # print('================')

        samples = torch.cat(samples, dim=1)
        samples_energy = torch.stack(samples_energy, dim=1)
        samples_logprobs = torch.stack(samples_logprobs, dim=1)

        assert len(samples.size()) == 2
        assert len(samples_energy.size()) == 2
        assert len(samples_logprobs.size()) == 2
        
        # print(samples.size())

        x_samples, y_samples = samples[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]
        x_energy, y_energy = samples_energy[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples_energy[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]
        x_logprobs, y_logprobs = samples_logprobs[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples_logprobs[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]

        return DictObj({
            'x_samples': x_samples,
            'y_samples': y_samples,
            'x_energy': x_energy,
            'y_energy': y_energy,
            'x_logprobs': x_logprobs,
            'y_logprobs': y_logprobs,
            # 'reference_energy': b_reference_energy
        })

    
class CustomGPTTokenizer(GPT2Tokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(args, kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        
        print('GPTtokenizer debug:', token_ids_0, token_ids_1)

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

class VinfoModel(nn.Module):
    def __init__(
        self, model_name="bert-base-cased", tokenizer=None, flag_debug=False
    ) -> None:
        assert tokenizer is not None
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            config=config,
        )
        self.model.resize_token_embeddings(len(tokenizer))
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        gpt_config = AutoConfig.from_pretrained('gpt2')
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token = True, add_prefix_space = False)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        self.model.eval()
        self.vocab_size = config.vocab_size
        self.gpt_vocab_size = gpt_config.vocab_size
        self.flag_debug = flag_debug
        self.energy_cache = {}

    def clear_energy_cache(self):
        # using (bid, x_id, y_id)
        self.energy_cache = {}

    
    def forward_gpt_energy_cached(self, b_encodings, b_idx_of_interest, b_bid, device='cuda:0', target_toks = 4096, mode = 'norm'):
        # print('cache:', self.energy_cache) 
        b_input_ids: torch.Tensor = b_encodings.input_ids
        bert_device = b_input_ids.device

        # need to convert bert_encodings to gpt encodings here
        input_sent = [' '.join(remove_cls_and_pad(self.tokenizer.decode(input_ids).split(' '))) for input_ids in b_input_ids.unbind(0)]
        gpt_encodings = self.gpt_tokenizer(input_sent,  padding=True, return_tensors="pt").to(bert_device)

        b_bz, b_max_seqlen = gpt_encodings.input_ids.size()
        b_gpt_input_ids = torch.cat([gpt_encodings.input_ids, torch.ones(b_bz, 1, device=device).long()*self.gpt_tokenizer.eos_token_id], dim=1)
        number_of_tokens_before_adding_eos = torch.sum(gpt_encodings.attention_mask, dim=1)
        attention_mask = torch.cat([gpt_encodings.attention_mask, torch.zeros(b_bz, 1, device=device).long()], dim=1).scatter(1, number_of_tokens_before_adding_eos.unsqueeze(1), value=1)
        b_max_seqlen += 1

        # fill cache
        b_input_energy_per_input = torch.zeros(b_bz, device=device)
        # b_bid = torch.arange(b_bz, device=device).unsqueeze(1)
        b_xy_id = b_encodings.input_ids.gather(1, b_idx_of_interest)
        b_bid_xy_id = torch.cat([b_bid, b_xy_id], dim=1)
        b_mask_needs_computation = torch.ones(b_bz, device=device).bool()
        b_cached_energy = []
        energy_cache_keys_from_last_step = set(self.energy_cache.keys())
        for id, bid_xy_id  in enumerate(b_bid_xy_id.unbind(0)):
           bid_xy_id = tuple(bid_xy_id.tolist())
           if bid_xy_id in self.energy_cache.keys():
            #    bid, _, _ = bid_xy_id
               b_mask_needs_computation[id] = False
               b_cached_energy.append(self.energy_cache[bid_xy_id])
        if len(b_cached_energy) > 0:
            b_cached_energy = torch.stack(b_cached_energy, dim=0)
            b_input_energy_per_input[torch.logical_not(b_mask_needs_computation)] = b_cached_energy                
        else:
            assert torch.all(b_mask_needs_computation)
        # print('model: forward GPT energy computation mask', b_mask_needs_computation)


        # actual computation
        if torch.any(b_mask_needs_computation):
            num_tokens_per_input = attention_mask[b_mask_needs_computation].sum(1)
            gather_idx = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(b_bz, 1)[b_mask_needs_computation]
            input_id_indicators = torch.arange(b_bz, device=device)[b_mask_needs_computation].unsqueeze(1).repeat(1, b_max_seqlen)
            gather_idx_mask = torch.logical_and(gather_idx < num_tokens_per_input.unsqueeze(1), gather_idx>0)
            # print('model: forward_norm_energy: gather_idx_mask', gather_idx_mask)
            # print('model: forward_norm_energy: gather_idx', gather_idx)
            # print('model: forward_norm_energy: gather-idx-size: ', gather_idx.size(), input_id_indicators.size(), gather_idx_mask.size(), gather_idx_mask.sum())





            b_samples_input_bid = input_id_indicators[gather_idx_mask].unsqueeze(1)
            b_samples_idx = gather_idx[gather_idx_mask].unsqueeze(1)#b_idx.gather(0, b_samples_input_id_indicators.unsqueeze(1).expand(-1, b_idx_size))


            b_samples_input_ids = b_gpt_input_ids.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
            num_working_inputs = b_samples_input_ids.size(0)
            b_samples_mask_indicator = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(num_working_inputs, 1) >= b_samples_idx
            # b_samples_input_ids_original = b_samples_input_ids.clone()
            # b_samples_input_ids[b_samples_mask_indicator] = self.gpt_tokenizer.pad_token_id
            b_samples_attention_mask = attention_mask.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
            b_samples_attention_mask[b_samples_mask_indicator] = 0
            

            b_samples_encodings = BatchEncoding({
                'input_ids': b_samples_input_ids,
                'attention_mask': b_samples_attention_mask,
            })

            b_samples_output = self.gpt_batched_forward_v2(b_samples_encodings, b_samples_idx-1, b_samples_input_ids.gather(1, b_samples_idx), target_toks=target_toks)

            b_input_energy = torch.zeros_like(b_gpt_input_ids[b_mask_needs_computation], device=device, dtype=torch.float)
            # print('model: forward_norm_energy: target_logprobs.size', -b_samples_output.target_logprobs.view(-1))
            b_input_energy[gather_idx_mask] = -b_samples_output.target_logprobs.view(-1) if mode =='norm' else -b_samples_output.target_logits.view(-1)
            # print('model: forward_gpt_energy: b_input_energy', b_input_energy)
            b_computed_input_energy_per_input = b_input_energy.sum(-1)
            # print('mod3el: forward_norm_energy, b_computed_input_energy_per_input.size:', b_computed_input_energy_per_input.size())
            assert b_computed_input_energy_per_input.size(0) == b_mask_needs_computation.sum() and len(b_computed_input_energy_per_input.size()) == 1
            b_input_energy_per_input[b_mask_needs_computation] = b_computed_input_energy_per_input

            # storing the computed energy to cache
            b_bid_xy_id_needs_computation = b_bid_xy_id[b_mask_needs_computation]
            for bid_xy_id, energy in zip(b_bid_xy_id_needs_computation.unbind(0), b_computed_input_energy_per_input.unbind(0)):
                # print('storing for key:', bid_xy_id.tolist())
                bid_xy_id = tuple(bid_xy_id.tolist())
                assert bid_xy_id not in energy_cache_keys_from_last_step
                self.energy_cache[bid_xy_id] = energy


        return DictObj({
            'energy': b_input_energy_per_input,
            'b_caching_mask': torch.logical_not(b_mask_needs_computation)
        })


    def forward_gpt_energy_cached_v2(self, b_encodings, b_idx_of_interest, b_bid,  device='cuda:0', target_toks = 4096, mode = None, tmp = 0., energy_tmp = 0.):
        def unique(x, dim=0):
            unique, inverse, counts = torch.unique(x, dim=dim, 
                sorted=True, return_inverse=True, return_counts=True)
            inv_sorted = inverse.argsort(stable=True)
            tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
            index = inv_sorted[tot_counts]
            return unique, inverse, counts, index
        b_input_ids: torch.Tensor = b_encodings.input_ids
        bert_device = b_input_ids.device

        b_existing_cache_keys = set(self.energy_cache.keys())

        # need to convert bert_encodings to gpt encodings here
        input_sent = [' '.join(remove_cls_and_pad(self.tokenizer.decode(input_ids).split(' '))) for input_ids in b_input_ids.unbind(0)]
        gpt_encodings = self.gpt_tokenizer(input_sent,  padding=True, return_tensors="pt").to(bert_device)

        b_gpt_input_ids = gpt_encodings.input_ids

        # print('model: forward_gpt_energy: gpt_inputs\n')
        # for sent, gpt_input_ids, attn_mask in zip(input_sent, b_gpt_input_ids.unbind(0), gpt_encodings.attention_mask):
        #     print(sent)
        #     print(self.gpt_tokenizer.convert_ids_to_tokens(gpt_input_ids))
        #     print(attn_mask)

        # print('********end*********')
        b_bz, b_max_seqlen = gpt_encodings.input_ids.size()
        b_gpt_input_ids = torch.cat([gpt_encodings.input_ids, torch.ones(b_bz, 1, device=device).long()*self.gpt_tokenizer.eos_token_id], dim=1)
        number_of_tokens_before_adding_eos = torch.sum(gpt_encodings.attention_mask, dim=1)
        attention_mask = torch.cat([gpt_encodings.attention_mask, torch.zeros(b_bz, 1, device=device).long()], dim=1).scatter(1, number_of_tokens_before_adding_eos.unsqueeze(1), value=1)
        b_max_seqlen += 1

        # print('model:forward gpt energy: gpt attention mask\n', gpt_encodings.attention_mask)
        # print('model:forward gpt energy: gpt with eos attention mask\n', attention_mask)

        # print('model:forward gpt energy: input_ids\n', b_gpt_input_ids)
        # print('model:forward gpt energy: attention_mask\n', /attention_mask)


        num_tokens_per_input = attention_mask.sum(1)
        gather_idx = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(b_bz, 1)
        input_id_indicators = torch.arange(b_bz, device=device).unsqueeze(1).repeat(1, b_max_seqlen)
        gather_idx_mask = torch.logical_and(gather_idx < num_tokens_per_input.unsqueeze(1), gather_idx>0) # here to remove the [SEP] and the [CLS] token

        b_samples_input_bid = input_id_indicators[gather_idx_mask].unsqueeze(1)
        b_samples_idx = gather_idx[gather_idx_mask].unsqueeze(1)#b_idx.gather(0, b_samples_input_id_indicators.unsqueeze(1).expand(-1, b_idx_size))

        b_samples_input_ids = b_gpt_input_ids.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
        num_working_inputs = b_samples_input_ids.size(0)
        b_samples_mask_indicator = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(num_working_inputs, 1) >= b_samples_idx
        b_samples_attention_mask = attention_mask.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
        b_samples_attention_mask[b_samples_mask_indicator] = 0

        b_samples_bz = b_samples_input_ids.size(0)
        cache_keys = torch.cat([b_samples_input_ids, b_samples_attention_mask.sum(-1).unsqueeze(1)], dim=1)
        b_mask_need_compute = torch.zeros(b_samples_bz, device=bert_device).bool()
        b_cached_energy = torch.zeros(b_samples_bz, device=bert_device)

        for id, k in enumerate(cache_keys.unbind(0)):
            # k = tuple(k.tolist())
            if k not in b_existing_cache_keys:#self.energy_cache.keys():
                b_mask_need_compute[id]=True
            else:
                b_cached_energy[id] = self.energy_cache[k]
            
        b_compute_samples_input_ids = b_samples_input_ids[b_mask_need_compute]
        b_compute_samples_attention_mask = b_samples_attention_mask[b_mask_need_compute]

        b_compute_unique_samples_input_ids_attn_masks, b_compute_unique_samples_inverse_indices, _, b_compute_unique_samples_indices= unique(torch.stack([b_compute_samples_input_ids, b_compute_samples_attention_mask], dim=1), dim=0)
        b_compute_unique_samples_input_ids, b_compute_unique_samples_attention_mask = b_compute_unique_samples_input_ids_attn_masks.split(1, dim=1)
        b_compute_unique_samples_input_ids, b_compute_unique_samples_attention_mask = b_compute_unique_samples_input_ids.squeeze(1), b_compute_unique_samples_attention_mask.squeeze(1)
        # print('MTMH: gather_unique_tensors that needs compute: b_input_ids\n', b_compute_unique_samples_input_ids, b_compute_unique_samples_input_ids.size())
        # print('MTMH: gather_unique_tensors that needs compute: b_inverse_indices\n', b_compute_unique_samples_inverse_indices)
        # print('MTMH: gather_unique_tensors that needs compute: b_indices\n', b_compute_unique_samples_indices)
        # print('MTMH: b_sample_idx that needs compute: b_samples_idx\n', b_samples_idx[b_mask_need_compute])
        b_compute_unique_samples_idx = b_samples_idx[b_mask_need_compute].gather(0, b_compute_unique_samples_indices.unsqueeze(1).expand(-1, 1))

        # print('MTMH: gather_unique_tensors that needs compute: b_input_ids\n', b_compute_unique_samples_input_ids)
        # print('MTMH: gather_unique_tensors that needs compute: b_inverse_indices\n', b_compute_unique_samples_inverse_indices)
        # print('MTMH: gather_unique_tensors that needs compute: b_indices\n', b_compute_unique_samples_indices)


        b_samples_encodings = BatchEncoding({
            'input_ids': b_compute_unique_samples_input_ids,
            'attention_mask': b_compute_unique_samples_attention_mask,
            # 'token_type_ids': b_samples_token_type_ids
        })
        # print('model:forward gpt energy: token to grab vs. ref token\n ', torch.cat([b_samples_input_ids.gather(1, b_samples_idx-1), b_samples_input_ids.gather(1, b_samples_idx)], dim=-1))
        # print('model:forward gpt energy: input_ids\n', b_samples_encodings)
        
        # print('model:forward gpt energy: attention_mask\n', attention_mask)

        b_samples_output = self.gpt_batched_forward_v2(b_samples_encodings, b_compute_unique_samples_idx-1, b_compute_unique_samples_input_ids.gather(1, b_compute_unique_samples_idx), target_toks=target_toks)

        b_compute_energy = (-b_samples_output.target_logprobs).view(-1).gather(0, b_compute_unique_samples_inverse_indices)

        b_input_energy = torch.zeros_like(b_gpt_input_ids, device=device, dtype=torch.float)
        b_cached_energy[b_mask_need_compute] = b_compute_energy#-b_samples_output.target_logprobs.view(-1)
        b_input_energy[gather_idx_mask] = b_cached_energy#-b_samples_output.target_logprobs.view(-1)
        # print('model: forward_norm_energy: b_input_energy', b_input_energy)
        b_input_energy_per_input = b_input_energy.sum(-1)

        compute_keys = cache_keys[b_mask_need_compute]

        for key, energy in zip(compute_keys.unbind(0), (-b_samples_output.target_logprobs).view(-1).unbind(0)):
            # assert key not in b_existing_cache_keys
            if key in b_existing_cache_keys: continue
            self.energy_cache[key] = energy


        return DictObj({
            'energy': b_input_energy_per_input
        })


    def forward_gpt_energy(self, b_encodings, b_idx_of_interest, device='cuda:0', target_toks = 4096, mode = None, tmp = 0., energy_tmp = 0.):
        b_input_ids: torch.Tensor = b_encodings.input_ids
        bert_device = b_input_ids.device

        # need to convert bert_encodings to gpt encodings here
        input_sent = [' '.join(remove_cls_and_pad(self.tokenizer.decode(input_ids).split(' '))) for input_ids in b_input_ids.unbind(0)]
        gpt_encodings = self.gpt_tokenizer(input_sent,  padding=True, return_tensors="pt").to(bert_device)

        b_gpt_input_ids = gpt_encodings.input_ids

        # print('model: forward_gpt_energy: gpt_inputs\n')
        # for sent, gpt_input_ids, attn_mask in zip(input_sent, b_gpt_input_ids.unbind(0), gpt_encodings.attention_mask):
        #     print(sent)
        #     print(self.gpt_tokenizer.convert_ids_to_tokens(gpt_input_ids))
        #     print(attn_mask)

        # print('********end*********')
        b_bz, b_max_seqlen = gpt_encodings.input_ids.size()
        b_gpt_input_ids = torch.cat([gpt_encodings.input_ids, torch.ones(b_bz, 1, device=device).long()*self.gpt_tokenizer.eos_token_id], dim=1)
        number_of_tokens_before_adding_eos = torch.sum(gpt_encodings.attention_mask, dim=1)
        attention_mask = torch.cat([gpt_encodings.attention_mask, torch.zeros(b_bz, 1, device=device).long()], dim=1).scatter(1, number_of_tokens_before_adding_eos.unsqueeze(1), value=1)
        b_max_seqlen += 1

        # print('model:forward gpt energy: gpt attention mask\n', gpt_encodings.attention_mask)
        # print('model:forward gpt energy: gpt with eos attention mask\n', attention_mask)

        # print('model:forward gpt energy: input_ids\n', b_gpt_input_ids)
        # print('model:forward gpt energy: attention_mask\n', /attention_mask)


        num_tokens_per_input = attention_mask.sum(1)
        gather_idx = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(b_bz, 1)
        input_id_indicators = torch.arange(b_bz, device=device).unsqueeze(1).repeat(1, b_max_seqlen)
        gather_idx_mask = torch.logical_and(gather_idx < num_tokens_per_input.unsqueeze(1), gather_idx>0) # here to remove the [SEP] and the [CLS] token

        b_samples_input_bid = input_id_indicators[gather_idx_mask].unsqueeze(1)
        b_samples_idx = gather_idx[gather_idx_mask].unsqueeze(1)#b_idx.gather(0, b_samples_input_id_indicators.unsqueeze(1).expand(-1, b_idx_size))

        b_samples_input_ids = b_gpt_input_ids.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
        num_working_inputs = b_samples_input_ids.size(0)
        b_samples_mask_indicator = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(num_working_inputs, 1) >= b_samples_idx
        b_samples_attention_mask = attention_mask.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
        b_samples_attention_mask[b_samples_mask_indicator] = 0

        b_samples_encodings = BatchEncoding({
            'input_ids': b_samples_input_ids,
            'attention_mask': b_samples_attention_mask,
            # 'token_type_ids': b_samples_token_type_ids
        })
        # print('model:forward gpt energy: token to grab vs. ref token\n ', torch.cat([b_samples_input_ids.gather(1, b_samples_idx-1), b_samples_input_ids.gather(1, b_samples_idx)], dim=-1))
        # print('model:forward gpt energy: input_ids\n', b_samples_encodings)
        
        # print('model:forward gpt energy: attention_mask\n', attention_mask)

        b_samples_output = self.gpt_batched_forward_v2(b_samples_encodings, b_samples_idx-1, b_samples_input_ids.gather(1, b_samples_idx), target_toks=target_toks)

        b_input_energy = torch.zeros_like(b_gpt_input_ids, device=device, dtype=torch.float)
        b_input_energy[gather_idx_mask] = -b_samples_output.target_logprobs.view(-1)
        # print('model: forward_norm_energy: b_input_energy', b_input_energy)
        b_input_energy_per_input = b_input_energy.sum(-1)


        return DictObj({
            'energy': b_input_energy_per_input
        })

    def forward_norm_energy(self, b_encodings, device='cuda:0', target_toks = 4096):
        input_ids: torch.Tensor = b_encodings.input_ids
        b_bz, b_max_seqlen = input_ids.size()
        attention_mask = b_encodings.attention_mask
        # b_idx_size = b_idx.size(1)

        num_tokens_per_input = attention_mask.sum(1)
        gather_idx = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(b_bz, 1)
        input_id_indicators = torch.arange(b_bz, device=device).unsqueeze(1).repeat(1, b_max_seqlen)
        # !! gpt's input format may not be the same as bert
        gather_idx_mask = torch.logical_and(gather_idx < num_tokens_per_input.unsqueeze(1), gather_idx>0) # here to remove the [SEP] and the [CLS] token
        # num_compute = torch.sum(gather_idx_mask).int()
        # print('model: forward_norm_energy: gather_idx_mask', gather_idx_mask)
        # print('model: forward_norm_energy: gather_idx', gather_idx)


        b_samples_input_bid = input_id_indicators[gather_idx_mask].unsqueeze(1)
        b_samples_idx = gather_idx[gather_idx_mask].unsqueeze(1)#b_idx.gather(0, b_samples_input_id_indicators.unsqueeze(1).expand(-1, b_idx_size))
        # b_samples_mask_indicator = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(num_compute, 1) >= b_samples_idx
        
        # print('model: forward_norm_energy: b_samples_input_bid_indicators', b_samples_input_bid)
        # print('model: forward_norm_energy: b_samples_idx', b_samples_idx)

        b_samples_input_ids = input_ids.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen)).scatter(1, b_samples_idx, value=self.tokenizer.mask_token_id)#[b_samples_mask_indicator] = self.tokenizer.mask_token_id
        b_samples_attention_mask = attention_mask.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
        b_samples_token_type_ids = b_encodings.token_type_ids.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
        # b_samples_bz = b_samples_input_ids.size(0)

        b_samples_encodings = BatchEncoding({
            'input_ids': b_samples_input_ids,
            'attention_mask': b_samples_attention_mask,
            'token_type_ids': b_samples_token_type_ids
        })

        b_samples_output = self.gpt_batched_forward_v2(b_samples_encodings, b_samples_idx, b_samples_input_ids.gather(1, b_samples_idx), target_toks=target_toks)

        b_input_energy = torch.zeros_like(input_ids, device=device, dtype=torch.float)
        b_input_energy[gather_idx_mask] = -b_samples_output.target_logprobs.view(-1)
        # print('model: forward_norm_energy: b_input_energy', b_input_energy)
        b_input_energy_per_input = b_input_energy.sum(-1)

        return DictObj({
            'energy': b_input_energy_per_input
        })



    def forward_norm_energy_cached(self, b_encodings, b_idx_of_interest, device='cuda:0', target_toks = 4096, mode = 'norm'):
        # print('cache:', self.energy_cache) 
        input_ids: torch.Tensor = b_encodings.input_ids
        b_bz, b_max_seqlen = input_ids.size()
        attention_mask = b_encodings.attention_mask


        # fill cache
        b_input_energy_per_input = torch.zeros(b_bz, device=device)
        b_bid = torch.arange(b_bz, device=device).unsqueeze(1)
        b_xy_id = b_encodings.input_ids.gather(1, b_idx_of_interest)
        b_bid_xy_id = torch.cat([b_bid, b_xy_id], dim=1)
        b_mask_needs_computation = torch.ones(b_bz, device=device).bool()
        b_cached_energy = []
        for bid_xy_id  in b_bid_xy_id.unbind(0):
           bid_xy_id = tuple(bid_xy_id.tolist())
           if bid_xy_id in self.energy_cache.keys():
               bid, _, _ = bid_xy_id
               b_mask_needs_computation[bid] = False
               b_cached_energy.append(self.energy_cache[bid_xy_id])
        if len(b_cached_energy) > 0:
            b_cached_energy = torch.stack(b_cached_energy, dim=0)
            b_input_energy_per_input[torch.logical_not(b_mask_needs_computation)] = b_cached_energy                
        else:
            assert torch.all(b_mask_needs_computation)
        # print('cache ratio:', torch.logical_not(b_mask_needs_computation).sum(0)/b_mask_needs_computation.size(0))



        # actual computation
        if torch.any(b_mask_needs_computation):
            num_tokens_per_input = attention_mask[b_mask_needs_computation].sum(1)
            gather_idx = torch.arange(0, b_max_seqlen, device=device).unsqueeze(0).repeat(b_bz, 1)[b_mask_needs_computation]
            input_id_indicators = torch.arange(b_bz, device=device)[b_mask_needs_computation].unsqueeze(1).repeat(1, b_max_seqlen)
            gather_idx_mask = torch.logical_and(gather_idx < num_tokens_per_input.unsqueeze(1)-1, gather_idx>0) # here to remove the [SEP] and the [CLS] token

            b_samples_input_bid = input_id_indicators[gather_idx_mask].unsqueeze(1)
            b_samples_idx = gather_idx[gather_idx_mask].unsqueeze(1)#b_idx.gather(0, b_samples_input_id_indicators.unsqueeze(1).expand(-1, b_idx_size))
            # b_samples_idx_mask = b_idx_mask.gather(0, b_samples_input_id_indicators.unsqueeze(1).expand(-1, b_idx_size))
            # b_samples_conditional_mask = b_conditional_mask.gather(0, b_samples_input_id_indicators.unsqueeze(1).expand(-1, b_idx_size))

            # print('model: forward_norm_energy: b_samples_input_bid_indicators', b_samples_input_bid.view(-1))
            # print('model: forward_norm_energy: b_samples_idx', b_samples_idx.view(-1))

            b_samples_input_ids = input_ids.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))

            # print('model: forward_energy: b_samples_input_ids', b_samples_input_ids[:15])
            b_samples_attention_mask = attention_mask.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
            b_samples_token_type_ids = b_encodings.token_type_ids.gather(0, b_samples_input_bid.expand(-1, b_max_seqlen))
            # b_samples_bz = b_samples_input_ids.size(0)

            b_samples_encodings = BatchEncoding({
                'input_ids': b_samples_input_ids.scatter(1, b_samples_idx, value=self.tokenizer.mask_token_id),
                'attention_mask': b_samples_attention_mask,
                'token_type_ids': b_samples_token_type_ids
            })

            b_samples_output = self.model_batched_forward_v2(b_samples_encodings, b_samples_idx, b_samples_input_ids.gather(1, b_samples_idx), target_toks=target_toks)
            # b_samples_output_2 = self.model_batched_forward_v2(b_samples_encodings, b_samples_idx, b_samples_input_ids.gather(1, b_samples_idx), target_toks=target_toks)

            # assert torch.all(torch.isclose(b_samples_output.target_logprobs, b_samples_output_2.target_logprobs))

            b_input_energy = torch.zeros_like(input_ids[b_mask_needs_computation], device=device, dtype=torch.float)
            # print('model: forward_norm_energy: target_logprobs.size', -b_samples_output.target_logprobs.view(-1))
            b_input_energy[gather_idx_mask] = -b_samples_output.target_logprobs.view(-1) if mode =='norm' else -b_samples_output.target_logits.view(-1)
            # print('model: forward_norm_energy: b_input_energy', b_input_energy)
            b_computed_input_energy_per_input = b_input_energy.sum(-1)
            # print('mod3el: forward_norm_energy, b_computed_input_energy_per_input.size:', b_computed_input_energy_per_input.size())
            assert b_computed_input_energy_per_input.size(0) == b_mask_needs_computation.sum() and len(b_computed_input_energy_per_input.size()) == 1
            b_input_energy_per_input[b_mask_needs_computation] = b_computed_input_energy_per_input

            # storing the computed energy to cache
            b_bid_xy_id_needs_computation = b_bid_xy_id[b_mask_needs_computation]
            for bid_xy_id, energy in zip(b_bid_xy_id_needs_computation.unbind(0), b_computed_input_energy_per_input.unbind(0)):
                bid_xy_id = tuple(bid_xy_id.tolist())
                assert bid_xy_id not in self.energy_cache.keys()
                self.energy_cache[bid_xy_id] = energy


        return DictObj({
            'energy': b_input_energy_per_input,
            'b_caching_mask': torch.logical_not(b_mask_needs_computation)
        })










    def model_batched_forward(self, b_encodings, target_toks=6128):
        # print('computing through batched_forward')
        if self.flag_debug:
            target_toks = 64
        b_bz, max_seqlen = b_encodings.input_ids.size()
        target_bz = math.floor(target_toks / max_seqlen)
        pt_batch_idx = 0
        output_pool = []
        cnt = 0
        while pt_batch_idx <= b_bz:
            pseudo_b_encodings = BatchEncoding(
                {
                    k: v[pt_batch_idx : pt_batch_idx + target_bz]
                    for k, v in b_encodings.items()
                }
            )
            output_pool.append(
                {k: v.detach() for k, v in self.model(**pseudo_b_encodings).items()}
            )
            # print(output_pool)
            pt_batch_idx += target_bz
            cnt += 1
        output_keys = list(output_pool[0].keys())
        output_dict = MaskedLMOutput(
            **{
                key: torch.cat([item[key] for item in output_pool], dim=0)
                for key in output_keys
            }
        )
        # print('combinging {} outputs'.format(cnt))
        # print(output_dict)
        return output_dict

    def gpt_batched_forward_v2(
        self, b_encodings, gather_idx, gather_target, target_toks=4096
    ):
        original_device = b_encodings.input_ids.device
        if self.flag_debug:
            target_toks = 64
        b_bz, max_seqlen = b_encodings.input_ids.size()
        # print('model: max_seqlen:', max_seqlen)
        target_bz = math.floor(target_toks / max_seqlen)
        pt_batch_idx = 0
        output_logits = []
        cnt = 0
        # assert target_bz > b_bz
        # print(b_encodings.input_ids.size())
        with tqdm(total=b_bz, disable=True) as pbar:
            while pt_batch_idx < b_bz:
                pseudo_b_encodings = BatchEncoding(
                    {
                        k: v[pt_batch_idx : pt_batch_idx + target_bz]
                        for k, v in b_encodings.items()
                    }
                )
                gathered_logits = self.gpt(**pseudo_b_encodings).logits.gather(
                    1,
                    gather_idx[pt_batch_idx : pt_batch_idx + target_bz]
                    .unsqueeze(-1)
                    .expand(-1, -1, self.gpt_vocab_size),
                )
                # move it to cpu to save gpu memory
                output_logits.append(gathered_logits)
                pt_batch_idx += target_bz
                pbar.update(target_bz)
                cnt += 1

        output_logits = torch.cat(output_logits, dim=0)  # (b_bz, idx_size, vocab)
        # print(gather_target.size())
        assert output_logits.size(0) == b_bz
        assert len(gather_target.size()) == 2 and gather_target.size(1) == 1 # gather_target should be(bz, target_size = 1)
        # assert gather_target.size(2) == 1

        vocab_size = output_logits.size(-1)
        output_logdists = torch.log_softmax(output_logits, dim=2)
        target_logprobs = output_logdists.gather(2, gather_target.unsqueeze(2))
        target_logits = output_logits.gather(2, gather_target.unsqueeze(2))
        # smoothed_logdists = torch.log(torch.softmax(output_logits, dim=2) * 0.9 + torch.ones_like(output_logits)*0.1/vocab_size)
        # smoothed_target_logprobs = smoothed_logdists.gather(2, gather_target.unsqueeze(2))
        return DictObj(
            {
                "logits": output_logits,
                "target_logprobs": target_logprobs.to(original_device),
                "target_logits": target_logits,
                "logdists_cpu": output_logdists,
                "logdists_fast": output_logdists.to(original_device),# if b_bz<4096 else None,
                # "smoothed_target_logprobs": smoothed_target_logprobs.to(original_device)
            }
        )
    def model_batched_forward_v2(
        self, b_encodings, gather_idx, gather_target, target_toks=4096
    ):
        original_device = b_encodings.input_ids.device
        # print('computing through batched_forward')
        if self.flag_debug:
            target_toks = 64
        b_bz, max_seqlen = b_encodings.input_ids.size()
        # print('model: max_seqlen:', max_seqlen)
        target_bz = math.floor(target_toks / max_seqlen)
        pt_batch_idx = 0
        output_logits = []
        cnt = 0
        # assert target_bz > b_bz
        # print(b_encodings.input_ids.size())
        with tqdm(total=b_bz, disable=True) as pbar:
            while pt_batch_idx <= b_bz:
                pseudo_b_encodings = BatchEncoding(
                    {
                        k: v[pt_batch_idx : pt_batch_idx + target_bz]
                        for k, v in b_encodings.items()
                    }
                )
                gathered_logits = self.model(**pseudo_b_encodings).logits.gather(
                    1,
                    gather_idx[pt_batch_idx : pt_batch_idx + target_bz]
                    .unsqueeze(-1)
                    .expand(-1, -1, self.vocab_size),
                )
                # move it to cpu to save gpu memory
                output_logits.append(gathered_logits)
                pt_batch_idx += target_bz
                pbar.update(target_bz)
                cnt += 1

        output_logits = torch.cat(output_logits, dim=0)  # (b_bz, idx_size, vocab)
        # print(gather_target.size())
        assert output_logits.size(0) == b_bz
        assert len(gather_target.size()) == 2 and gather_target.size(1) == 1 # gather_target should be(bz, target_size = 1)
        # assert gather_target.size(2) == 1

        vocab_size = output_logits.size(-1)
        output_logdists = torch.log_softmax(output_logits, dim=2)
        target_logprobs = output_logdists.gather(2, gather_target.unsqueeze(2))
        target_logits = output_logits.gather(2, gather_target.unsqueeze(2))
        # smoothed_logdists = torch.log(torch.softmax(output_logits, dim=2) * 0.9 + torch.ones_like(output_logits)*0.1/vocab_size)
        # smoothed_target_logprobs = smoothed_logdists.gather(2, gather_target.unsqueeze(2))
        return DictObj(
            {
                "logits": output_logits,
                "target_logprobs": target_logprobs.to(original_device),
                "target_logits": target_logits,
                "logdists_cpu": output_logdists,
                "logdists_fast": output_logdists.to(original_device),# if b_bz<4096 else None,
                # "smoothed_target_logprobs": smoothed_target_logprobs.to(original_device)
            }
        )

    def forward_pxy_partition_v2_stub(
        self,
        b_x_log_proposal_dists,
        b_encodings,
        b_canonical_x_idx,
        b_canonical_y_idx,
        b_canonical_x_idx_mask,
        b_canonical_y_idx_mask,
        b_canonical_x_conditional_mask,
        b_canonical_y_conditional_mask,
        num_samples,
        device="cuda:0",
        **kwargs
    ):
        """This function computes the partition function of
        \int_{S_{XY}} P_v(x, y|c)dxy = \int_{S_X}q(x|\phi,c)\frac{P(x|\phi, c)}{q(x|\phi, c)}[\int_{S_Y}P_v(y|x, c)dy]dx

        CURRENT ASSUMPTIONS ARE
        1. b_x_size=b_y_size = 1
        1.1. both P_v(x|\phi, c) and q(x|\phi, c) are bert mlm head for a single token

        This function returns a (b_bz) shape tensor


        Args:
            b_x_proposal_dists: tensor for the !!log!! proposal distribution of x (bz, b_x_size=1, logit_size)
            b_encodings (_type_): _description_
            b_canonical_x_idx (_type_): tensor of head indices (bz, b_x_size),
            b_canonical_y_idx (_type_): tensor of dependent indices (bz, b_y_size)
            b_x_idx_mask (_type_): _description_
            b_y_idx_mask (_type_): _description_
            b_x_conditional_mask (_type_): mask tensor that specify the word set S_X: (bz, b_x_size, logit_size)
            b_y_conditional_mask (_type_): mask tensor that specify the word set S_Y: (bz, b_y_size, logit_size)
            num_samples (_type_): number of samples used to estimate the partition function
            device (str, optional): _description_. Defaults to "cuda:0".

        Returns:
            _type_: _description_
        """

        # TODO: current sampling may not do the \int_{S_X} p(x|\phi, c)...

        # b_x_log_proposal_dists = b_x_log_proposal_dists + (torch.logical_not(b_canonical_x_conditional_mask) * -1e9)
        prod_output = self.forward_prod_logprobs(
            b_encodings,
            b_canonical_x_idx,
            b_canonical_x_idx_mask,
            b_canonical_y_idx,
            b_canonical_y_idx_mask,
            device=device,
        )
        b_x_bz, b_x_size, b_x_logit_size = prod_output.b_x_logits.size()
        assert b_x_size == 1
        b_bz = b_x_bz
        b_logit_size = b_x_logit_size
        # sampling - q(x|\phi, c) for

        b_x_log_dists = torch.log_softmax(prod_output.b_x_logits, dim=-1) + (
            torch.logical_not(b_canonical_x_conditional_mask) * -1e9
        )

        b_x_proposal_dists = b_x_log_proposal_dists.exp()  # pseudo probability
        # print('proposal mass', b_x_proposal_dists.sum(-1))
        b_x_samples = torch.multinomial(
            b_x_proposal_dists.view(b_x_bz * b_x_size, b_x_logit_size),
            num_samples=num_samples,
            replacement=True,
        )
        b_x_samples_proposal_logprobs = (
            (
                b_x_log_proposal_dists.gather(
                    2, b_x_samples.view(b_x_bz, b_x_size, num_samples)
                )
                * b_canonical_x_idx_mask.unsqueeze(-1)
            )
            .transpose(1, 2)
            .view(b_x_bz, num_samples, b_x_size)
        )
        b_x_samples_logprobs = (
            (
                b_x_log_dists.gather(2, b_x_samples.view(b_x_bz, b_x_size, num_samples))
                * b_canonical_x_idx_mask.unsqueeze(-1)
            )
            .transpose(1, 2)
            .view(b_x_bz, num_samples, b_x_size)
        )
        # print(b_x_samples_for_pxy_partitions)
        # print( b_x_samples_for_pxy_partitions.view(b_x_bz, b_x_size, num_samples))
        # b_x_probs_for_pxy_partitions = (b_x_dists.gather(2, b_x_samples_for_pxy_partitions.view(b_x_bz, b_x_size, num_samples))*b_x_idx_mask.unsqueeze(-1)).transpose(1, 2).view(b_x_bz*num_samples, b_x_size)
        b_x_samples = (
            b_x_samples.view(b_x_bz, b_x_size, num_samples)
            .transpose(1, 2)
            .view(b_x_bz * num_samples, b_x_size)
        )

        # compute the effective sample size
        b_effective_sample_size = (
            torch.ones(b_bz, device=device)
            * num_samples
            / (torch.logsumexp(b_x_log_proposal_dists, dim=2).squeeze(1).exp())
        )  # again, assuming b_x_size = 1

        pass

        # build the x, y_idx for samples
        b_sample_x_size = b_canonical_x_idx.size(1)
        b_sample_x_idx = b_canonical_x_idx.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_x_size
        )
        b_sample_x_idx_mask = b_canonical_x_idx_mask.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_x_size
        )
        b_sample_y_size = b_canonical_y_idx.size(1)
        b_sample_y_idx = b_canonical_y_idx.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_y_size
        )
        b_sample_y_idx_mask = b_canonical_x_idx_mask.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_y_size
        )
        b_sample_y_conditional_mask = (
            torch.logical_not(b_canonical_y_conditional_mask)
            .repeat(1, num_samples, 1)
            .view(b_bz * num_samples, b_sample_y_size, b_logit_size)
        )  # (b_bz*num_samples, b_y_size, logit_size), for creating the -1e9 mask

        # print(b_sample_x_idx)
        # print(b_sample_y_idx)
        # print(b_sample_y_conditional_mask)

        # building b_sample_encodings
        b_cls_idx = torch.zeros(b_bz * num_samples, 1, device=device).long()
        b_sample_input_ids = (
            b_encodings.input_ids.clone()
            .repeat(1, num_samples)
            .view(b_bz * num_samples, -1)
            .scatter(1, b_sample_x_idx, src=b_x_samples)
            .scatter(1, b_cls_idx, value=self.tokenizer.cls_token_id)
        )
        # print(b_encodings.input_ids.clone().repeat(1, num_samples))
        # print(b_encodings.input_ids.clone().repeat(1, num_samples).view(b_bz*num_samples, -1))
        # print(b_sample_input_ids)
        b_sample_attention_mask = (
            b_encodings.attention_mask.clone()
            .repeat(1, num_samples)
            .view(b_bz * num_samples, -1)
        )
        b_sample_token_type_ids = (
            b_encodings.token_type_ids.clone()
            .repeat(1, num_samples)
            .view(b_bz * num_samples, -1)
        )
        b_sample_encodings = BatchEncoding(
            {
                "input_ids": b_sample_input_ids,
                "attention_mask": b_sample_attention_mask,
                "token_type_ids": b_sample_token_type_ids,
            }
        )
        b_sample_y_outputs = self.forward_lr_logprobs_stub(
            b_sample_encodings,
            b_sample_y_idx,
            b_sample_y_idx_mask,
            b_sample_encodings.input_ids.gather(1, b_sample_y_idx),
            device=device,
        )

        # print(b_sample_y_outputs)

        b_sample_y_log_dists = torch.log_softmax(
            b_sample_y_outputs.logits, dim=-1
        )  # (bz*num_samples, b_y_size, logit_size)
        # b_y_conditional_mask = torch.logical_not(b_y_conditional_mask).repeat(1, num_samples, 1).view(b_x_bz*num_samples, b_sample_y_size).float() * -9999
        b_sample_log_y_integral = torch.logsumexp(
            b_sample_y_log_dists + b_sample_y_conditional_mask * -1e9, dim=2
        )
        assert b_sample_log_y_integral.size(1) == 1
        # print((b_x_samples_logprobs - b_x_samples_proposal_logprobs).squeeze(2))
        is_ratio = (
            (b_x_samples_logprobs - b_x_samples_proposal_logprobs).exp().squeeze(2)
        )  # (b_bz, num_samples)
        print("is ratio - pxy partition", torch.std_mean(is_ratio, dim=-1))
        partition = (
            b_sample_log_y_integral.exp().view(b_bz, num_samples) * is_ratio
        ).sum(
            -1
        ) / b_effective_sample_size  # (b_bz)
        # print('pxy partition prob', (b_sample_log_y_integral.exp().view(b_bz, num_samples) * is_ratio).sum(-1), b_effective_sample_size.size())
        # print('pxy partition prob', partition.size(), b_bz)
        assert partition.size(0) == b_bz and len(partition.size()) == 1
        return DictObj(
            {
                "partition": partition,
                "b_sample_integral": b_sample_log_y_integral.view(b_bz, num_samples),
            }
        )

    def forward_pxy_partition_v2(
        self,
        b_log_proposal_dists,
        b_encodings,
        b_x_idx,
        b_y_idx,
        b_x_idx_mask,
        b_y_idx_mask,
        b_x_conditional_mask,
        b_y_conditional_mask,
        num_samples,
        device="cuda:0",
        **kwargs
    ):
        forward_output = self.forward_pxy_partition_v2_stub(
            b_log_proposal_dists["x"],
            b_encodings,
            b_x_idx,
            b_y_idx,
            b_x_idx_mask,
            b_y_idx_mask,
            b_x_conditional_mask,
            b_y_conditional_mask,
            num_samples,
            device,
        )
        backward_output = self.forward_pxy_partition_v2_stub(
            b_log_proposal_dists["y"],
            b_encodings,
            b_y_idx,
            b_x_idx,
            b_y_idx_mask,
            b_x_idx_mask,
            b_y_conditional_mask,
            b_x_conditional_mask,
            num_samples,
            device,
        )
        partition = (forward_output.partition + backward_output.partition) / 2
        # b_sample_integral =
        return DictObj({"partition": partition})

    def forward_pxy_partition_stub(
        self,
        b_encodings,
        b_canonical_x_idx,
        b_canonical_y_idx,
        b_canonical_x_idx_mask,
        b_canonical_y_idx_mask,
        b_canonical_x_conditional_mask,
        b_canonical_y_conditional_mask,
        num_samples,
        device="cuda:0",
        **kwargs
    ):
        """This function computes the partition function of
        \int_{S_{XY}} P_v(x, y|c)dxy = \int_{S_X}q(x|\phi,c)\frac{P(x|\phi, c)}{q(x|\phi, c)}[\int_{S_Y}P_v(y|x, c)dy]dx

        CURRENT ASSUMPTIONS ARE
        1. b_x_size=b_y_size = 1
        1.1. both P_v(x|\phi, c) and q(x|\phi, c) are bert mlm head for a single token

        This function returns a (b_bz) shape tensor


        Args:
            b_encodings (_type_): _description_
            b_canonical_x_idx (_type_): tensor of head indices (bz, b_x_size),
            b_canonical_y_idx (_type_): tensor of dependent indices (bz, b_y_size)
            b_x_idx_mask (_type_): _description_
            b_y_idx_mask (_type_): _description_
            b_x_conditional_mask (_type_): mask tensor that specify the word set S_X: (bz, b_x_size, logit_size)
            b_y_conditional_mask (_type_): mask tensor that specify the word set S_Y: (bz, b_y_size, logit_size)
            num_samples (_type_): number of samples used to estimate the partition function
            device (str, optional): _description_. Defaults to "cuda:0".

        Returns:
            _type_: _description_
        """

        # TODO: current sampling may not do the \int_{S_X} p(x|\phi, c)...
        # !!Current implementation in fact does the \int p(x|\phi, c)/\int_{S_X} p(x|\phi, c)
        prod_output = self.forward_prod_logprobs_v2(
            b_encodings,
            b_canonical_x_idx,
            b_canonical_x_idx_mask,
            b_canonical_y_idx,
            b_canonical_y_idx_mask,
            device=device,
        )
        b_x_bz, b_x_size, b_x_logit_size = prod_output.b_x_logits.size()
        b_bz = b_x_bz
        b_logit_size = b_x_logit_size
        # sampling - q(x|\phi, c) for
        b_x_dists = (
            torch.softmax(prod_output.b_x_logits, dim=-1)
            * b_canonical_x_conditional_mask
        )  # pseudo probability
        b_x_samples_for_pxy_partitions = torch.multinomial(
            b_x_dists.view(b_x_bz * b_x_size, b_x_logit_size),
            num_samples=num_samples,
            replacement=True,
        )
        # print(b_x_samples_for_pxy_partitions)
        # print( b_x_samples_for_pxy_partitions.view(b_x_bz, b_x_size, num_samples))
        # b_x_probs_for_pxy_partitions = (b_x_dists.gather(2, b_x_samples_for_pxy_partitions.view(b_x_bz, b_x_size, num_samples))*b_x_idx_mask.unsqueeze(-1)).transpose(1, 2).view(b_x_bz*num_samples, b_x_size)
        b_x_samples_for_pxy_partitions = (
            b_x_samples_for_pxy_partitions.view(b_x_bz, b_x_size, num_samples)
            .transpose(1, 2)
            .view(b_x_bz * num_samples, b_x_size)
        )

        # compute the effective sample size
        b_x_log_dists = torch.log_softmax(prod_output.b_x_logits, dim=2) + (
            torch.logical_not(b_canonical_x_conditional_mask) * -1e9
        )
        b_effective_sample_size = (
            torch.ones(b_bz, device=device)
            * num_samples
            / (torch.logsumexp(b_x_log_dists, dim=2).squeeze(1).exp())
        )  # again, assuming b_x_size = 1
        # print((torch.ones(b_bz) * num_samples).size(), (torch.logsumexp(b_x_log_dists, dim=2).squeeze(1).exp()).size())
        # print('effective sample size', b_effective_sample_size )
        # print('effective sample size assertion flag', b_effective_sample_size+1e-4>=num_samples)
        # assert torch.all(b_effective_sample_size+1e-4>=num_samples)

        # print(b_x_samples_for_pxy_partitions)
        # print(b_bz, b_x_size, b_x_logit_size)

        pass

        # build the x, y_idx for samples
        b_sample_x_size = b_canonical_x_idx.size(1)
        b_sample_x_idx = b_canonical_x_idx.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_x_size
        )
        b_sample_x_idx_mask = b_canonical_x_idx_mask.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_x_size
        )
        b_sample_y_size = b_canonical_y_idx.size(1)
        b_sample_y_idx = b_canonical_y_idx.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_y_size
        )
        b_sample_y_idx_mask = b_canonical_x_idx_mask.repeat(1, num_samples).view(
            b_bz * num_samples, b_sample_y_size
        )
        b_sample_y_conditional_mask = (
            torch.logical_not(b_canonical_y_conditional_mask)
            .repeat(1, num_samples, 1)
            .view(b_bz * num_samples, b_sample_y_size, b_logit_size)
        )  # (b_bz*num_samples, b_y_size, logit_size), for creating the -1e9 mask

        # print(b_sample_x_idx)
        # print(b_sample_y_idx)
        # print(b_sample_y_conditional_mask)

        # building b_sample_encodings
        b_cls_idx = torch.zeros(b_bz * num_samples, 1, device=device).long()
        b_sample_input_ids = (
            b_encodings.input_ids.clone()
            .repeat(1, num_samples)
            .view(b_bz * num_samples, -1)
            .scatter(1, b_sample_x_idx, src=b_x_samples_for_pxy_partitions)
            .scatter(1, b_cls_idx, value=self.tokenizer.cls_token_id)
        )
        # print(b_encodings.input_ids.clone().repeat(1, num_samples))
        # print(b_encodings.input_ids.clone().repeat(1, num_samples).view(b_bz*num_samples, -1))
        # print(b_sample_input_ids)
        b_sample_attention_mask = (
            b_encodings.attention_mask.clone()
            .repeat(1, num_samples)
            .view(b_bz * num_samples, -1)
        )
        b_sample_token_type_ids = (
            b_encodings.token_type_ids.clone()
            .repeat(1, num_samples)
            .view(b_bz * num_samples, -1)
        )
        b_sample_encodings = BatchEncoding(
            {
                "input_ids": b_sample_input_ids,
                "attention_mask": b_sample_attention_mask,
                "token_type_ids": b_sample_token_type_ids,
            }
        )
        b_sample_y_outputs = self.forward_lr_logprobs(
            b_sample_encodings,
            b_sample_y_idx,
            b_sample_y_idx_mask,
            b_sample_encodings.input_ids.gather(1, b_sample_y_idx),
            device=device,
        )

        # print(b_sample_y_outputs)

        b_sample_y_log_dists = torch.log_softmax(
            b_sample_y_outputs.logits, dim=-1
        )  # (bz*num_samples, b_y_size, logit_size)
        # b_y_conditional_mask = torch.logical_not(b_y_conditional_mask).repeat(1, num_samples, 1).view(b_x_bz*num_samples, b_sample_y_size).float() * -9999
        b_sample_integral = torch.logsumexp(
            b_sample_y_log_dists + b_sample_y_conditional_mask * -1e9, dim=2
        )
        assert b_sample_integral.size(1) == 1
        partition = (
            b_sample_integral.exp().view(b_bz, num_samples).sum(-1)
            / b_effective_sample_size
        )  # (b_bz)
        return DictObj(
            {
                "partition": partition,
                "b_sample_integral": b_sample_integral.view(b_bz, num_samples),
            }
        )

    def forward_vinfo(
        self,
        b_unique_samples_bert_pack,
        # b_encodings,
        # b_x_idx,
        # b_y_idx,
        # b_x_idx_mask,
        # b_y_idx_mask,
        # b_y_conditional_mask,
        device="cuda:0",
    ):
        """
        Computing pw-vinfo on $$(x, y)$$ samples, only be called after sampling

        Args:
            b_encodings (_type_): _description_
            b_x_idx (_type_): _description_
            b_y_idx (_type_): _description_
            b_x_idx_mask (_type_): _description_
            b_y_idx_mask (_type_): _description_
            device (str, optional): _description_. Defaults to 'cuda:0'.

        Raises:
            NotImplemented: _description_
        """
        b_encodings = b_unique_samples_bert_pack.b_encodings
        b_y_idx = b_unique_samples_bert_pack.b_y_idx
        b_x_idx = b_unique_samples_bert_pack.b_x_idx
        # b_x_idx_mask = b_unique_samples_bert_pack.b_x_idx_mask
        b_y_idx_mask = b_unique_samples_bert_pack.b_y_idx_mask
        b_y_conditional_mask = b_unique_samples_bert_pack.b_y_conditional_mask

        batch_size = b_x_idx.size(0)
        cls_idx = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        b_encodings_null = b_encodings.copy()
        b_encodings_informed = b_encodings.copy()
        b_encodings_null["input_ids"] = (
            b_encodings_null["input_ids"]
            .scatter(1, b_x_idx, value=self.tokenizer.mask_token_id)
            .scatter(1, cls_idx, value=self.tokenizer.cls_token_id)
        )
        # b_encodings_null['input_ids'].scatter_(1, cls_idx, value = self.tokenizer.cls_token_id)
        b_y_idx_target = b_encodings["input_ids"].gather(1, b_y_idx)
        null_output = self.model_batched_forward_v2(
            b_encodings_null, b_y_idx, b_y_idx_target
        )
        # print('computing informed output')
        informed_output = self.model_batched_forward_v2(
            b_encodings_informed, b_y_idx, b_y_idx_target
        )

        b_y_conditional_mask = torch.logical_not(
            b_y_conditional_mask
        )  # converts b_y_conditional_mask for -1e9 masks
        informed_log_partition = torch.logsumexp(
            # torch.log_softmax(informed_output.logits, dim=2)
            informed_output.logdists_fast
            + b_y_conditional_mask * -1e9,
            dim=2,
        ).to(device)
        null_log_partition = torch.logsumexp(
            # torch.log_softmax(null_output.logits, dim=2) 
            null_output.logdists_fast
            + b_y_conditional_mask * -1e9,
            dim=2,
        ).to(device)

        assert informed_output.target_logprobs.size(1) == 1
        assert null_output.target_logprobs.size(1) == 1
        pw_vinfo = (informed_output.target_logprobs.sum(
            -1
        ) - null_output.target_logprobs.sum(
            -1
        )).squeeze(1)  # (bsize,!b_x_size, !logit_size)

        # print(pw_vinfo.size(), null_log_partition.size())
        assert null_log_partition.size(1) == 1  # only allows single-token words
        # print('log_partitions', null_log_partition, informed_log_partition)

        conditional_pwvinfo = (
            pw_vinfo + null_log_partition.squeeze(1) - informed_log_partition.squeeze(1)
        )

        return DictObj(
            {
                "pw_vinfo": pw_vinfo,
                "conditional_pw_vinfo": conditional_pwvinfo,
                "informed_log_partition": informed_log_partition,
                "null_log_partition": null_log_partition,
                "informed_output": informed_output,
                "null_output": null_output,
            }
        )

    def forward_gibbs_logprobs(
        self,
        b_unique_samples_bert_pack,
        num_samples=4,
        device="cuda:0",
    ):
        # b_x_idx: (bz*num_samples, x_size)
        b_encodings = b_unique_samples_bert_pack.b_encodings
        b_y_idx = b_unique_samples_bert_pack.b_y_idx
        b_x_idx = b_unique_samples_bert_pack.b_x_idx
        assert b_x_idx.size(1) == 1 
        b_x_idx_mask = torch.ones_like(b_x_idx, device=b_y_idx.device).bool()
        b_y_idx_mask = b_x_idx_mask



        output_y_given_x = self.forward_lr_logprobs_batched(
            b_encodings,
            b_y_idx,
            b_y_idx_mask,
            b_encodings.input_ids.gather(1, b_y_idx),
            device=device,
        ) 
        # !! output_y_given_x resides at cpu because of batched requirement

        b_bz, max_seqlen = b_encodings.input_ids.size()
        b_y_size = b_y_idx.size(1)
        b_x_size = b_x_idx.size(1)
        assert b_x_size == 1 and b_y_size == 1
        assert output_y_given_x.logdists_cpu.size(1) == 1
        # assert 
        b_dists_y_given_x = output_y_given_x.logdists_fast.exp().squeeze(
            1
        )  # giving a (bz, vocab_size)
        b_y_samples = torch.multinomial(
            b_dists_y_given_x, num_samples, replacement=True
        ).view(b_bz * num_samples, 1)
        b_y_samples = b_y_samples#.to(device)  # moving the samples back to gpu
        #output_y_given_x.logdists: (b_bz, y_size, vocab_size)
        # b_y_logprobs = output_y_given_x.logdists_fast.gather(2, b_y_samples.view(b_bz, 1, num_samples)).transpose(1,2).view(b_bz*num_samples, 1)

        b_samples_batch_id = (
            torch.arange(b_bz, device=device)
            .unsqueeze(-1)
            .repeat(1, num_samples)
            .view(b_bz * num_samples, 1)
        )
        b_samples_tuple_bid_y = torch.cat([b_samples_batch_id, b_y_samples], dim=1)

        b_unique_samples_tuple_bid_y, b_unique_samples_inverse_indices = torch.unique(
            b_samples_tuple_bid_y, dim=0, return_inverse=True
        )
        b_unique_samples_bid, b_unique_samples_y = b_unique_samples_tuple_bid_y.unbind(
            1
        )
        # b_unisque_samples_size = b_unique_samples_tuple_bid_y.size(0)
        b_unique_samples_bid, b_unique_samples_y = b_unique_samples_bid.unsqueeze(1), b_unique_samples_y.unsqueeze(1)
        # assert len(b_unique_samples_tuple_)
        # print(b_unique_samples_tuple_bid_y.size())
        assert b_unique_samples_tuple_bid_y.size(1) == 2
        assert len(b_unique_samples_tuple_bid_y.size()) == 2
        # print(b_x_idx.size())
        # print(b_unique_samples_bid.size())
        # print(b_unique_samples_bid.view(-1))
        b_unique_samples_x_idx = b_x_idx.gather(
            0, b_unique_samples_bid.expand(-1, 1)
        )
        b_unique_samples_y_idx = b_y_idx.gather(
            0, b_unique_samples_bid.expand(-1, 1)
        )
        b_unique_samples_input_ids = b_encodings.input_ids.gather(
            0, b_unique_samples_bid.expand(-1, max_seqlen)
        ).scatter(1, b_unique_samples_y_idx, src=b_unique_samples_y)
        b_unique_samples_attention_mask = b_encodings.attention_mask.gather(
            0, b_unique_samples_bid.expand(-1, max_seqlen)
        )
        b_unique_samples_token_type_ids = b_encodings.token_type_ids.gather(
            0, b_unique_samples_bid.expand(-1, max_seqlen)
        )

        # b_samples_y_idx = b_y_idx.repeat(1, num_samples).view(b_bz*num_samples, b_y_size)
        # b_samples_x_idx = b_x_idx.repeat(1, num_samples).view(b_bz*num_samples, b_x_size)
        # b_samples_x_idx_mask = b_x_idx_mask.repeat(1, num_samples).view(b_bz*num_samples, b_x_size)
        # # TODO: double check if the x token has been modified
        # b_encodings_samples_x_given_y_input_ids = b_encodings.input_ids.clone().repeat(1, num_samples).view(b_bz*num_samples, max_seqlen).scatter(1, b_samples_y_idx, src=b_y_samples)
        # b_encodings_samples_x_given_y_attention_mask = b_encodings.attention_mask.clone().repeat(1, num_samples).view(b_bz* num_samples, max_seqlen)
        # b_encodings_samples_x_given_y_token_type_ids = b_encodings.token_type_ids.clone().repeat(1, num_samples).view(b_bz* num_samples, max_seqlen)
        b_unique_samples_encodings = BatchEncoding(
            {
                "input_ids": b_unique_samples_input_ids,
                "attention_mask": b_unique_samples_attention_mask,
                "token_type_ids": b_unique_samples_token_type_ids,
            }
        )
        # print(b_encodings_samples_x_given_y.input_ids.size(), b_samples_x_idx.size(), b_encodings_samples_x_given_y.token_type_ids.size())
        # print(b_encodings_samples_x_given_y_input_ids.gather(1, b_samples_x_idx))

        # output_samples_x_given_y = self.forward_lr_logprobs_stub(b_encodings_samples_x_given_y, b_samples_x_idx, b_samples_x_idx_mask, b_encodings_samples_x_given_y_input_ids.gather(1, b_samples_x_idx), device=device)
        # print('gibbs logprobs', b_encodings_samples_x_given_y_input_ids.size())
        output_unique_samples_x_given_y = self.model_batched_forward_v2(
            b_unique_samples_encodings,
            b_unique_samples_x_idx,
            b_unique_samples_input_ids.gather(1, b_unique_samples_x_idx),
        )

        # check the indices
        # print(b_unique_samples_inverse_indices)
        # print(b_unique_samples_inverse_indices.size())
        # print(output_unique_samples_x_given_y.target_logprobs.size())
        # return


        # unpack the unique_samples to samples, and do the computation
        output_samples_x_given_y_target_logprobs = (
            output_unique_samples_x_given_y.smoothed_target_logprobs.gather(
                0, b_unique_samples_inverse_indices.view(-1, 1, 1)
            )
        )

        b_samples_x_given_y_logprobs = output_samples_x_given_y_target_logprobs#output_samples_x_given_y.target_logprobs
        # assert b_samples_x_given_y_logprobs.size(0) == b_bz*num_samples
        # print("gibbs logprob: p(x|y) estimation", b_samples_x_given_y_logprobs)
        b_samples_x_given_y_logprobs = b_samples_x_given_y_logprobs.view(
            b_bz, num_samples
        )
        # print('gibbs logpxy b_sample_x_given_y_logprobs', b_samples_x_given_y_logprobs)

        # MC sampling
        logpxy_gibbs_divisor = torch.logsumexp(
            -b_samples_x_given_y_logprobs, dim=1
        ) - torch.log(
            torch.ones(b_bz, device=b_samples_x_given_y_logprobs.device) * num_samples
        )

        # print('gibbs x|y logprobs', b_samples_x_given_y_logprobs)
        # print('gibbs logpxy divisor', logpxy_gibbs_divisor)
        logpxy_gibbs_dividend = output_y_given_x.logprobs_target.view(b_bz)
        # print('gibbs logpxy divident', logpxy_gibbs_dividend)
        logpxy = logpxy_gibbs_dividend - logpxy_gibbs_divisor
        # print('gibbs pxy || py|x', logpxy.exp(), logpxy_gibbs_dividend.exp(), (logpxy-logpxy_gibbs_dividend).exp())


        # unique_sample counterpart
        # b_unique_samples_x_given_y_target_logprobs = output_unique_samples_x_given_y.target_logprobs
        # b_unique_samples_





        return DictObj(
            {"logprobs": logpxy}#, "output_samples_x_given_y": output_samples_x_given_y}
        )

    def forward_lr_logprobs(
        self, b_encodings, b_xy_idx, b_xy_idx_mask, b_xy_idx_target, device="cuda:0"
    ):
        """This is the api for computing the lr-prob

        Args:
            b_encodings (_type_): _description_
            b_xy_idx (_type_): _description_
            b_xy_idx_mask (_type_): _description_
            b_xy_idx_target (_type_): _description_
            device (str, optional): _description_. Defaults to "cuda:0".
        """
        # print(b_xy_idx.size())
        assert len(b_xy_idx.size()) == 2 and b_xy_idx.size(1) == 2
        forward_output = self.forward_lr_logprobs_stub(
            b_encodings, b_xy_idx, b_xy_idx_mask, b_xy_idx_target, device
        )
        backward_output = self.forward_lr_logprobs_stub(
            b_encodings,
            b_xy_idx.flip([1]),
            b_xy_idx_mask.flip([1]),
            b_xy_idx_target.flip([1]),
            device,
        )
        assert forward_output.logprobs_target.size(1) == 2
        logprobs_target = (
            forward_output.logprobs_target + backward_output.logprobs_target
        )
        return DictObj({"logprobs_target": logprobs_target})

    def forward_lr_logprobs_batched(
        self, b_encodings, b_idx, b_idx_mask, b_target, device="cuda:0"
    ):
        """_summary_

        Args:
            b_encodings (_type_): _description_
            b_idx (_type_): _description_
            b_idx_mask (_type_): _description_
            device (str, optional): _description_. Defaults to 'cuda:0'.

        Raises:
            NotImplemented: _description_
        """

        batch_size, xy_idx_size = b_idx.size()
        assert xy_idx_size == 1
        max_seqlen = b_encodings.input_ids.size(1)
        cls_idx = torch.zeros(
            batch_size * xy_idx_size, 1, dtype=torch.int64, device=device
        )
        # print(b_encodings['input_ids'])
        # building lr-masked-input sequences
        masked_input_ids = [
            b_encodings["input_ids"].scatter(
                1, b_idx[:, mask_tok_idx:], value=self.tokenizer.mask_token_id
            )
            for mask_tok_idx in range(xy_idx_size)
        ]
        # stack them at dim1 -> make (bz, lr_step, seqlen) pack
        masked_input_ids = torch.stack(
            masked_input_ids, dim=1
        ).view(  # -> (bsize*y_idx_size, max_seq_size)
            batch_size * xy_idx_size, max_seqlen
        )
        masked_input_ids = masked_input_ids.scatter(
            1, cls_idx, value=self.tokenizer.cls_token_id
        )

        # print(masked_input_ids)

        token_type_ids = (
            b_encodings["token_type_ids"]
            .repeat(1, xy_idx_size)
            .view(batch_size * xy_idx_size, max_seqlen)
        )
        attention_mask = (
            b_encodings["attention_mask"]
            .repeat(1, xy_idx_size)
            .view(batch_size * xy_idx_size, max_seqlen)
        )

        # print(attention_mask)

        # xy_logits_raw = self.model(
        #     **{
        #         **b_encodings,
        #         "input_ids": masked_input_ids,
        #         "token_type_ids": token_type_ids,
        #         "attention_mask": attention_mask,
        #     }
        # )["logits"]
        xy_logits_b_encodings = BatchEncoding(
            {
                "input_ids": masked_input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }
        )
        xy_logit_outputs = self.model_batched_forward_v2(
            xy_logits_b_encodings,
            b_idx.view(batch_size * xy_idx_size, 1),
            b_target,
        )
        # xy_logit_size = xy_logits_raw.size(-1)

        # xy_logits = xy_logits_outputs.#xy_logits_raw.gather(1, b_xy_idx.view(batch_size*xy_idx_size, 1, 1).expand(-1, -1, xy_logit_size))
        # xy_logits = xy_logits.view(batch_size, xy_idx_size, xy_logit_size)
        # print(xy_logits)

        # TODO: No masking for xy_logits
        # xy_logits*=b_xy_idx_mask.unsqueeze(-1)
        # print(xy_logits)

        # xy_logprobs_target = (
        #     torch.log_softmax(xy_logits, dim=-1)
        #     .gather(2, b_xy_idx_target.unsqueeze(-1))
        #     .squeeze(-1)
        # )

        return DictObj(
            {
                "logits": xy_logit_outputs.logits,  # xy_logits,
                "logdists_fast": xy_logit_outputs.logdists_fast,  # torch.log_softmax(xy_logits, dim=2),
                "logdists_cpu": xy_logit_outputs.logdists_cpu,
                "mask": b_idx_mask,
                "logprobs_target": xy_logit_outputs.target_logprobs.view(
                    batch_size, xy_idx_size
                ),  # xy_logprobs_target,
            }
        )

    def forward_prod_logprobs(
        self,
        b_encodings: BatchEncoding,
        b_x_idx: torch.Tensor,
        b_x_idx_mask: torch.Tensor,
        b_y_idx: torch.Tensor,
        b_y_idx_mask: torch.Tensor,
        device="cuda:0",
    ):
        """compute the proposal distribution given by the product of probs on each position

        Args:
            b_encodings (BatchEncoding): _description_
            b_xy_idx (torch.Tensor): tensor of shape (bsize, max_idx_length) with 0 padding
            b_xy_idx_mask (torch.Tensor): tensor of shape (bsize, max_idx_length), serving as a mask to b_xy_idx
            device (str, optional): _description_. Defaults to 'cuda:0'.

        Returns:
            DictObj: _description_
        """
        batch_size = b_x_idx.size(0)
        cls_idx = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        masked_input_ids = (
            b_encodings.input_ids
            .scatter(1, b_x_idx, value=self.tokenizer.mask_token_id)
            .scatter(1, b_y_idx, value=self.tokenizer.mask_token_id)
            .scatter(1, cls_idx, value=self.tokenizer.cls_token_id)
        )
        logits = self.model(**{**b_encodings, "input_ids": masked_input_ids})["logits"]
        logits_size = logits.size(-1)
        b_x_logits = logits.gather(1, b_x_idx.unsqueeze(-1).expand(-1, -1, logits_size))
        b_x_logits *= b_x_idx_mask.unsqueeze(-1)
        b_y_logits = logits.gather(1, b_y_idx.unsqueeze(-1).expand(-1, -1, logits_size))
        b_y_logits *= b_y_idx_mask.unsqueeze(-1)
        return DictObj(
            {
                "b_x_logits": b_x_logits,
                "b_x_mask": b_x_idx_mask,
                "b_y_logits": b_y_logits,
                "b_y_mask": b_y_idx_mask,
            }
        )

    def forward_prod_logprobs_target_word_only(
        self,
        b_encodings: BatchEncoding,
        b_x_idx: torch.Tensor,
        b_x_idx_mask: torch.Tensor,
        b_y_idx: torch.Tensor,
        b_y_idx_mask: torch.Tensor,
        device="cuda:0",
    ):
        """compute the proposal distribution given by the product of probs on each position

        Args:
            b_encodings (BatchEncoding): _description_
            b_xy_idx (torch.Tensor): tensor of shape (bsize, max_idx_length) with 0 padding
            b_xy_idx_mask (torch.Tensor): tensor of shape (bsize, max_idx_length), serving as a mask to b_xy_idx
            device (str, optional): _description_. Defaults to 'cuda:0'.

        Returns:
            DictObj: _description_
        """
        batch_size = b_x_idx.size(0)
        cls_idx = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        masked_y_input_ids = (
            b_encodings.input_ids
            # .scatter(1, b_x_idx, value=self.tokenizer.mask_token_id)
            .scatter(1, b_y_idx, value=self.tokenizer.mask_token_id)
            .scatter(1, cls_idx, value=self.tokenizer.cls_token_id)
        )
        y_logits = self.model(**{**b_encodings, "input_ids": masked_y_input_ids})["logits"]
        y_logits_size = y_logits.size(2)
        b_y_logits = y_logits.gather(1, b_y_idx.unsqueeze(-1).expand(-1, -1, y_logits_size))
        b_y_logits *= b_y_idx_mask.unsqueeze(-1)
        
        masked_x_input_ids = (
            b_encodings.input_ids
            .scatter(1, b_x_idx, value=self.tokenizer.mask_token_id)
            # .scatter(1, b_y_idx, value=self.tokenizer.mask_token_id)
            .scatter(1, cls_idx, value=self.tokenizer.cls_token_id)
        )
        x_logits = self.model(**{**b_encodings, "input_ids": masked_x_input_ids})["logits"]
        x_logits_size = x_logits.size(-1)
        b_x_logits = x_logits.gather(1, b_x_idx.unsqueeze(-1).expand(-1, -1, x_logits_size))
        b_x_logits *= b_x_idx_mask.unsqueeze(-1)
        return DictObj(
            {
                "b_x_logits": b_x_logits,
                "b_x_mask": b_x_idx_mask,
                "b_y_logits": b_y_logits,
                "b_y_mask": b_y_idx_mask,
            }
        )



class VinfoInformedWrapper(nn.Module):
    def __init__(self, model_name, tokenizer) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            config=config,
        )
        self.model.resize_token_embeddings(len(tokenizer))
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

    def forward(
        self,
        x_idx,
        y_idx,
        b_encodings,
        device="cuda:0",
        logit_compute_strategy="mlm",
        context_mask_probability=0.0,
        y_neighbor_idx=None,
    ):
        batch_size = x_idx.size(0)
        # seq_size = b_encodings[]
        y_idx_size = y_idx.size(1)
        max_seq_size = b_encodings["input_ids"].size(1)
        idx_mask_idx = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)

        # random_mask = torch.bernoulli((context_mask_probability * torch.ones(batch_size, max_seq_size, device=device)).scatter(1, y_idx, value=0).scatter(1, x_idx, value=0).scatter(1, idx_mask_idx, value=0))

        if logit_compute_strategy == "mlm":
            masked_input_ids = b_encodings[
                "input_ids"
            ]  # .scatter(1, y_idx, value=self.tokenizer.mask_token_id)
            masked_input_ids = masked_input_ids.scatter(
                1,
                y_neighbor_idx if y_neighbor_idx is not None else y_idx,
                value=self.tokenizer.mask_token_id,
            )
            masked_input_ids = masked_input_ids.scatter(
                1, idx_mask_idx, value=self.tokenizer.cls_token_id
            )
            true_ys = b_encodings["input_ids"].gather(1, y_idx)
            y_logits = self.model(**{**b_encodings, "input_ids": masked_input_ids})[
                "logits"
            ]
            y_logits_size = y_logits.size(-1)
            y_logits = y_logits.gather(
                1, y_idx.unsqueeze(-1).expand(-1, -1, y_logits_size)
            )
        elif logit_compute_strategy == "clm":
            true_ys = b_encodings["input_ids"].gather(1, y_idx)
            if y_neighbor_idx is not None:
                masked_input_ids_bak = b_encodings["input_ids"].clone()
                masked_input_ids = [
                    b_encodings["input_ids"]
                    .scatter(1, y_neighbor_idx, value=self.tokenizer.mask_token_id)
                    .scatter(
                        1,
                        y_idx[:, : mask_tok_idx + 1],
                        src=masked_input_ids_bak.gather(
                            1, y_idx[:, : mask_tok_idx + 1]
                        ),
                    )
                    .scatter(1, idx_mask_idx, value=self.tokenizer.cls_token_id)
                    for mask_tok_idx in range(y_idx_size)
                ]
            else:
                masked_input_ids = [
                    b_encodings["input_ids"]
                    .scatter(
                        1, y_idx[:, mask_tok_idx:], value=self.tokenizer.mask_token_id
                    )
                    .scatter(1, idx_mask_idx, value=self.tokenizer.cls_token_id)
                    for mask_tok_idx in range(y_idx_size)
                ]
            masked_input_ids = torch.stack(masked_input_ids, dim=1).view(
                batch_size * y_idx_size, max_seq_size
            )
            token_type_ids = b_encodings["token_type_ids"].repeat(y_idx_size, 1)
            attention_mask = b_encodings["attention_mask"].repeat(y_idx_size, 1)
            y_logits_raw = self.model(
                **{
                    **b_encodings,
                    "input_ids": masked_input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
            )["logits"]
            y_logits_size = y_logits_raw.size(-1)
            y_idx_for_raw_logit = y_idx.view(batch_size * y_idx_size, 1)
            y_logits = y_logits_raw.gather(
                1, y_idx_for_raw_logit.unsqueeze(-1).expand(-1, -1, y_logits_size)
            ).view(batch_size, y_idx_size, y_logits_size)
        else:
            raise NotImplementedError
        return {"logits": y_logits, "true_ys": true_ys}

    def save(self, training_args: TrainingArguments):
        training_params = copy(training_args.__dict__)
        training_params.pop("output_dir")
        training_params.pop("amp")
        training_params.pop("patience")
        training_params.pop("device_id")
        training_params.pop("gradient_clip")
        training_params.pop("warmup_startup_factor")
        training_params.pop("warmup_epochs")
        training_params.pop("epochs")
        save_dir = os.path.join(
            os.path.join(training_args.output_dir, "informed_model"),
            json.dumps(training_params).replace(" ", ""),
        )
        print("{}: saving model @ {}".format(datetime.now(), save_dir))
        self.model.save_pretrained(save_dir)


class VinfoNullWrapper(nn.Module):
    def __init__(self, model_name, tokenizer) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            config=config,
        )
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        x_idx,
        y_idx,
        b_encodings,
        device="cuda:0",
        logit_compute_strategy: str = "mlm",
        context_mask_probability=0.0,
        y_neighbor_idx=None,
    ):
        # device = x_idx.device
        batch_size = x_idx.size(0)
        y_idx_size = y_idx.size(1)
        max_seq_size = b_encodings["input_ids"].size(1)
        idx_mask_idx = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        # print(y_idx)
        # print(y_neighbor_idx)
        # random_mask = torch.bernoulli((context_mask_probability * torch.ones(batch_size, max_seq_size, device=device)).scatter(1, y_idx, value=0).scatter(1, x_idx, value=0).scatter(1, idx_mask_idx, value=0))
        # print(random_mask)

        if logit_compute_strategy == "mlm":
            masked_input_ids = (
                b_encodings["input_ids"]
                .scatter(1, y_idx, value=self.tokenizer.mask_token_id)
                .scatter(1, x_idx, value=self.tokenizer.mask_token_id)
            )
            if y_neighbor_idx is not None:
                masked_input_ids = masked_input_ids.scatter(
                    1, y_neighbor_idx, value=self.tokenizer.mask_token_id
                )
            masked_input_ids = masked_input_ids.scatter(
                1, idx_mask_idx, value=self.tokenizer.cls_token_id
            )
            true_ys = b_encodings["input_ids"].gather(1, y_idx)
            y_logits = self.model(**{**b_encodings, "input_ids": masked_input_ids})[
                "logits"
            ]
            y_logits_size = y_logits.size(-1)
            y_logits = y_logits.gather(
                1, y_idx.unsqueeze(-1).expand(-1, -1, y_logits_size)
            )
        elif logit_compute_strategy == "clm":
            y_idx_size = y_idx.size(1)
            true_ys = b_encodings["input_ids"].gather(1, y_idx)
            if y_neighbor_idx is not None:
                y_neighbor_idx_size = y_neighbor_idx.size(1)
                y_idx_size = y_idx.size(1)
                masked_input_ids_bak = b_encodings["input_ids"].clone()
                masked_input_ids = [
                    b_encodings["input_ids"]
                    .scatter(1, y_neighbor_idx, value=self.tokenizer.mask_token_id)
                    .scatter(1, x_idx, value=self.tokenizer.mask_token_id)
                    .scatter(
                        1,
                        y_idx[:, : mask_tok_idx + 1],
                        src=masked_input_ids_bak.gather(
                            1, y_idx[:, : mask_tok_idx + 1]
                        ),
                    )
                    .scatter(1, idx_mask_idx, value=self.tokenizer.cls_token_id)
                    for mask_tok_idx in range(y_idx_size)
                ]
            else:
                y_idx_size = y_idx.size(1)
                masked_input_ids = [
                    b_encodings["input_ids"]
                    .scatter(1, x_idx, value=self.tokenizer.mask_token_id)
                    .scatter(
                        1, y_idx[:, mask_tok_idx:], value=self.tokenizer.mask_token_id
                    )
                    .scatter(1, idx_mask_idx, value=self.tokenizer.cls_token_id)
                    for mask_tok_idx in range(y_idx_size)
                ]
            # masked_input_ids = [b_encodings['input_ids'].scatter(1, y_idx[:, mask_tok_idx:], value=self.tokenizer.mask_token_id).scatter(1, x_idx, value=self.tokenizer.mask_token_id).scatter(1, idx_mask_idx, value=self.tokenizer.cls_token_id) for mask_tok_idx in range(y_idx_size)]
            masked_input_ids = torch.stack(masked_input_ids, dim=1).view(
                batch_size * y_idx_size, max_seq_size
            )
            token_type_ids = b_encodings["token_type_ids"].repeat(y_idx_size, 1)
            attention_mask = b_encodings["attention_mask"].repeat(y_idx_size, 1)
            # if y_idx_size>1:
            #     print(masked_input_ids)
            y_logits_raw = self.model(
                **{
                    **b_encodings,
                    "input_ids": masked_input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
            )["logits"]
            y_logits_size = y_logits_raw.size(-1)
            y_idx_for_raw_logit = y_idx.view(
                batch_size * y_idx_size, 1
            )  # torch.stack(y_idx.chunk(y_idx_size, dim=1), dim=1).view(batch_size*y_idx_size, 1)
            # print(y_idx_for_raw_logit)
            # if y_idx_size>1:
            #     print(y_idx_for_raw_logit)
            y_logits = y_logits_raw.gather(
                1, y_idx_for_raw_logit.unsqueeze(-1).expand(-1, -1, y_logits_size)
            ).view(batch_size, y_idx_size, y_logits_size)
            # print(y_logits)

        else:
            raise NotImplementedError
        return {"logits": y_logits, "true_ys": true_ys}

    def save(self, training_args: TrainingArguments):
        training_params = copy(training_args.__dict__)
        training_params.pop("output_dir")
        training_params.pop("amp")
        training_params.pop("patience")
        training_params.pop("device_id")
        training_params.pop("gradient_clip")
        training_params.pop("warmup_startup_factor")
        training_params.pop("warmup_epochs")
        training_params.pop("epochs")
        save_dir = os.path.join(
            os.path.join(training_args.output_dir, "null_model"),
            json.dumps(training_params).replace(" ", ""),
        )
        print("{}: saving model".format(datetime.now()))
        self.model.save_pretrained(save_dir)

class MLMLogitCache:
    # the cache must be stored at cpu to save memory
    # cache:Dict[Any, torch.Tensor] = field(default_factory=lambda: {})
    # here the tensor should be of $$(b_gather_idx_size, vocab_size)
    def __init__(self) -> None:
        self.cache = {}
        pass
    
    def clear_cache(self):
        del self.cache
        self.cache = {}

    def search_from_cache(self, b_encodings, b_bid, b_idx, b_target_idx, vocab_size):
        target_device = b_encodings.input_ids.device

        b_bz = b_encodings.input_ids.size(0)
        b_xy_input_ids = b_encodings.input_ids.gather(1, b_idx)
        # b_idx_size = 1 #b_idx.size(1)
        b_bid_xyid = torch.cat([b_bid.view(b_bz, 1), b_xy_input_ids], dim=1).cpu()
        
        b_mask_hit_cache = torch.zeros(b_bz).bool()
        # This implementation only deals with the single-token case
        b_cached_mlm = torch.zeros(b_bz, 1, vocab_size)
        for bid_xyid in b_bid_xyid.unbind(0):
            bid_xyid = tuple(bid_xyid.tolist())
            bid = bid_xyid[0]
            if bid_xyid in self.cache.keys():
                b_mask_hit_cache[bid] = True
                b_cached_mlm[bid] = self.cache[bid_xyid]

        b_mask_miss_cache = torch.logical_not(b_mask_hit_cache)
        b_miss_input_ids = b_encodings.input_ids[b_mask_miss_cache]
        b_miss_attention_mask = b_encodings.attention_mask[b_mask_miss_cache]
        b_miss_token_type_ids = b_encodings.token_type_ids[b_mask_miss_cache]
        b_miss_encodings = BatchEncoding({
            'input_ids': b_miss_input_ids,
            'attention_mask': b_miss_attention_mask,
            'token_type_ids': b_miss_token_type_ids
        })
        b_miss_target_idx = b_target_idx[b_mask_miss_cache]
        
        # print('mlmlogitcache: b_mask_hit_cache', b_mask_hit_cache)
        # print('mlmlogitcache: b_miss_input_ids', b_miss_input_ids)

        return DictObj({
            'b_bid_xyid': b_bid_xyid,
            'b_mask_miss_cache': b_mask_miss_cache,
            'b_cached_logits': b_cached_mlm.to(target_device),
            'b_encodings': b_miss_encodings,
            'b_idx': b_miss_target_idx,
            'b_miss_count': torch.sum(b_mask_miss_cache)

        })

    def rebuild_full_output_and_cache_miss_output(self, b_hit_outputs, b_miss_outputs, b_mask_miss_cache, b_bid_xyid):
        # assuming that both b_bit_outputs and miss_outputs have (bz, 1, vocab_size) shape
        b_miss_bid_xyid = b_bid_xyid[b_mask_miss_cache]
        for miss_bid_xyid, miss_output in zip(b_miss_bid_xyid.unbind(0), b_miss_outputs.unbind(0)):
            miss_bid_xyid = tuple(miss_bid_xyid.tolist())
            assert miss_bid_xyid not in self.cache.keys()
            self.cache[miss_bid_xyid] = miss_output
        
        b_hit_outputs[b_mask_miss_cache] = b_miss_outputs
        return DictObj({
            'rebuilt_outputs': b_hit_outputs
        })


# %%
def convert_encodings_and_samples_to_bertpack(b_encodings, b_x_idx, b_y_idx, b_x_idx_mask, b_y_idx_mask, b_x_conditional_mask, b_y_conditional_mask, b_x_samples: torch.Tensor, b_y_samples: torch.Tensor):
    device = b_x_samples.device
    vocab_size = model.vocab_size
    # print(vocab_size)
    b_bz, num_samples = b_x_samples.size()
    b_max_seqlen = b_encodings.input_ids.size(1)
    b_bid = torch.arange(b_bz, device = device).unsqueeze(1).repeat(1, num_samples)
    b_samples_bid_x_y = torch.stack([b_bid.flatten(), b_x_samples.flatten(), b_y_samples.flatten()], dim=1)
    # print(b_samples_bid_x_y)
    b_unique_samples_bid_x_y, b_unique_samples_inverse_indices = torch.unique(b_samples_bid_x_y, dim=0, return_inverse=True)

    b_unique_samples_bid, b_unique_samples_x_samples, b_unique_samples_y_samples = b_unique_samples_bid_x_y.unbind(1)
    b_unique_samples_bid, b_unique_samples_x_samples, b_unique_samples_y_samples = b_unique_samples_bid.unsqueeze(1), b_unique_samples_x_samples.unsqueeze(1), b_unique_samples_y_samples.unsqueeze(1)
    b_num_unique_samples = b_unique_samples_bid.size(0)
 

    b_unique_samples_x_idx = b_x_idx.gather(0, b_unique_samples_bid)
    b_unique_samples_y_idx = b_y_idx.gather(0, b_unique_samples_bid)
    b_unique_samples_x_idx_mask = b_x_idx_mask.gather(0, b_unique_samples_bid)
    b_unique_samples_y_idx_mask = b_y_idx_mask.gather(0, b_unique_samples_bid)
    b_unique_samples_x_conditional_mask = b_x_conditional_mask.gather(0, b_unique_samples_bid.view(b_num_unique_samples, 1, 1).expand(b_num_unique_samples, 1, vocab_size))
    b_unique_samples_y_conditional_mask = b_y_conditional_mask.gather(0, b_unique_samples_bid.view(b_num_unique_samples, 1, 1).expand(b_num_unique_samples, 1, vocab_size))

    b_unique_samples_input_ids = b_encodings.input_ids.gather(0, b_unique_samples_bid.expand(-1, b_max_seqlen)).scatter(1, b_unique_samples_x_idx, b_unique_samples_x_samples).scatter(1, b_unique_samples_y_idx, b_unique_samples_y_samples)
    b_unique_samples_attention_mask = b_encodings.attention_mask.gather(0, b_unique_samples_bid.expand(-1, b_max_seqlen))
    b_unique_samples_token_type_ids = b_encodings.token_type_ids.gather(0, b_unique_samples_bid.expand(-1, b_max_seqlen))
    b_unique_samples_encodings = BatchEncoding({
        'input_ids': b_unique_samples_input_ids,
        'attention_mask': b_unique_samples_attention_mask,
        'token_type_ids': b_unique_samples_token_type_ids
    })

    return DictObj({
        'b_x_idx': b_unique_samples_x_idx,
        'b_y_idx': b_unique_samples_y_idx,
        'b_encodings': b_unique_samples_encodings,
        'inverse_indices': b_unique_samples_inverse_indices,
        'b_bid': b_unique_samples_bid,
        'b_x_idx_mask': b_unique_samples_x_idx_mask,
        'b_y_idx_mask': b_unique_samples_y_idx_mask,
        'b_x_conditional_mask': b_unique_samples_x_conditional_mask,
        'b_y_conditional_mask': b_unique_samples_y_conditional_mask,
    })
    
    pass
