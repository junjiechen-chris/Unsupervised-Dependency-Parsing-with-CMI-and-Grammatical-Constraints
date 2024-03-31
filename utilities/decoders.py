# from turtle import back
from typing import Any, Optional, List, Dict, Tuple, Union, Type, Set
from .data import UDSentence, CoNLLDataset
# from tqdm import tqdm
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    TrainingArguments,
    AutoTokenizer
)
import numpy as np
from torch.optim import Optimizer
import networkx as nx
from copy import copy
from datetime import datetime
import time
import os
import json

from .auxobjs import DictObj, ListObj



# Do i really need to implement this?!
class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases
        ln = nn.LayerNorm
        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    # @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        # print(input.size(), self.weight_ih.t().size())
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)



class BilinearAttn(nn.Module):
    def __init__(self, hidden_size_v, hidden_size_k) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(hidden_size_v, hidden_size_k))
        torch.nn.init.xavier_uniform_(self.W)
    def forward(self, k, V, attention_mask):
        # attention_mask (b, l)  bnv
        # suppose V -> (b*beams, l, h), k -> (b*beams, h)
        batch_size = k.size(0)
        # print(self.W.unsqueeze(0).expand(batch_size, -1, -1).size(), V.size())
        scores = torch.bmm(V, self.W.unsqueeze(0).expand(batch_size, -1, -1)).bmm(k.unsqueeze(-1)).squeeze(-1)
        # print(nn.functional.softmax(scores + (1-attention_mask) * -9999., dim=-1), flush=True)
        return nn.functional.softmax(scores + (1-attention_mask) * -9999., dim=-1)

def clamp(num, min_value, max_value):
    num = max(min(num, max_value), min_value)
    return num

class SRAgent(nn.Module):
    LEFT_ARC: int = 0
    RIGHT_ARC: int = 1
    SHIFT: int = 2
    REDUCE: int = 3
    def __init__(self, model_name:str, tokenizer, finetune_pretrained:bool = False, baseline_method = 'a2c') -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
        )
        self.model.resize_token_embeddings(len(tokenizer))
        bert_output_size = config.hidden_size
        self.lstm_hidden_size = 768 
        self.lstm_input_size = 128
        self.lstmcell_1 = LayerNormLSTMCell(self.lstm_input_size, self.lstm_hidden_size)
        self.lstmcell_2 = LayerNormLSTMCell(self.lstm_hidden_size, self.lstm_hidden_size)
        self.actor = nn.Linear(self.lstm_hidden_size+bert_output_size, 4)
        self.critic = nn.Linear(self.lstm_hidden_size, 1)
        self.action_emb = nn.Embedding(4, self.lstm_input_size)
        self.REDUCE=3
        self.LEFT_ARC = 0
        self.RIGHT_ARC = 1
        self.SHIFT = 2
        self.illegal_action_reward = -2.
        self.bilinear_attn = BilinearAttn(bert_output_size, self.lstm_hidden_size)
        self.finetune_pretrained:bool = finetune_pretrained
        self.baseline_method = baseline_method
        # 0-left_arc, 1-right_arc, 2-shift, 3-reduce

    def lstmcell(self, x, state_1, state_2):
        x = self.action_emb(x.squeeze(1))
        h1, new_state_1 = self.lstmcell_1(x, state_1)
        h2, new_state_2 = self.lstmcell_2(h1, state_2)
        return new_state_1, new_state_2

    @staticmethod
    def decode_dependency_structure_from_beam_history(beam_complete, beam_trace, beam_actions, beam_logprob, vinfo_mtx, buffer_size, device='cuda:0', decode_mode:str = 'vinfo'):
        # here to search in a beam batch
        # print(beam_complete)
        batch_size, max_seq_len = beam_complete.size()
        beam_complete_np = beam_complete.cpu().numpy()
        beam_trace_np = beam_trace.cpu().numpy()
        beam_actions_np = beam_actions.cpu().numpy()
        vinfo_mtx_np = vinfo_mtx.cpu().numpy()
        beams = []
        for i in range(batch_size):
            for j in range(max_seq_len):
                if beam_complete_np[i, j] == 1:
                    beams.append((i, j))
        # print(beams)
        # print(buffer_size)
        def back_trace(i, j):
            action_chain = []
            logprob = 0.
            while j>=0:
                action_chain.append(beam_actions_np[i, j])
                logprob+=beam_logprob[i, j]
                i = beam_trace_np[i, j]
                j -= 1
            return list(reversed(action_chain)), logprob


        def replay(actions, logprob):
            dependencies = []
            stack_head_status: List[int] = [0]
            stack_internal_node_status: List[int] = [0]
            stack: List[int] = [0]
            buffer_internal_node_status = 0
            total_head_count = 0
            buffer = list(range(1, buffer_size))
            vinfo = 0.
            # logprob = 0.
            for a in actions:
                if a == SRAgent.LEFT_ARC:
                    assert not( stack_head_status[-1] == 1 and stack[-1]==0)
                    dependencies.append((buffer[0], stack[-1]))
                    vinfo+=vinfo_mtx_np[buffer[0], stack[-1]]
                    stack.pop()
                    stack_head_status.pop()
                    buffer_internal_node_status = 1
                    total_head_count += stack_internal_node_status.pop()
                elif a == SRAgent.RIGHT_ARC:
                    dependencies.append((stack[-1], buffer[0]))
                    vinfo+=vinfo_mtx_np[stack[-1], buffer[0]]
                    stack.append(buffer.pop(0))
                    stack_head_status.append(1)
                    stack_internal_node_status[-1] = 1
                    stack_internal_node_status.append(buffer_internal_node_status)
                    buffer_internal_node_status = 0
                elif a == SRAgent.SHIFT:
                    stack.append(buffer.pop(0))
                    stack_head_status.append(0)
                    stack_internal_node_status.append(buffer_internal_node_status)
                    buffer_internal_node_status = 0
                elif a == SRAgent.REDUCE:
                    assert not stack_head_status[-1] == 0
                    stack.pop()
                    stack_head_status.pop()
                    total_head_count += stack_internal_node_status.pop()

                else:
                    raise NotImplementedError
                
                # print(stack)
                # print(stack_head_status)
            g: nx.DiGraph = nx.DiGraph()
            g.add_edges_from(dependencies)
            avg_vinfo = (vinfo - 3. * total_head_count)/(buffer_size-1)
            assert len(stack) == 1
            assert len(buffer) == 0
            return {
                'logprob': logprob,
                'nx': g,
                'vinfo': torch.clamp(avg_vinfo, min=-10., max=10),
                # 'vinfo': torch.where(torch.logical_and(avg_vinfo<=10.,avg_vinfo>=-10.), avg_vinfo, torch.tensor(-100., dtype=torch.float, device=device)) ,
                'plain_vinfo': torch.clamp(vinfo/(buffer_size-1), -10, 10)
                # torch.clamp(vinfo/(buffer_size-1), min=-10., max=10.)#min(max(vinfo/(buffer_size-1), 10), -10)
            }

        deps = sorted(list(map(lambda x: replay(*back_trace(*x)), beams)), key=lambda x: x[decode_mode], reverse=True)
        return deps
        # print(deps)


    def beam_decode(self, b_encodings, vinfo_mtx, buffer_size, forward_view_range = 1000, sample_beams = 2, device = 'cuda', stack_size = 10, decode_mode='vinfo'):

        remaining_buffer_status = []
        stack_cnt_status = []
        beam_status_wather = []
        beam_logp_watcher = []


        # pass
        input_ids = b_encodings['input_ids']
        batch_size = input_ids.size(0)
        masks = b_encodings['attention_mask']
        with torch.no_grad():
            output_hs = self.model(**b_encodings, output_hidden_states =True)['hidden_states'][-1]
        attention_mask  = b_encodings['attention_mask'].unsqueeze(1).repeat(1, sample_beams, 1).view(batch_size*sample_beams, -1)
        output_hs = output_hs.unsqueeze(1).repeat(1, sample_beams, 1, 1).view(batch_size*sample_beams, -1, output_hs.size(-1))
        max_buffer_size = torch.max(buffer_size)
        buffer_size = buffer_size.repeat(1, sample_beams).view(-1, 1)
        vinfo_mtx = vinfo_mtx.unsqueeze(1).repeat(1, sample_beams, 1, 1).view(batch_size*sample_beams, max_buffer_size, max_buffer_size)
        beam_probs = []
        beam_trace = []
        beam_actions = []
        beam_complete = []
        beam_logprobs = torch.zeros(batch_size*sample_beams, 1, device=device)
        
        parser_attn_masks = masks.clone()
        parser_attn_masks[:, 0] = 0
        sep_idx = torch.sum(masks, 1, keepdim=True)-1
        parser_attn_masks = parser_attn_masks.scatter(1, sep_idx, 0)


        stack_cnt = torch.zeros(batch_size*sample_beams, 1, dtype=torch.int64, device=device)
        stack = torch.zeros(batch_size*sample_beams, stack_size, dtype=torch.int64, device=device)
        stack_head_status = torch.zeros(batch_size*sample_beams, stack_size, dtype=torch.int64, device=device)
        beam_status = torch.cat([torch.ones(batch_size, 1, dtype = torch.int, device=device),
                                 torch.zeros(batch_size, sample_beams-1, dtype=torch.int, device=device)], -1).view(batch_size*sample_beams, 1) # a zero beam indicates invalid/finished beam
        buffer_pointer = torch.ones(batch_size* sample_beams, 1, dtype=torch.int64, device=device)
        # buffer_head_status = torch.zeros(batch_size*sample_beams, 1, dtype = torch.int64, device=device)

        beam_status_wather.append(beam_status)
        
        state_1 = torch.split(torch.zeros(2*sample_beams*batch_size, self.lstm_hidden_size, device=device), batch_size*sample_beams)
        state_2 = torch.split(torch.zeros(2*sample_beams*batch_size, self.lstm_hidden_size, device=device), batch_size*sample_beams)
        state_sequence = [(state_1, state_2)]


        while torch.any(beam_status):
            state_1, state_2 = state_sequence[-1]
            attn_scores = self.bilinear_attn(state_2[0], output_hs, attention_mask)  # ([b*beams, h], [b*beams, seq, h_bert]) -> (b*beams, seq)
            # print(attn_scores)
            attn_h = torch.sum(attn_scores.unsqueeze(-1) * output_hs, dim=1) # ([b*beams, seq], [b*beams, seq, h_bert]) -> (b*beams, h_bert)
            action_logits = self.actor(torch.cat([state_2[0], attn_h], dim=1)) # ([b*beams, h+h_bert]) -> ([b*beams, actions])
            action_logprobs = torch.log_softmax(action_logits, dim=-1)
            candidate_beam_logprobs = (beam_logprobs + action_logprobs)

            stack_top_head_status = stack_head_status.gather(1, stack_cnt)
            # print('stack head status', stack_head_status)
            # print('stack top head status', stack_top_head_status)

            actions = torch.arange(4).unsqueeze(0).expand(batch_size*sample_beams, 4).to(device)

            cond_valid_reduce = torch.logical_and(actions == self.REDUCE, 
                                    torch.logical_and(stack_top_head_status.expand(batch_size*sample_beams, 4) == 1,
                                                stack_cnt.expand(batch_size*sample_beams, 4)>0)) #([b*beams, 1])
            cond_valid_la = torch.logical_and(actions == self.LEFT_ARC, 
                                torch.logical_and(buffer_pointer.expand(batch_size*sample_beams, 4) < buffer_size.expand(batch_size*sample_beams, 4), 
                                    torch.logical_and(stack_top_head_status.expand(batch_size*sample_beams, 4) == 0,
                                                    stack_cnt.expand(batch_size*sample_beams, 4)>0)))
            cond_valid_ra = torch.logical_and(actions == self.RIGHT_ARC, torch.logical_and(buffer_pointer.expand(batch_size*sample_beams, 4)<buffer_size.expand(batch_size*sample_beams, 4), stack_cnt.expand(batch_size*sample_beams, 4)<stack_size-1))
            cond_valid_shift = torch.logical_and(actions == self.SHIFT, torch.logical_and(buffer_pointer.expand(batch_size*sample_beams, 4)<buffer_size.expand(batch_size*sample_beams, 4), stack_cnt.expand(batch_size*sample_beams, 4)<stack_size-1))

            mask_valid_beam = beam_status #* torch.where(cond_parser_status_guard, 1, 0)
            mask_valid_shift = torch.where(cond_valid_shift, 1, 0) * mask_valid_beam
            mask_valid_la = torch.where(cond_valid_la, 1, 0)* mask_valid_beam
            mask_valid_ra = torch.where(cond_valid_ra, 1, 0)* mask_valid_beam
            mask_valid_reduce = torch.where(cond_valid_reduce, 1, 0)* mask_valid_beam

            
            mask_illegal_actions = mask_valid_beam * (1 - (mask_valid_la + mask_valid_ra + mask_valid_shift + mask_valid_reduce))
            mask_legal_actions = mask_valid_beam * (1 - mask_illegal_actions)
            mask_valid_la *= mask_legal_actions
            mask_valid_ra *= mask_legal_actions
            mask_valid_stack_push = (mask_valid_ra + mask_valid_shift) * mask_legal_actions
            mask_valid_stack_pop = (mask_valid_la + mask_valid_reduce) * mask_legal_actions
            mask_valid_buffer_pop = (mask_valid_ra + mask_valid_shift) * mask_legal_actions
            #until now it operates over [batch_size * beams, ...] shape

            # find the top-k actions from the beam 
            # now everything should operate at [batchsize, .* beams]
            legal_action_logprobs = (mask_legal_actions * candidate_beam_logprobs + (1-mask_legal_actions) * -9999.).view(batch_size, sample_beams * 4)
            # print('legal_action_logps', legal_action_logprobs)
            topk_action_logp, topk_action_idx = torch.topk(legal_action_logprobs, sample_beams)
            # print('topk logp', topk_action_logp)
            topk_action_mask = torch.where(topk_action_logp>-9999., 1, 0).view(batch_size*sample_beams, 1)
            # print('topk action idx', topk_action_idx)
            topk_action_prev_beam, topk_actions = (topk_action_idx / 4).int(), topk_action_idx % 4
            beam_actions.append(topk_actions.view(batch_size*sample_beams, 1))
            beam_probs.append(topk_action_logp.view(batch_size*sample_beams, 1))
            beam_trace.append(topk_action_prev_beam.view(batch_size*sample_beams, 1) * topk_action_mask + (1-topk_action_mask) * -1)
            
            # refresh the stack/pointer status
            # print(stack.unsqueeze(1).repeat(1, 4, 1).view(batch_size, sample_beams*4, -1))
            actions = actions.reshape(batch_size, sample_beams*4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)
            stack = stack.unsqueeze(1).repeat(1, 4, 1).view(batch_size, sample_beams*4, stack_size).gather(1, topk_action_idx.unsqueeze(-1).expand(-1, -1, stack_size)).view(batch_size*sample_beams, stack_size) * topk_action_mask
            stack_cnt = stack_cnt.repeat(1, 4).view(batch_size, sample_beams*4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1) * topk_action_mask
            stack_head_status = stack_head_status.unsqueeze(1).repeat(1, 4, 1).view(batch_size, sample_beams*4, stack_size).gather(1, topk_action_idx.unsqueeze(-1).expand(-1, -1, stack_size)).view(batch_size*sample_beams, stack_size) * topk_action_mask
            buffer_pointer = buffer_pointer.repeat(1, 4).view(batch_size, sample_beams*4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1) * topk_action_mask
            # print(state_1[0].unsqueeze(1).repeat(1, 4, 1).view(batch_size, sample_beams*4, -1).size())
            gathered_state_1 = tuple((item.unsqueeze(1).repeat(1, 4, 1).view(batch_size, sample_beams*4, -1).gather(1, topk_action_idx.unsqueeze(-1).expand(-1, -1, self.lstm_hidden_size)).view(batch_size*sample_beams, self.lstm_hidden_size) for item in state_1))
            gathered_state_2 = tuple((item.unsqueeze(1).repeat(1, 4, 1).view(batch_size, sample_beams*4, -1).gather(1, topk_action_idx.unsqueeze(-1).expand(-1, -1, self.lstm_hidden_size)).view(batch_size*sample_beams, self.lstm_hidden_size) for item in state_2))
            # print('candidate beam logprob size', candidate_beam_logprobs.size())
            beam_logprobs = candidate_beam_logprobs.view(batch_size, sample_beams* 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1) * topk_action_mask
            # print(stack_cnt)
            # print('\n')

            gathered_mask_illegal_actions = mask_illegal_actions.view(batch_size, sample_beams * 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)
            gathered_mask_legal_actions = mask_legal_actions.view(batch_size, sample_beams * 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)
            gathered_mask_valid_stack_push = mask_valid_stack_push.view(batch_size, sample_beams * 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)
            gathered_mask_valid_stack_pop = mask_valid_stack_pop.view(batch_size, sample_beams * 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)
            gathered_mask_valid_buffer_pop = mask_valid_buffer_pop.view(batch_size, sample_beams * 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)
            gathered_mask_valid_la = mask_valid_la.view(batch_size, sample_beams * 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)
            gathered_mask_valid_ra = mask_valid_ra.view(batch_size, sample_beams * 4).gather(1, topk_action_idx).view(batch_size*sample_beams, 1)


            # print('gathered mask', gathered_mask_illegal_actions)



            # stack_word_idx = (stack.gather(1, stack_cnt) * mask_valid_beam).int()
            buffer_word_idx = (buffer_pointer * mask_valid_beam).int()
            # ra_vinfo_idx = (buffer_word_idx * max_buffer_size + stack_word_idx).type(torch.int64).to(device) * gathered_mask_valid_ra
            # la_vinfo_idx = (stack_word_idx * max_buffer_size + buffer_word_idx).type(torch.int64).to(device) * gathered_mask_valid_la
            # ra_vinfo_reward = torch.gather(vinfo_mtx.view(batch_size*sample_beams, -1), 1, ra_vinfo_idx) * gathered_mask_valid_ra
            # la_vinfo_reward = torch.gather(vinfo_mtx.view(batch_size*sample_beams, -1), 1, la_vinfo_idx) * gathered_mask_valid_la
            
            

            # buffer_head_status += gathered_mask_valid_ra

            stack_cnt = stack_cnt + 1 * gathered_mask_valid_stack_push# - 1 * mask_valid_stack_pop
            stack_head_status = stack_head_status.scatter(1, stack_cnt, 1 * gathered_mask_valid_ra)
            # print('stack_cnt', stack_cnt.view(-1))
            # print('remaining buffer size', (buffer_size-buffer_pointer).view(-1))
            stack = stack.scatter(1, stack_cnt, buffer_word_idx * gathered_mask_valid_stack_push ) # ignoring invalid update...?
            stack_cnt -= 1 * gathered_mask_valid_stack_pop
            # print('stack', stack)

            # print('gathered_mask_valid_ra', gathered_mask_valid_ra)
            # stack_head_status = stack_head_status.scatter(1, stack_cnt, 1 * gathered_mask_valid_ra)


            buffer_pointer = buffer_pointer + 1 * gathered_mask_valid_buffer_pop
            
            cond_valid_complete = torch.logical_and(buffer_pointer == buffer_size, stack_cnt == 0)
            mask_valid_complete = mask_valid_beam * torch.where(cond_valid_complete, 1, 0)
            beam_complete.append(mask_valid_complete)
            # remaining_buffer_status.append(buffer_size-buffer_pointer)
            # stack_cnt_status.append(stack_cnt)





            beam_status = gathered_mask_legal_actions * (1 - mask_valid_complete) #* mask_valid_beam
            beam_status_wather.append(beam_status)#gathered_mask_legal_actions * (1-mask_valid_complete))
            # print("update state sequence")
            # print('beam status', beam_status)
            # print(actions)
            # print(stack)
            # print(stack_cnt)
            # print(buffer_pointer)
            
            # print(gathered_state_1[0].size(), gathered_state_2[0].size())
        
            state_sequence.append(self.lstmcell(actions, gathered_state_1, gathered_state_2))
            # print('\n\n')
        
        # torch.set_printoptions(profile="full")
        # print('beam complete', torch.cat(beam_complete, dim=-1))
        # print('beam status', torch.cat(beam_status_wather, dim=-1))
        # print('stack cnt stauts', torch.cat(stack_cnt_status, dim=-1))
        # print('buffer status', torch.cat(remaining_buffer_status, dim=-1))
        # print('prev beam status', torch.cat(beam_trace, dim=-1))
        # print('beam actions', torch.cat(beam_actions, dim=-1))
        # print('beam logp', torch.cat(beam_probs, dim=-1))
        # torch.set_printoptions(profile="default")

        beam_complete = torch.cat(beam_complete, dim=-1)
        beam_trace = torch.cat(beam_trace, dim=-1)
        beam_actions = torch.cat(beam_actions, dim=-1)
        beam_probs = torch.cat(beam_probs, dim=-1)

        # reconstruct from complete beams
        # return 
        # print(list(range(0, batch_size*sample_beams, sample_beams)))
        outputs = []
        for beam_l in range(0, batch_size*sample_beams, sample_beams):
            batch_id = int(beam_l/sample_beams)
            # print("start reconstruction for ", batch_id)
            outputs.append(SRAgent.decode_dependency_structure_from_beam_history(beam_complete[beam_l:beam_l+sample_beams],
                                                                beam_trace[beam_l:beam_l+sample_beams],
                                                                beam_actions[beam_l:beam_l+sample_beams],
                                                                beam_logprob=beam_probs,
                                                                vinfo_mtx=vinfo_mtx[beam_l, :buffer_size[beam_l], :buffer_size[beam_l]],
                                                                buffer_size = buffer_size[beam_l], 
                                                                decode_mode=decode_mode,
                                                                device=device))
        return outputs
            
        




    def forward(self, b_encodings, vinfo_mtx, buffer_size,  forward_view_range = 1, sample_beams = 1, device = 'cuda', stack_size = 10, baseline_params: Optional[Dict[str, Any]] = None, optimizer:Optimizer = None):
        # TODO: need to double check the implementation

        # do seq2seq parsing
        #sample-beam: the number of workers in a2/3c
        input_ids = b_encodings['input_ids']
        batch_size = input_ids.size(0)
        # max_seq_len = input_ids.size(1)
        # device = input_ids.device
        # masks = b_encodings['attention_mask']
        if self.finetune_pretrained:
            output_hs = self.model(**b_encodings, output_hidden_states =True)['hidden_states'][-1]
        else:
            with torch.no_grad():
                output_hs = self.model(**b_encodings, output_hidden_states =True)['hidden_states'][-1]
        output_hs = output_hs.unsqueeze(1).repeat(1, sample_beams, 1, 1).view(batch_size*sample_beams, -1, output_hs.size(-1))
        attention_mask  = b_encodings['attention_mask'].unsqueeze(1).repeat(1, sample_beams, 1).view(batch_size*sample_beams, -1)
        # print(output_hs.size())
        max_buffer_size = torch.max(buffer_size)
        buffer_size = buffer_size.repeat(1, sample_beams).view(-1, 1)
        # logits = logits.repeat(1, sample_beams, 1).resize(batch_size*sample_beams, max_seq_len, -1)
        vinfo_mtx = vinfo_mtx.unsqueeze(1).repeat(1, sample_beams, 1, 1).view(batch_size*sample_beams, max_buffer_size, max_buffer_size)



        stack_cnt = torch.zeros(batch_size*sample_beams, 1, dtype=torch.int64, device=device)
        stack = torch.zeros(batch_size*sample_beams, stack_size, dtype=torch.int64, device=device)
        stack_head_status = torch.zeros(batch_size*sample_beams, stack_size, dtype=torch.int64, device=device)
        stack_internal_node_status = torch.zeros(batch_size*sample_beams, stack_size, dtype=torch.int64, device=device)
        buffer_internal_node_status = torch.zeros(batch_size*sample_beams, 1, dtype=torch.int64, device=device)
        # stack_head
        # beam_status = torch.ones(batch_size*sample_beams, 1, dtype = torch.int, device=device) # a zero beam indicates invalid/finished beam
        beam_status = torch.cat([torch.ones(batch_size, 1, dtype = torch.int, device=device),
                                 torch.zeros(batch_size, sample_beams-1, dtype=torch.int, device=device)], -1).view(batch_size*sample_beams, 1) # a zero beam indicates invalid/finished beam
        buffer_pointer = torch.ones(batch_size* sample_beams, 1, dtype=torch.int64, device=device)
        
        
        state_1 = torch.split(torch.zeros(2*sample_beams*batch_size, self.lstm_hidden_size, device=device), batch_size*sample_beams)
        state_2 = torch.split(torch.zeros(2*sample_beams*batch_size, self.lstm_hidden_size, device=device), batch_size*sample_beams)
        state_sequence = [(state_1, state_2)]


        #doing some one-sample monte-carlo
        loop_cnt = 0
        gradients = []
        rewards = []
        critic_loss = []
        active_rewards = 0


        def translate_ordered_idx_to_running_beam_idx(ordered_selection_idx, beam_status, beam_size):
            # print('beam status', beam_status)
            # print('ordered selection idx', ordered_selection_idx)
            if torch.sum(beam_status) == 0:
                return torch.zeros(beam_size, dtype=torch.int64, device=device)
            permutation_idx = torch.randperm(ordered_selection_idx.size(0), device = device)
            ordered_selection_idx = ordered_selection_idx[permutation_idx][:beam_size]
            # print('permuted ordered selection idx', ordered_selection_idx)
            translation_map = torch.tensor([id for id, status in enumerate(beam_status) if status == 1], device=device)
            # print('translated idx', translation_map.gather(0, ordered_selection_idx))
            return translation_map.gather(0, ordered_selection_idx)
        
        def run_beam_a2c(states, beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, mode = 'multinomial'):            
            state_1, state_2 = states
            action_probs = []
            step_rewards = []
            beam_history = []
            v_values = [self.critic(state_2[0]) * beam_status]
            for step_cnt in range(forward_view_range):
                if torch.all(1-beam_status): break
                
                # print(beam_status, flush=True)
                # state_1, state_2 = state_sequence[-1]
                # print(state_2)

                #TODO: enable the attention mechanism to handle beam > 1
                # print(state_2[0].size(), output_hs.size())
                attn_scores = self.bilinear_attn(state_2[0], output_hs, attention_mask)  # ([b, h], [b, seq, h_bert]) -> (b, seq)
                attn_h = torch.sum(attn_scores.unsqueeze(-1) * output_hs, dim=1) # ([b, seq], [b, seq, h_bert]) -> (b, h_bert)
                action_logits = self.actor(torch.cat([state_2[0], attn_h], dim=1)) # ([b, h+h_bert]) -> ([b, actions])

                # assert torch.isfinite(action_logits)
                if not torch.isfinite(action_logits):
                    print(action_logits)
                actions = torch.multinomial(torch.softmax(torch.clamp(action_logits, min=-1e8), dim=-1), 1).view(batch_size*sample_beams, 1) # ([b, actions]) -> ([b, beams]) -> ([b*beams])
                action_probs.append(torch.gather(torch.log_softmax(action_logits, dim=-1), 1, actions)) #([b, actions], [b*beams]) -> ([b, beams])

                # cond_valid_complete =                 
                stack_top_head_status = stack_head_status.gather(1, stack_cnt)
                cond_valid_reduce = torch.logical_and(actions == self.REDUCE, 
                                        torch.logical_and(stack_top_head_status == 1, stack_cnt>0)) #([b*beams, 1])
                cond_valid_la = torch.logical_and(actions == self.LEFT_ARC, 
                                    torch.logical_and(stack_top_head_status == 0,
                                        torch.logical_and(buffer_pointer < buffer_size, stack_cnt>0)))
                cond_valid_ra = torch.logical_and(actions == self.RIGHT_ARC, torch.logical_and(buffer_pointer<buffer_size, stack_cnt<stack_size-1))
                cond_valid_shift = torch.logical_and(actions == self.SHIFT, torch.logical_and(buffer_pointer<buffer_size, stack_cnt<stack_size-1))

                mask_valid_beam = beam_status #* torch.where(cond_parser_status_guard, 1, 0)
                mask_valid_shift = torch.where(cond_valid_shift, 1, 0) * mask_valid_beam
                mask_valid_la = torch.where(cond_valid_la, 1, 0) * mask_valid_beam
                mask_valid_ra = torch.where(cond_valid_ra, 1, 0) * mask_valid_beam
                mask_valid_reduce = torch.where(cond_valid_reduce, 1, 0) * mask_valid_beam

                mask_illegal_actions = mask_valid_beam * (1 - (mask_valid_la + mask_valid_ra + mask_valid_shift + mask_valid_reduce))
                mask_legal_actions = mask_valid_beam * (1 - mask_illegal_actions)
                mask_valid_la *= mask_legal_actions
                mask_valid_ra *= mask_legal_actions
                mask_valid_stack_push = (mask_valid_ra + mask_valid_shift) * mask_legal_actions
                mask_valid_stack_pop = (mask_valid_la + mask_valid_reduce) * mask_legal_actions
                mask_valid_buffer_pop = (mask_valid_ra + mask_valid_shift) * mask_legal_actions
                mask_beam_should_be_active = torch.where(step_cnt < 2*buffer_size, 1, 0)
                mask_invalid_beam = 1-mask_valid_beam


                stack_word_idx = (stack.gather(1, stack_cnt) * mask_valid_beam).int()
                buffer_word_idx = (buffer_pointer * mask_valid_beam).int()
                ra_vinfo_idx = (buffer_word_idx * max_buffer_size + stack_word_idx).type(torch.int64).to(device) * mask_valid_ra
                la_vinfo_idx = (stack_word_idx * max_buffer_size + buffer_word_idx).type(torch.int64).to(device) * mask_valid_la
                ra_vinfo_reward = torch.gather(vinfo_mtx.view(batch_size*sample_beams, -1), 1, ra_vinfo_idx) * mask_valid_ra
                la_vinfo_reward = torch.gather(vinfo_mtx.view(batch_size*sample_beams, -1), 1, la_vinfo_idx) * mask_valid_la
                assert torch.logical_not(torch.any(torch.isnan(ra_vinfo_reward)))
                assert torch.logical_not(torch.any(torch.isnan(la_vinfo_reward)))

                step_rewards.append( mask_valid_beam * (mask_illegal_actions * self.illegal_action_reward + ra_vinfo_reward + la_vinfo_reward) + mask_invalid_beam * mask_beam_should_be_active * self.illegal_action_reward )
                

                stack_cnt = stack_cnt + 1 * mask_valid_stack_push
                stack_head_status = stack_head_status.scatter(1, stack_cnt, 1*mask_valid_ra)
                stack = stack.scatter(1, stack_cnt, buffer_word_idx * mask_valid_stack_push ) # ignoring invalid update...?
                stack_cnt -= 1 * mask_valid_stack_pop
                # print("stack", stack)

                # print("buffer_pointer", buffer_pointer, mask_valid_buffer_pop)
                buffer_pointer = buffer_pointer + 1 * mask_valid_buffer_pop
                
                # print('stack head status', stack_head_status)
                # print('stack', stack)
                # print("buffer_pointer", buffer_pointer)

                cond_valid_complete = torch.logical_and(buffer_pointer == buffer_size, stack_cnt == 0)
                mask_valid_complete = mask_valid_beam * torch.where(cond_valid_complete, 1, 0)
                # print(torch.any(mask_valid_complete))




                beam_status = mask_legal_actions * (1 - mask_valid_complete) * mask_valid_beam
                state_1, state_2 = self.lstmcell(actions, state_1, state_2)
                v_values.append(self.critic(state_2[0]) * beam_status)
                beam_history.append(beam_status)
                # print("update state sequence")
                # state_sequence.append(self.lstmcell(actions, state_1, state_2))
                
                # print('end updating')
                # loop_cnt += 1

            # step_rewards = torch.cat(step_rewards, dim=-1)
            # v_values = torch.cat(v_values, dim=-1)
            # action_probs = torch.cat(action_probs, dim=-1)



            return (state_1, state_2), beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, step_rewards, action_probs, v_values, beam_history


        def run_beam_scst(states, beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, stack_internal_node_status, buffer_internal_node_status):
            # stack_head_status: whether a token has a head
            # stack_internal_node_status: whether a token served as a head
            # buffer_internal_node_status: whether the leftmost token served as a head
            device = beam_status.device

            state_1, state_2 = states
            action_probs = 0.
            step_rewards:torch.Tensor = 0.
            entropy_loss = 0.
            collected_internal_nodes = torch.zeros_like(stack_cnt, device = device)
            active_beam_steps = 0
            for step_cnt in range(forward_view_range):
                if torch.all(1-beam_status): break
                
                # print(beam_status, flush=True)
                # state_1, state_2 = state_sequence[-1]
                # print(state_2)

                #TODO: enable the attention mechanism to handle beam > 1
                # print(state_2[0].size(), output_hs.size())
                attn_scores = self.bilinear_attn(state_2[0], output_hs, attention_mask)  # ([b, h], [b, seq, h_bert]) -> (b, seq)
                attn_h = torch.sum(attn_scores.unsqueeze(-1) * output_hs, dim=1) # ([b, seq], [b, seq, h_bert]) -> (b, h_bert)
                action_logits = self.actor(torch.cat([state_2[0], attn_h], dim=1)) # ([b, h+h_bert]) -> ([b, actions])

                # print(torch.softmax(action_logits, dim=-1))


                actions = torch.multinomial(torch.softmax(torch.clamp(action_logits, min=-9e9), dim=-1), 1).view(batch_size*sample_beams, 1) # ([b, actions]) -> ([b, beams]) -> ([b*beams])
                action_probs += torch.gather(torch.log_softmax(action_logits, dim=-1), 1, actions) #([b, actions], [b*beams]) -> ([b, beams])
                # print(entropy(action_logits))
                entropy_loss += torch.sum(entropy(action_logits) * beam_status)
                active_beam_steps += torch.sum(beam_status)
                # print("actions", torch.cat([actions, action_probs], dim=-1))
                # print(actions.size())

                # cond_valid_complete =                 
                stack_top_head_status = stack_head_status.gather(1, stack_cnt)
                # stack_internal_node_status = stack_head_status.gather(1, stack_cnt)
                cond_valid_reduce = torch.logical_and(actions == self.REDUCE, 
                                        torch.logical_and(stack_top_head_status == 1, stack_cnt>0)) #([b*beams, 1])
                cond_valid_la = torch.logical_and(actions == self.LEFT_ARC, 
                                    torch.logical_and(stack_top_head_status == 0,
                                        torch.logical_and(buffer_pointer < buffer_size, stack_cnt>0)))
                cond_valid_ra = torch.logical_and(actions == self.RIGHT_ARC, torch.logical_and(buffer_pointer<buffer_size, stack_cnt<stack_size-1))
                cond_valid_shift = torch.logical_and(actions == self.SHIFT, torch.logical_and(buffer_pointer<buffer_size, stack_cnt<stack_size-1))
                # print(buffer_cnt)
                # print(cond_invalid_reduce)
                # cond_parser_status_guard = torch.logical_and(torch.logical_and(stack_cnt >= 0, stack_cnt < 10), buffer_cnt>=0)

                mask_valid_beam = beam_status #* torch.where(cond_parser_status_guard, 1, 0)
                mask_valid_shift = torch.where(cond_valid_shift, 1, 0) * mask_valid_beam
                mask_valid_la = torch.where(cond_valid_la, 1, 0) * mask_valid_beam
                mask_valid_ra = torch.where(cond_valid_ra, 1, 0) * mask_valid_beam
                mask_valid_reduce = torch.where(cond_valid_reduce, 1, 0) * mask_valid_beam
                # print(torch.cat([mask_valid_beam, mask_valid_shift, mask_valid_la, mask_valid_ra, mask_valid_reduce], dim=-1))
                # print(mask_valid_beam.size(), mask_valid_shift.size(), mask_valid_la.size(), mask_valid_reduce.size(), mask_valid_ra.size())

                mask_illegal_actions = mask_valid_beam * (1 - (mask_valid_la + mask_valid_ra + mask_valid_shift + mask_valid_reduce))
                mask_legal_actions = mask_valid_beam * (1 - mask_illegal_actions)
                mask_valid_la *= mask_legal_actions
                mask_valid_ra *= mask_legal_actions
                mask_valid_stack_push = (mask_valid_ra + mask_valid_shift) * mask_legal_actions
                mask_valid_stack_pop = (mask_valid_la + mask_valid_reduce) * mask_legal_actions
                mask_valid_buffer_pop = (mask_valid_ra + mask_valid_shift) * mask_legal_actions
                mask_beam_should_be_active = torch.where(step_cnt < 2*buffer_size, 1, 0)
                mask_invalid_beam = 1-mask_valid_beam

                # print("illegal actions", 'stack_pop', 'buffer_pop', 'stack_push')
                # print(torch.cat([mask_illegal_actions, mask_valid_stack_pop, mask_valid_buffer_pop, mask_valid_stack_push], dim=-1))

                stack_word_idx = (stack.gather(1, stack_cnt) * mask_valid_beam).int()
                buffer_word_idx = (buffer_pointer * mask_valid_beam).int()
                # print()
                ra_vinfo_idx = (buffer_word_idx * max_buffer_size + stack_word_idx).type(torch.int64).to(device) * mask_valid_ra
                la_vinfo_idx = (stack_word_idx * max_buffer_size + buffer_word_idx).type(torch.int64).to(device) * mask_valid_la
                # print(torch.cat([stack_word_idx, buffer_word_idx], dim=-1))
                # print('ra_vinfo_idx', ra_vinfo_idx)
                # print("vinfo mtx", vinfo_mtx)
                ra_vinfo_reward = torch.gather(vinfo_mtx.view(batch_size*sample_beams, -1), 1, ra_vinfo_idx) * mask_valid_ra
                la_vinfo_reward = torch.gather(vinfo_mtx.view(batch_size*sample_beams, -1), 1, la_vinfo_idx) * mask_valid_la
                # print(ra_vinfo_reward)
                assert torch.logical_not(torch.any(torch.isnan(ra_vinfo_reward)))
                assert torch.logical_not(torch.any(torch.isnan(la_vinfo_reward)))

                # print(ra_vinfo_reward)

                # print('ra_vinfo_reward', ra_vinfo_reward)
                # print('la_vinfo_reward', la_vinfo_reward)

                step_rewards += mask_valid_beam * (mask_illegal_actions * self.illegal_action_reward + ra_vinfo_reward + la_vinfo_reward ) + mask_invalid_beam * mask_beam_should_be_active * self.illegal_action_reward
                

                
                # !! HOW TO SET THE STACK-TOP TO THE HAVE-CHILD STATE?
                # ! NOT SURE IF IT'S WORKING CORRECTLY
                stack_internal_node_status = stack_internal_node_status.scatter(1, stack_cnt, 1*mask_valid_ra + stack_internal_node_status.gather(1, stack_cnt) * (1-mask_valid_ra))
                # print(stack_internal_node_status)
                stack_cnt = stack_cnt + 1 * mask_valid_stack_push# - 1 * mask_valid_stack_pop
                # !! watchout for the mask_valid_ra shape
                # print(buffer_internal_node_status, mask_valid_ra)
                # print(stack_internal_node_status)
                collected_internal_nodes+=stack_internal_node_status.gather(1, stack_cnt) * mask_valid_stack_pop
                # print('stack_top_internal_status', stack_internal_node_status.gather(1, stack_cnt), stack_internal_node_status.gather(1, stack_cnt) * mask_valid_stack_pop)
                stack_internal_node_status = stack_internal_node_status.scatter(1, stack_cnt, buffer_internal_node_status*mask_valid_stack_push)
                buffer_internal_node_status = torch.clamp(buffer_internal_node_status + mask_valid_la, min = 0, max = 1)
                # print('mask valid ra', mask_valid_ra)
                stack_head_status = stack_head_status.scatter(1, stack_cnt, 1*mask_valid_ra)
                stack = stack.scatter(1, stack_cnt, buffer_word_idx * mask_valid_stack_push ) # ignoring invalid update...?
                stack_cnt -= 1 * mask_valid_stack_pop
                # print("stack", stack)

                # print("buffer_pointer", buffer_pointer, mask_valid_buffer_pop)
                buffer_pointer = buffer_pointer + 1 * mask_valid_buffer_pop
                
                # print('stack head status', stack_head_status)
                # print('stack', stack)
                # print("buffer_pointer", buffer_pointer)

                cond_valid_complete = torch.logical_and(buffer_pointer == buffer_size, stack_cnt == 0)
                mask_valid_complete = mask_valid_beam * torch.where(cond_valid_complete, 1, 0)
                # print(torch.any(mask_valid_complete))




                beam_status = mask_legal_actions * (1 - mask_valid_complete) * mask_valid_beam
                state_1, state_2 = self.lstmcell(actions, state_1, state_2)
                # print("update state sequence")
                # state_sequence.append(self.lstmcell(actions, state_1, state_2))
                
                # print('end updating')
                # loop_cnt += 1
            assert list(step_rewards.size()) == [batch_size*sample_beams, 1]
            # print(collected_internal_nodes)
            step_rewards -= 3. * collected_internal_nodes
            
            return (state_1, state_2), beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, step_rewards, action_probs, entropy_loss/active_beam_steps


        loss = torch.tensor(0, dtype=torch.float, device=device)
        accumulation_cnt = 0
        v_function_loss = 0.
        accumulated_active_rewards = 0.
        active_rewards = 0
        log_dict = {
            'advantages': 0.,
            'critic MSE': 0.,
            'loss': 0.
            }
        while torch.any(beam_status):



            # if loop_cnt > 120:
                # print('too much ops')
                # break
            # print('####################new episode##################')
            # print(beam_status.view(-1))

            # populate the slots with available beams
            # leave the group by [a, b, c, a, b, c] where [a, b, c] are running beams
            #what if none of the beams are valid?
            running_beams = torch.sum(beam_status.view(batch_size, sample_beams), dim=1)
            running_beam_status = beam_status.view(batch_size, sample_beams)
            beam_collecting_idx = [batch_id * sample_beams + translate_ordered_idx_to_running_beam_idx(torch.arange(max(1, running_beams[batch_id]), device=device).repeat(sample_beams), running_beam_status[batch_id], sample_beams) for batch_id in range(batch_size)]
            beam_collecting_idx = torch.cat(beam_collecting_idx, dim=0).view(batch_size*sample_beams, 1).to(device)
            # print('beam collecting idx', beam_collecting_idx)

            state_1, state_2 = state_sequence[-1]
            state_1 = tuple((item.gather(0, beam_collecting_idx.expand(-1, self.lstm_hidden_size)) for item in state_1))
            state_2 = tuple((item.gather(0, beam_collecting_idx.expand(-1, self.lstm_hidden_size)) for item in state_2))
            stack = stack.gather(0, beam_collecting_idx.expand(-1, stack_size))
            stack_cnt = stack_cnt.gather(0, beam_collecting_idx)
            buffer_pointer = buffer_pointer.gather(0, beam_collecting_idx)
            stack_head_status = stack_head_status.gather(0, beam_collecting_idx.expand(-1, stack_size))
            stack_internal_node_status = stack_internal_node_status.gather(0, beam_collecting_idx.expand(-1, stack_size))
            buffer_internal_node_status = buffer_internal_node_status.gather(0, beam_collecting_idx)
            beam_status = beam_status.gather(0, beam_collecting_idx)

            active_rewards += torch.sum(beam_status)
            accumulated_active_rewards += torch.sum(beam_status)
            # print('currently running beams', torch.sum(beam_status))


            if baseline_params is not None:
                baseline_method = baseline_params.get('method', 'scst')
                if baseline_method == 'scst':
                    baseline_rewards = baseline_params['baseline_rewards']
                    baseline_rewards = baseline_rewards.unsqueeze(1).repeat(1, sample_beams).view(batch_size*sample_beams, 1)
                    _, beam_status, _, _, _, _, step_rewards, action_probs, entropy_loss = run_beam_scst((state_1, state_2), beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, stack_internal_node_status, buffer_internal_node_status)
                    actor_rewards = step_rewards - baseline_rewards
                    loss = -torch.mean(actor_rewards * action_probs) - 0.1 * entropy_loss
                    log_dict['advantage'] = torch.mean(actor_rewards)
                    log_dict['loss'] = loss
                    log_dict['policy entropy'] = entropy_loss
                elif baseline_method == 'a2c':
                    _, beam_status, _, _, _, _, step_rewards, action_probs, v_values, beam_history = run_beam_a2c((state_1, state_2), beam_status, stack_head_status, stack, stack_cnt, buffer_pointer)
                    R = v_values[-1]
                    rl_loss = torch.zeros(batch_size*sample_beams, 1, dtype=torch.float, device=device)
                    v_mse_loss = torch.zeros(batch_size*sample_beams, 1, dtype=torch.float, device=device)
                    dividend = torch.sum(torch.cat(beam_history, dim=-1))
                    for r, v, logp in zip(reversed(step_rewards), reversed(v_values[:-1]), reversed(action_probs)):
                        R = r + R
                        # print(R, v)
                        rl_loss += logp*(R-v)
                        v_mse_loss += torch.pow((R-v), 2)
                    loss = torch.sum(rl_loss)/dividend + torch.sum(v_mse_loss)/dividend
                    log_dict['critic MSE'] = torch.sum(v_mse_loss)/dividend
                    log_dict['loss'] = loss
            assert torch.logical_not(torch.any(beam_status))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return log_dict
    

                    









            # if self.baseline_method == 'a2c':
            #     v_head = self.critic(state_2[0]) * beam_status
            #     states, beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, step_reward, action_probs, optional_outputs = run_beam((state_1, state_2), beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, mode='multinomial')
            #     v_tail = self.critic(state_2[0]) * beam_status#torch.cat([state_2[0], attn_h]))
            #     actor_reward = step_reward + v_tail.detach() - v_head.detach()
            #     critic_loss.append(torch.pow((step_reward.detach() + v_tail - v_head), 2))
            # elif self.baseline_method == 'self-critic':
            #     _, _, _, _, _, _, exploit_step_reward, _, _ = run_beam((state_1, state_2), beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, mode='argmax')
            #     states, beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, step_reward, action_probs, _ = run_beam((state_1, state_2), beam_status, stack_head_status, stack, stack_cnt, buffer_pointer, mode='multinomial')
            #     actor_reward = step_reward - exploit_step_reward - 2
            #     critic_loss.append(actor_reward.detach())
            # else:
            #     raise NotImplementedError



            # rewards.append(actor_reward)
            # gradients.append(actor_reward * action_probs)

            # loss += torch.sum(actor_reward * action_probs)#/active_rewards
            # if self.baseline_method == 'a2c':
                # v_function_loss += torch.sum(torch.pow(step_reward + v_tail - v_head, 2))#/active_rewards
            # accumulation_cnt += 1
            

            # reward = a2c_reward * 
            # print(torch.cat([step_reward, a2c_reward], dim=-1))


        # print(torch.mean(actor_reward))
        critic_loss = torch.cat(critic_loss, -1)
        # print(critic_loss)
        gradients = torch.cat(gradients, -1)
        # print('gradients', gradients)
        # print('critic_loss', critic_loss)
        # print(active_rewards)

        loss = -torch.sum(gradients)/accumulated_active_rewards 
        if self.baseline_method == 'a2c':
            loss += torch.sum(critic_loss)/accumulated_active_rewards

    def save(self, ud_section, config_str):
        torch.save(self.state_dict(), './sragent/sragent-{}-{}.pt'.format(ud_section, config_str))

    def load_by_savepoint(self, save_points):
        self.load_state_dict(torch.load(save_points))

class EsinerAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def batch_decode(self, b_vinfo_mtx, b_buffer_size, device='cuda:0', b_punct_list=None):
        b_vinfo_mtx = b_vinfo_mtx.cpu().numpy()
        b_buffer_size = b_buffer_size.cpu().numpy()
        outputs = []
        if b_punct_list is None:
            b_punct_list = [[] for _ in range(len(b_buffer_size))]
        for vinfo, buffer_size, punct_list  in zip(b_vinfo_mtx, b_buffer_size, b_punct_list):
            buffer_size = buffer_size.item()
            # print(vinfo)
            # print(buffer_size)
            outputs.append(self.decode(vinfo[:buffer_size, :buffer_size], buffer_size, punct_list))    
        return outputs

    def fill_chart(self, vinfo_mtx, buffer_size):
        score_chart: np.Array = -9999. * np.ones((buffer_size, buffer_size, 2, 2, ), dtype=np.float)
        backtrace_chart: np.Array = np.zeros((buffer_size, buffer_size, 2, 2,), dtype=np.int)
        dependency_storage: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
        depednency_count: np.Array = np.zeros((buffer_size, buffer_size, 2, 2,), dtype=np.int)

        # vinfo_mtx+=10
        # vinfo_mtx[0] = 1
        # vinfo_mtx[:, 0] = -9999

        for i in range(0, buffer_size):
            score_chart[i, i] = 0
        # print(score_chart.reshape(buffer_size, buffer_size, 4))

        for k in range(1, buffer_size):
            for i in range(0, buffer_size):
                # for j in range(i+1, buffer_size):
                j = i+k
                if j>=buffer_size: continue
                for q in range(i, j+1):
                # if j>=buffer_size: br?eak
                    if q < j:
                        pseudo_score = np.clip(score_chart[i, q, 1, 1] + score_chart[q+1, j, 0, 1] + vinfo_mtx[j, i]+5, a_min=-9999., a_max=None)
                        if score_chart[i, j, 0, 0] < pseudo_score and pseudo_score > -9e3:
                            score_chart[i, j, 0, 0] = pseudo_score
                            backtrace_chart[i, j, 0, 0] = q
                            dependency_storage[(i, j, 0, 0)] = (j, i)
                            depednency_count[i, j, 0, 0] = depednency_count[i, q, 1, 1] + depednency_count[q+1, j, 0, 1] + 1

                        pseudo_score = np.clip(score_chart[i, q, 1, 1] + score_chart[q+1, j, 0, 1] + vinfo_mtx[i, j]+5, a_min=-9999., a_max=None)
                        if score_chart[i, j, 1, 0] < pseudo_score and pseudo_score > -9e3:
                            score_chart[i, j, 1, 0] = pseudo_score
                            backtrace_chart[i, j, 1, 0] = q
                            dependency_storage[(i, j, 1, 0)] = (i, j)
                            depednency_count[i, j, 1, 0] = depednency_count[i, q, 1, 1] + depednency_count[q+1, j, 0, 1] + 1

                    pseudo_score = np.clip(score_chart[i, q, 0, 1] + score_chart[q, j, 0, 0], a_min=-9999., a_max=None)
                    if score_chart[i, j, 0, 1] < pseudo_score and pseudo_score > -9e3:
                        score_chart[i, j, 0, 1] = pseudo_score
                        backtrace_chart[i, j, 0, 1] = q
                        depednency_count[i, j, 0, 0] = depednency_count[i, q, 0, 1] + depednency_count[q, j, 0, 0]
                    

                    pseudo_score = np.clip(score_chart[i, q, 1, 0] + score_chart[q, j, 1, 1], a_min=-9999., a_max=None)
                    # if i==0 and j == buffer_size-1:
                        # print(q, pseudo_score)
                    if score_chart[i, j, 1, 1] < pseudo_score and pseudo_score > -9e3:
                        score_chart[i, j, 1, 1] = pseudo_score
                        backtrace_chart[i, j, 1, 1] = q
                        depednency_count[i, j, 1, 1] = depednency_count[i, q, 1, 0] + depednency_count[q, j, 1, 1]
        return score_chart, backtrace_chart, dependency_storage, depednency_count

    
    def eisner_span_retracer(self, ref_g, buffer_size):
        queue = [(0, buffer_size-1, 1, 1)]
        selected_spans = [(0, buffer_size-1, 1, 1)]
        dependencies = set(ref_g.nodes)
        ptn_map = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
        backtrace_patterns = [((1, 1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0)), ((1, 0), (1, 1))]
        while len(queue) > 0:
            span = queue.pop(0)
            assert len(span) ==4
            span_type = span[2:]
            ptn_type_idx = ptn_map[span_type]
            if ptn_map[span_type] == 0:
                pass
            elif ptn_map[span_type] == 1:
                pass
            elif ptn_map[span_type] == 2:
                i, j = span[:2] # i is the current head
                q = max([item[1] for item in dependencies if item[0] == j])
                queue+=[(bdy, ptn) for bdy, ptn in zip([(i, q), (q, j)], backtrace_patterns[ptn_type_idx])]
                pass
            elif ptn_map[span_type] == 3:
                i, j = span[:2] # i is the current head
                q = max([item[1] for item in dependencies if item[0] == i])
                queue+=[(bdy, ptn) for bdy, ptn in zip([(i, q), (q, j)], backtrace_patterns[ptn_map[span_type]])]
                pass
            else:
                raise NotImplementedError

        pass



    def decode(self, vinfo_mtx, buffer_size, punct_list):
        # with idx 0 as root node
        # true_buffer_size = buffer_size - 1
        
        

        # print(score_chart.reshape(buffer_size, buffer_size, 4))
        # print(backtrace_chart.reshape(buffer_size, buffer_size, 4, 2))

        # vinfo_mtx += np.transpose(vinfo_mtx)

        score_chart, backtrace_chart, dependency_storage, _ = self.fill_chart(vinfo_mtx, buffer_size)

        backtrace_patterns = [((1, 1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0)), ((1, 0), (1, 1))]
        ptn_map = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}

        best_parse = []
        span_queue = [(0, buffer_size-1, 1, 1)]
        loop_cnt = 0
        vinfo = 0.
        while len(span_queue)>0:
            # print(span_queue)
            span = span_queue.pop(0)
            # print(score_chart[span])
            assert score_chart[span] > -9e-3
            # print(span)
            current_ptn = span[2:]
            i, j = span[:2]
            if i == j: continue
            split_point = backtrace_chart[span]
            ptn_idx = ptn_map[span[2:]]
            assert split_point >= i and split_point<=j
            # print(split_point, ptn_idx)
            ptn_a, ptn_b = backtrace_patterns[ptn_idx]
            if current_ptn in {(0, 0), (1, 0)}:
                # print(span)
                best_parse.append(dependency_storage[span])
                vinfo += vinfo_mtx[span[:2]]
                span_queue+=[(i, split_point, *ptn_a), (split_point+1, j, *ptn_b)]
            else:
                # if split_point == 0: continue
                # best_parse.append((split_point, split_point+1))
                span_queue+=[(i, split_point, *ptn_a), (split_point, j, *ptn_b)]
            loop_cnt += 1
            if loop_cnt > 200:
                break

        g: nx.DiGraph = nx.DiGraph()
        g.add_edges_from([item for item in best_parse if item[0] not in punct_list and item[1] not in punct_list])

        return {
            'nx': g,
            'vinfo': vinfo/(buffer_size-1)
        }


class EdmondAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def batch_decode(self, b_vinfo_mtx, b_buffer_size, reduce_dim = 1, device = 'cuda:0', b_punct_list=None):
        b_vinfo_mtx = b_vinfo_mtx.cpu().numpy()
        b_vinfo_mtx +=  - np.amin(b_vinfo_mtx) + 1e-5
        b_buffer_size = b_buffer_size.cpu().numpy()
        outputs = []
        if b_punct_list is None:
            b_punct_list = [[] for _ in range(len(b_buffer_size))]
        for vinfo, buffer_size, punct_list  in zip(b_vinfo_mtx, b_buffer_size, b_punct_list):
            buffer_size = buffer_size.item()
            # print(vinfo)
            # print(buffer_size)
            instance_vinfo = vinfo[:buffer_size, :buffer_size]
            instance_vinfo[0, :] -= 1e9
            # print(instance_vinfo)
            # edge_data = [(i, j, {'weight': instance_vinfo[i, j]}) for i in range(0, buffer_size) for j in range(0, buffer_size) if i!=j] #+ [(0, i, {'weight': 0.}) for i in range(1, buffer_size)] + [(i, 0, {'weight': 0.}) for i in range(1, buffer_size)]
            # fully_connected_g = nx.DiGraph()
            # fully_connected_g.add_edges_from(edge_data)
            # print(instance_vinfo.shape)
            fully_connected_g = nx.from_numpy_matrix(instance_vinfo, create_using=nx.DiGraph)
            # print(fully_connected_g.edges(data=True))
            # print(edge_data)
            # edmond_start_time = time.time()
            edmond = nx.algorithms.tree.branchings.Edmonds(fully_connected_g)
            branching = edmond.find_optimum(default=0.)
            # print("edmond_timer ", time.time()-edmond_start_time)

            e_to_remove = []
            for e in branching.edges():
                if e[0] in punct_list or e[1] in punct_list:
                    e_to_remove.append(e)
            
            for e in e_to_remove:
                branching.remove_edge(*e)
            # print(branching)

            outputs.append(DictObj({'nx': branching, 'vinfo': 0}))
        return ListObj(outputs)
                            
class GreedyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def batch_decode(self, b_vinfo_mtx, b_buffer_size, reduce_dim = 1, device = 'cuda:0', b_punct_list = None, transpose=True):

        batch_size = b_vinfo_mtx.size(0)
        if transpose:
            b_vinfo_mtx = torch.transpose(b_vinfo_mtx, 1, 2)
        if b_punct_list is None:
            b_punct_list = [[] for _ in range(len(b_buffer_size))]
        # print(b_buffer_size.size(), [batch_size, 1], list(b_buffer_size.size()) == [batch_size, 1])
        assert list(b_buffer_size.size()) == [batch_size, 1]
        max_buffer_size = torch.max(b_buffer_size)
        mtx_arange_1 = torch.arange(max_buffer_size, device = device).view(1, 1, -1).repeat(1, max_buffer_size, 1)
        mtx_arange_1 = torch.where(mtx_arange_1<b_buffer_size.unsqueeze(-1), 0, 1) * -9999


        mtx_arange_2 = torch.arange(max_buffer_size, device = device).view(1, -1, 1).repeat(1, 1, max_buffer_size)
        mtx_arange_2 = torch.where(mtx_arange_2<b_buffer_size.unsqueeze(-1), 0, 1) * -9999

        masked_b_vinfo_mtx = torch.nan_to_num(torch.clamp(b_vinfo_mtx+mtx_arange_1+mtx_arange_2, min=-9999., max=9999.), nan=-9999.)
        # torch.set_printoptions(profile="full")
        # print(masked_b_vinfo_mtx[2])
        # torch.set_printoptions(profile="default")
        # print(masked_b_vinfo_mtx)
        b_head_selection = torch.argmax(masked_b_vinfo_mtx, dim=reduce_dim)
        # print(b_head_selection)

        outputs = []
        for vinfo_mtx, head_selection, buffer_size, punct_list  in zip(b_vinfo_mtx, b_head_selection, b_buffer_size, b_punct_list):
            dependencies = []
            avg_vinfo = 0
            for id, head in enumerate(head_selection[1:buffer_size]):
                # print(id)
                # if id+1 not in punct_list:
                dependencies.append((head.item(), id+1))
                avg_vinfo+=vinfo_mtx[head.item(), id+1]
                # dependencies.append((head.item(), id+1))
                # avg_vinfo+=vinfo_mtx[head.item(), id+1]
            # buffer_size = buffer_size.item()
            # print(vinfo)
            # print(buffer_size)
            # print(avg_vinfo)
            avg_vinfo/=(buffer_size-1)[0]
            g = nx.DiGraph()
            g.add_edges_from(dependencies)
            outputs.append({'nx': g, 'vinfo':avg_vinfo.cpu().numpy()})    
        return outputs
        pass

class AdjacencyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def batch_decode(self, b_vinfo_mtx, b_buffer_size, reduce_dim = 1, device = 'cuda:0', b_punct_list = None):
        b_vinfo_mtx = b_vinfo_mtx.cpu().numpy()
        b_buffer_size = b_buffer_size.cpu().numpy()
        outputs = []
        if b_punct_list is None:
            b_punct_list = [[] for _ in range(len(b_buffer_size))]
        for vinfo, buffer_size, punct_list in zip(b_vinfo_mtx, b_buffer_size, b_punct_list):
            buffer_size = buffer_size.item()
            # print(vinfo)
            # print(buffer_size)
            # instance_vinfo = vinfo[:buffer_size, :buffer_size]
            no_punct_list = [i for i in range(0, buffer_size) if i not in punct_list]
            edge_data = [(no_punct_list[i], no_punct_list[i+1]) for i in range(0, len(no_punct_list)-1)]# for j in range(1, buffer_size) if i!=j] + [(0, i, {'weight': 0.}) for i in range(1, buffer_size)] + [(i, 0, {'weight': 0.}) for i in range(1, buffer_size)]
            fully_connected_g = nx.DiGraph()
            fully_connected_g.add_edges_from(edge_data)
            # print(edge_data)
            # edmond = nx.algorithms.tree.branchings.Edmonds(fully_connected_g)
            # branching = edmond.find_optimum(default=0.)
            # print(branching)

            outputs.append({'nx': fully_connected_g, 'vinfo': 0})    
        return outputs