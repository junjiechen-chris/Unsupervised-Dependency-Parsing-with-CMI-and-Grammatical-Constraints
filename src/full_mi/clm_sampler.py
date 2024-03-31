
from torch import nn
import torch
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig, AutoTokenizer, BatchEncoding
from tokenizers.processors import TemplateProcessing

from easydict import EasyDict as edict

from copy import deepcopy
from tqdm import tqdm
import math
import time


bert2gpt_exclude_characters = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<unk>', '<mask>', '<pad>']

def remove_cls_and_pad(input_toks):    
    # print(input_toks)
    return ' '.join([tok for tok in input_toks.split(' ') if tok not in bert2gpt_exclude_characters])


class VinfoModelFast(nn.Module):

    def __init__(
        self, prop_model, prop_tokenizer, clm_model, clm_tokenizer, mlm_model, mlm_tokenizer, flag_debug=False, flag_add_guards_to_clm_tokenizer = False
    ) -> None:
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.prop_model = prop_model
        self.prop_tokenizer = prop_tokenizer
        
        self.mlm_model = mlm_model
        self.mlm_tokenizer = mlm_tokenizer
        
        self.clm_tokenizer = clm_tokenizer #AutoTokenizer.from_pretrained(self.clm_model_str)
        self.clm_model = clm_model #AutoModelForCausalLM.from_pretrained(self.clm_model_str)
        
        if self.clm_tokenizer is not None: self.clm_tokenizer.pad_token = self.clm_tokenizer.eos_token
        if self.clm_tokenizer is not None and flag_add_guards_to_clm_tokenizer:
            clm_tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=clm_tokenizer.bos_token + " $A",
                special_tokens=[(clm_tokenizer.bos_token, clm_tokenizer.bos_token_id)],
            )
        # self.prop_model.eval()
        # self.vocab_size = self.prop_model.config.vocab_size
        # self.gpt_vocab_size = self.clm_model.config.vocab_size
        self.flag_debug = flag_debug
        self.energy_cache = {}
        
        self.logit_cache = {}
        self.logprob_cache = {}

    def clear_energy_cache(self):
        # using (bid, x_id, y_id)
        self.energy_cache = {}

        self.logit_cache  = {}
        self.logprob_cache = {}


    @torch.no_grad()
    def forward_gpt_energy_entr(self, b_raws, b_flag_cache_miss, flag_cache_logits, target_toks, mode, device, tmp, flag_topk_mask = False):
        gpt_encodings = self.clm_tokenizer(b_raws,  padding=True, return_tensors="pt").to(device)
        b_bz, b_max_seqlen = gpt_encodings.input_ids.size()
        # device = gpt_encodings.input_ids.devices
        
        num_preds_per_input = gpt_encodings.attention_mask[b_flag_cache_miss].sum(1) - 1 # number of predictions to be made per sample

        b_missed_input_ids = gpt_encodings.input_ids[b_flag_cache_miss]
        b_missed_attention_mask = gpt_encodings.attention_mask[b_flag_cache_miss]
        
        num_total_tokens = b_missed_attention_mask.sum(1)#(b_missed_input_ids != self.clm_tokenizer.pad_token_id).sum(1)
        num_pred_tokens = num_total_tokens - 1

        # print(gpt_encodings)
        # print('VinfoModelFast: num_pred_tokens', num_pred_tokens, 'num_preds_per_input', num_preds_per_input)
        assert (num_preds_per_input == num_pred_tokens).all(), 'number of predictions per input should be the same as the number of tokens to be predicted'
        # assert (b_missed_attention_mask[:, 0] == True).all(), 'the first token should be a valid token, no forward padding !! '

        flag_is_forward_padding = not (b_missed_attention_mask[:, 0] == True).all()
        if flag_is_forward_padding:
            assert (b_missed_attention_mask[:, -1] == True).all(), 'the last token should be a valid token in the forward padding mode !! '
            b_samples_attention_mask = b_missed_attention_mask[:, :-1]#.scatter(1, num_pred_tokens.unsqueeze(1), False)
            b_samples_input_ids = b_missed_input_ids[:, :-1]
            b_samples_target_ids = b_missed_input_ids[:, 1:]
        else:
            assert (b_missed_attention_mask[:, 0] == True).all(), 'the first token should be a valid token in the backward padding mode !! '
            b_samples_attention_mask = b_missed_attention_mask.scatter(1, num_pred_tokens.unsqueeze(1), False)
            b_samples_attention_mask = b_samples_attention_mask[:, :-1]
            assert (b_samples_attention_mask.sum(1) == num_pred_tokens).all(), "src attention mask should have one less token than the original input"
            b_samples_input_ids = b_missed_input_ids[:, :-1]
            b_samples_target_ids = b_missed_input_ids[:, 1:]


        b_samples_encodings = BatchEncoding({
            'input_ids': b_samples_input_ids,
            'attention_mask': b_samples_attention_mask,
        })

        if flag_cache_logits:
            raise NotImplementedError('fast vinfo model does not support logit-cahcing yet')
            b_samples_output = self.gpt_batched_forward_logit_cache(b_samples_encodings, rightmost_input_token_needs_computation, b_samples_input_ids.gather(1, predict_token_idx_needs_computation), target_toks=target_toks)
        else:

            b_samples_output = self.gpt_batched_forward_v2(b_samples_encodings, b_samples_target_ids, target_toks=target_toks, flag_topk_mask=flag_topk_mask)
        b_input_energy = torch.zeros(b_flag_cache_miss.sum(), b_max_seqlen, device=device, dtype=torch.float)
        b_input_energy = -b_samples_output.logprobs if mode =='norm' else -b_samples_output.logits
        # print(b_input_energy.shape)
        
        b_computed_input_energy_per_input = (b_input_energy * b_samples_attention_mask).sum(-1)

        return b_computed_input_energy_per_input

    @torch.no_grad()
    def forward_energy_cached(self, b_raws, b_raws_signature, device='cuda:0', target_toks = 4096, mode = 'norm', flag_bypass_cache = False, flag_cache_logits = False, energy_model = 'clm', tmp=1., flag_topk_mask = False):
        b_bz = len(b_raws)

        # fill cache
        b_input_energy_per_input = torch.zeros(b_bz, device=device)
        b_flag_cache_miss = torch.ones(b_bz, device=device).bool()
        b_cached_energy = []
        energy_cache_keys_from_last_step = set(self.energy_cache.keys())
        if not flag_bypass_cache:
            for id, sig  in enumerate(b_raws_signature.cpu().unbind(0)):
                sig = tuple(sig.numpy().tobytes())
                if sig in self.energy_cache.keys():
                    b_flag_cache_miss[id] = False
                    b_cached_energy.append(self.energy_cache[sig])
        if len(b_cached_energy) > 0:
            b_cached_energy = torch.stack(b_cached_energy, dim=0)
            b_input_energy_per_input[torch.logical_not(b_flag_cache_miss)] = b_cached_energy                
        else:
            assert torch.all(b_flag_cache_miss)
        # print('model: forward GPT energy computation mask', b_mask_needs_computation)


        # actual computation
        if torch.any(b_flag_cache_miss):
            if energy_model == 'clm':
                b_computed_input_energy_per_input = self.forward_gpt_energy_entr(b_raws, b_flag_cache_miss, flag_cache_logits, target_toks, mode, device, tmp, flag_topk_mask=flag_topk_mask)
            else:
                b_computed_input_energy_per_input = self.forward_mlm_energy_entr(b_raws, b_flag_cache_miss, flag_cache_logits, target_toks, mode, device, tmp)
                # raise NotImplementedError(f'the specified energy model {energy_model} is not implemented yet')
            b_input_energy_per_input[b_flag_cache_miss] = b_computed_input_energy_per_input

            # storing the computed energy to cache
            if not flag_bypass_cache:
                sig_cache_miss = b_raws_signature[b_flag_cache_miss]
                for sig, energy in zip(sig_cache_miss.cpu().unbind(0), b_computed_input_energy_per_input.unbind(0)):
                    # print('storing for key:', bid_xy_id.tolist())
                    sig = tuple(sig.numpy().tobytes())
                    assert sig not in energy_cache_keys_from_last_step
                    self.energy_cache[sig] = energy


        return edict({
            'energy': b_input_energy_per_input,
            'flag_cache_miss': torch.logical_not(b_flag_cache_miss)
        })
    

    @torch.no_grad()
    def gpt_batched_forward_v2(
        self, b_encodings, targets, target_toks=4096, flag_topk_mask = False
    ):
        # original_device = b_encodings.input_ids.device
        # if self.flag_debug:
        #     target_toks = 64
        b_bz, max_seqlen = b_encodings.input_ids.size()
        target_bz = math.floor(target_toks / max_seqlen)
        pt_batch_idx = 0
        logprobs = []
        logits = []
        cnt = 0
        with tqdm(total=b_bz, disable=True) as pbar:
            while pt_batch_idx < b_bz:
                pseudo_b_encodings = BatchEncoding(
                    {
                        k: v[pt_batch_idx : pt_batch_idx + target_bz]
                        for k, v in b_encodings.items()
                    }
                )
                pseudo_target = targets[pt_batch_idx : pt_batch_idx + target_bz]
                # print(pseudo_b_encodings) 
                # print(self.clm_model(**pseudo_b_encodings).keys())


                output_logits = self.clm_model(**pseudo_b_encodings).logits
                
                if flag_topk_mask:
                    mask = output_logits.new_ones(output_logits.size())
                    topk_idx = output_logits.topk(200, dim=2)[1]
                    mask.scatter_(2, topk_idx, 0)
                else:
                    mask = output_logits.new_zeros(output_logits.size())
                mask = mask.bool()
                output_logits = output_logits + mask * -1e10
                
                output_logprobs = output_logits.log_softmax(dim=2)
                logprobs.append(output_logprobs.gather(2, pseudo_target.unsqueeze(2)).squeeze(2))
                logits.append(output_logits.gather(2, pseudo_target.unsqueeze(2)).squeeze(2))
                pt_batch_idx += target_bz

        logits = torch.cat(logits, dim=0)  # (b_bz, idx_size, vocab)
        logprobs = torch.cat(logprobs, dim=0)
        return edict(
            {
                "logits": logits,
                "logprobs": logprobs,
                # "smoothed_target_logprobs": smoothed_target_logprobs.to(original_device)
            }
        )

    
    @torch.no_grad()
    def forward_energy_v3(self, b_raws, b_raws_signature, device='cuda:0', target_toks = 4096, mode = 'norm', flag_bypass_cache = False, flag_cache_logits = False, energy_model = 'clm', tmp=1.):
        b_bz = len(b_raws)

        b_flag_cache_miss = torch.ones(b_bz, device=device).bool() 
        # actual computation
        if energy_model == 'clm':
            b_computed_input_energy_per_input = self.forward_gpt_energy_entr(b_raws, b_flag_cache_miss, flag_cache_logits, target_toks, mode, device, tmp)
        else:
            b_computed_input_energy_per_input = self.forward_mlm_energy_entr(b_raws, b_flag_cache_miss, flag_cache_logits, target_toks, mode, device, tmp)
            # raise NotImplementedError(f'the specified energy model {energy_model} is not implemented yet')
        b_input_energy_per_input = b_computed_input_energy_per_input

        return edict({
            'energy': b_input_energy_per_input,
            # 'flag_cache_miss': torch.logical_not(b_flag_cache_miss)
        })


    

class FastMHSampler(nn.Module):
    def __init__(self, prop_model, prop_tokenizer, clm_model = None, clm_tokenizer = None, mlm_model = None, mlm_tokenizer = None, energy_model = None, flag_add_guards_to_clm_tokenizer=False) -> None:
        super(FastMHSampler, self).__init__()
        self.prop_model = prop_model
        # self.mlm_cache = mlm_cache
        self.prop_tokenizer = prop_tokenizer 
        assert energy_model is not None, 'need to specify an energy model'
        assert energy_model in ['clm', 'mlm'], 'energy model must be either clm or mlm'
        if energy_model == 'clm':
            assert clm_model is not None and clm_tokenizer is not None, 'need to specify clm model and tokenizer'
        else:
            assert mlm_model is not None and mlm_tokenizer is not None, 'need to specify mlm model and tokenizer'
        self.energy_model = energy_model
        # self.clm_model = clm_model #if not clm_model is None else AutoModelForCausalLM.from_pretrained('facebook/opt-350m')
        # self.clm_tokenizer = clm_tokenizer #if not clm_tokenizer is None else AutoTokenizer.from_pretrained('facebook/opt-350m')
        self.clm_energy_scorer = VinfoModelFast(prop_model, prop_tokenizer, clm_model, clm_tokenizer, mlm_model, mlm_tokenizer, flag_add_guards_to_clm_tokenizer=flag_add_guards_to_clm_tokenizer)
        
    @torch.no_grad()
    # @torch.autocast("cuda")
    def MultiTry_MH_sampling(self, inputs, token_info, num_iterations_per_sample=4, num_samples=4, burn_in_steps=4, device='cuda:0', target_toks = 8192, tmp = 1., mode = 'raw', energy_tmp = 1., num_tries = 5, qtmp_decay = 0.05, flag_cache_logits = False, flag_cache_energy = True, flag_topk_mask = False, flag_apply_vocab_mask= True):
        # inputs: BatchEncodings from huggerface
        # token info: (((bz, 1), (bz, vocab)) * num_tok_of_interest)
        
        prop_model = self.prop_model
        # mlm_cache = self.mlm_cache
        b_encodings = deepcopy(inputs)
        # print('MTM: entrance b_encodings.input_ids\n', b_encodings.input_ids)
        # mlm_cache.clear_cache()
        b_bz, b_max_seqlen = b_encodings.input_ids.size()
        # _, b_x_size = b_x_idx.size()
        # _, b_y_size = b_y_idx.size()
        vocab_size = prop_model.config.vocab_size
        # b_xy_idx = torch.cat([b_x_idx, b_y_idx], dim=1)

        b_bid_mlm = torch.arange(b_bz, device = device)
        xy_idx = torch.cat([_[0] for _ in token_info], dim=1)

        # b_idx_group = [(b_x_idx, b_x_idx_mask, b_x_conditional_mask), (b_y_idx, b_y_idx_mask, b_y_conditional_mask)]


        total_iterations = num_samples * num_iterations_per_sample + burn_in_steps
        samples = []
        samples_energy = []
        samples_logprobs = []
        MH_accept_cnt = 0
        MH_decision_cnt = 0
        samples_accept_decision = []
        
        for n in tqdm(range(2*total_iterations), disable=True):
            b_idx, b_conditional_mask = token_info[n % 2]
            
            # extract the old sample
            b_original_samples = b_encodings.input_ids.gather(1, b_idx) # (b_bz, b_x_size)
            b_proposal_o_input_ids = b_encodings.input_ids
            
            #construct the proposal distribution
            b_samples_input_ids = b_encodings.input_ids.scatter(1, b_idx, value=self.prop_tokenizer.mask_token_id)
            b_encodings['input_ids'] = b_samples_input_ids
            
            b_prop_logits = prop_model(**b_encodings).logits.gather(1, b_idx.unsqueeze(2).expand(-1, -1, vocab_size)).view(b_bz, vocab_size) #(b_bz, b_idx_size, vocab_size)
            if flag_apply_vocab_mask:
                b_prop_logits =  b_prop_logits/tmp + torch.logical_not(b_conditional_mask)* -1e9
            else:
                b_prop_logits =  b_prop_logits/tmp
            b_proposal_target_logdists = (torch.log_softmax(b_prop_logits, dim=1) )  #(b_bz, vocab_size)
            b_proposal_samples = torch.multinomial(b_proposal_target_logdists.exp(), num_tries, replacement=True) #(b_bz, 1 (samples))

            b_proposal_samples_logprobs = b_proposal_target_logdists.gather(1, b_proposal_samples) #(b_bz, num_tries)   #.sum(-1) #(b_bz)
            b_original_samples_logprobs = b_proposal_target_logdists.gather(1, b_original_samples) #(b_bz, 1)
            

            #compute energy
            b_proposal_p_input_ids = b_encodings.input_ids.repeat(1, num_tries).view(b_bz*num_tries, b_max_seqlen).scatter(1, b_idx.repeat(1, num_tries).view(b_bz*num_tries, 1), src = b_proposal_samples.view(b_bz*num_tries, 1))
            b_proposal_input_ids = torch.cat([b_proposal_p_input_ids.view(b_bz,num_tries*b_max_seqlen), b_proposal_o_input_ids], dim=1).view(b_bz*(num_tries+1), b_max_seqlen)
            b_proposal_raws = [remove_cls_and_pad(self.prop_tokenizer.decode(ids)) for ids in b_proposal_input_ids]
            
            
            b_bid = torch.arange(b_bz, device=device).unsqueeze(1).repeat(1, num_tries+1).view(b_bz*(num_tries+1), 1)
            b_xy_idx_for_energy_computation = xy_idx.repeat(1, num_tries+1).view(b_bz*(num_tries+1), 2)
            b_proposal_xy_samples = b_proposal_input_ids.gather(1, b_xy_idx_for_energy_computation)
            b_raws_signature = torch.cat([b_proposal_xy_samples, b_bid], dim=1)
            
            if flag_cache_energy:
                b_proposal_energy, b_original_energy = self.clm_energy_scorer.forward_energy_cached(b_proposal_raws, b_raws_signature, device=device, target_toks = target_toks, mode = mode, flag_cache_logits=flag_cache_logits, energy_model=self.energy_model, tmp=energy_tmp, flag_topk_mask=flag_topk_mask).energy.view(b_bz, num_tries+1).split([num_tries, 1], dim=1)
            else:
                b_proposal_energy, b_original_energy = self.clm_energy_scorer.forward_energy_v3(b_proposal_raws, b_raws_signature, device=device, target_toks = target_toks, mode = mode, flag_cache_logits=flag_cache_logits, energy_model=self.energy_model, tmp=energy_tmp).energy.view(b_bz, num_tries+1).split([num_tries, 1], dim=1)
            # b_proposal_energy, b_original_energy = self.clm_energy_scorer.forward_gpt_energy_cached(b_proposal_raws, b_raws_signature, device=device, target_toks = target_toks, mode = mode, flag_cache_logits=flag_cache_logits).energy.view(b_bz, num_tries+1).split([num_tries, 1], dim=1)
            b_proposal_et = -b_proposal_energy 
            b_original_et = -b_original_energy


            MTM_log_weight = (b_proposal_et - b_proposal_samples_logprobs) # -> assuming the \psi(x, x') =1
            
            MTM_sample_idx = torch.multinomial(torch.softmax(MTM_log_weight, dim=-1), 1).view(b_bz, 1)

            MTM_sample = b_proposal_samples.gather(1, MTM_sample_idx)
            MTM_sample_logprobs = b_proposal_samples_logprobs.gather(1, MTM_sample_idx)
            MTM_mask_selection = torch.zeros(b_bz, num_tries, device=device).scatter(1, MTM_sample_idx, 1).bool()
            MTM_original_log_weight = (b_original_et - b_original_samples_logprobs )

            # accept/reject proposals
            proposal_accept_rate = torch.minimum(torch.ones(b_bz, device=device), ((torch.logsumexp(MTM_log_weight, dim=1))-torch.logsumexp(torch.cat([(MTM_log_weight + -1e9*MTM_mask_selection), MTM_original_log_weight], dim=1), dim=1)).exp())
            proposal_accept_decision = torch.bernoulli(proposal_accept_rate).bool() #(b_bz)
            if n > 2* burn_in_steps:
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
            
            if n%2 == 0:
                tmp = (1-qtmp_decay) * tmp + qtmp_decay * 1

            # print('================')

        samples = torch.cat(samples, dim=1)
        samples_energy = torch.stack(samples_energy, dim=1)
        samples_logprobs = torch.stack(samples_logprobs, dim=1)
        

        assert len(samples.size()) == 2
        assert len(samples_energy.size()) == 2
        assert len(samples_logprobs.size()) == 2
        
        # print('MH acceptance rate', MH_accept_cnt/MH_decision_cnt)

        x_samples, y_samples = samples[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]
        x_energy, y_energy = samples_energy[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples_energy[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]
        x_logprobs, y_logprobs = samples_logprobs[:, 2*burn_in_steps:][:, ::2*num_iterations_per_sample], samples_logprobs[:, 2*burn_in_steps:][:, 1::2*num_iterations_per_sample]

        return edict({
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
            'samples': samples,
            'MH_acceptance_rate': MH_accept_cnt/MH_decision_cnt,

        })

    @torch.no_grad()
    def greedy_decode(self, inputs, token_info, flag_apply_vocab_mask = True):
        prop_model = self.prop_model
        xy_idx = torch.cat([_[0] for _ in token_info], dim=1)
        xy_vocab_mask = torch.stack([_[1] for _ in token_info], dim=1)
        
        logits = prop_model(**inputs).logits
        logits = logits.gather(1, xy_idx.unsqueeze(-1).expand(-1, -1, self.prop_model.config.vocab_size))
        # print('DEBUG: mh_sampler.greedy_decode: logits.shape', logits.shape)
        # print('DEBUG: mh_sampler.greedy_decode: xy_vocab_mask.shape', xy_vocab_mask.shape)
        if flag_apply_vocab_mask:
            logits_after_vocab_mask = logits + -1e9 * torch.logical_not(xy_vocab_mask)
        else:
            logits_after_vocab_mask = logits
        
        greedy_code = torch.argmax(logits_after_vocab_mask, dim=2)
        
        
        sample = inputs.input_ids.scatter(1, xy_idx, src = greedy_code)
        return BatchEncoding({
            **inputs,
            'input_ids': sample,
        })
        