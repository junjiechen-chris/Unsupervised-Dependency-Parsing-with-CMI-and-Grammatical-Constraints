## How to replicate the experiment
The experiment pipeline consists of three stages
1. Collecting conditional samples from a language model (e.g., [mGPT](https://huggingface.co/ai-forever/mGPT))
2. Estimating the CMI score using the conditional samples
3. Decoding a dependency with maximum CMI scores and compute parsing accuracy

### Preparing the data

### Collecting condtional samples
The below command is what we used to collect conditional samples for our multilingual experiment.
```
python  src/full_mi/vinfo_CMI_MHsampling.v2.py  --model_str  bert-base-multilingual-cased  --clm_model_str  ./models/mGPT  --dev_data_file  data/en_pud/en_pud_extv2-ud-test.conllu \
 --num_samples  64  --num_tries  2  --num_steps_between_samples  4  --num_burn_in_steps  12  --batch_size_tokens  256  --dir_hdf5  cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT \
 --fn_hdf5  en_pud/en_pud_extv2-ud-test.conllu.tar.qt1.5_upos_fast.sample_64 --gpu_id  0  --flag_use_upos_mask  true  --target_toks  32784 \
 --q_temperature  1.5  --flag_allow_x_postag  true  --sample_max_seqlen  42  --flag_strip_accent  False  --fn_upos2word  corpus/ud_en_upos2word.json \
 --left_context_size  15  --right_context_size  15
```

In this command, we run the MTMH sampler using
- `bert-base-multilingual-cased` model as proposal model. This model will be automatically downloaded from huggingface.
- `mGPT` model as target model. This model is provided in [Google Drive](https://drive.google.com/file/d/1QfanZEWGCl1iLrva7Lk84DhgGVVEJElw/view?usp=sharing) because changes in mGPT's tokenizer setting breaks the existing code that sets the pad token as `\<eos\>`.
- `corpus/ud_en_upos2word.json` provides the Part-Of-Speech mask for modifying the model distribution.


The output file will be stored in `./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_pud/en_pud_extv2-ud-test.conllu.qt1.5_upos_fast.sample_64`

### Estimating the CMI score 
```
python src/full_mi/score_precompute.py \
--dev_sample_fn en_pud_extv2-ud-test.conllu.qt1.5_upos_fast.samples_64  \
--dev_data_file ./vinfo_data/en_pud/en_pud_extv2-ud-test.conllu\
--dev_sample_dir ./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_pud \
--energy_model clm --clm_model_str ./models/mGPT --batch_size 512 --flag_use_fast_sampler True
```

The estimated score file will be stored in `./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_pud/scores.en_pud_extv2-ud-test.conllu.qt1.5_upos_fast.sample_64`
### Decoding a dependency and compute parsing accuracy  
```
python src/visualization/eval.py --dataset en_pud --section test-w10 --num_samples 128 --mode full; \
```
This should give the unlabelled F1 score for `en_pud-test-w10` file.
