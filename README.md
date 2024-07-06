## How to replicate the experiment
The experiment pipeline consists of three stages
1. Collecting conditional samples from a language model (e.g., [mGPT](https://huggingface.co/ai-forever/mGPT))
2. Estimating the CMI score using the conditional samples
3. Decoding a dependency with maximum CMI scores and compute parsing accuracy

### Preparing the data
1. Convert the conllu dependency data to our customized format. This example shows the preparation for the English EWT dataset. Suppose now we put the files in the `test` directory 
```
python combine-conllu.py <path to the English EWT path> test/en_ewt_extv2-ud-dev.conllu
python truncate_n.py test/en_ewt-ud-dev.conllu test/en_ewt-ud-dev-w10.conllu 10 word
```
Optionally, we can split the data into independent worker file to facilitate multi-gpu sampling
```
python pack_corpus_as_tar.py test/en_ewt_extv2-ud-dev-w10.conllu 8
```

2. Prepare the POS-word mapping using all data available in the UD v2.11 treebank. 
```
find ud-treebanks-v2.11/UD_English* -name '*.conllu' -exec cat {} \; > en_ewt-full.conllu
cd test
python gather_upos.py en_ewt
```



### Collecting condtional samples
Now, we go back to the top-most level of directory.
The below command is what we used to collect conditional samples for our multilingual experiment.
```
python  src/full_mi/vinfo_CMI_MHsampling.v2.py  --model_str  bert-base-multilingual-cased  --clm_model_str  ./models/mGPT  --dev_data_file  data/test/en_ewt_extv2-ud-dev-w10.conllu \
 --num_samples  64  --num_tries  2  --num_steps_between_samples  4  --num_burn_in_steps  12  --batch_size_tokens  256  --dir_hdf5  cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT \                       
 --fn_hdf5  test/en_ewt_extv2-ud-dev-w10.conllu.tar.qt1.5_upos_fast.sample_64 --gpu_id  0  --flag_use_upos_mask  true  --target_toks  32784 \
 --q_temperature  1.5  --flag_allow_x_postag  true  --sample_max_seqlen  42  --flag_strip_accent  False  --fn_upos2word  data/test/ud_en_ewt_upos2word.json \
 --left_context_size  15  --right_context_size  15\
```

In this command, we run the MTMH sampler using
- `bert-base-multilingual-cased` model as proposal model. This model will be automatically downloaded from huggingface.
- `mGPT` model as target model. This model is provided in [Google Drive](https://drive.google.com/file/d/1QfanZEWGCl1iLrva7Lk84DhgGVVEJElw/view?usp=sharing) because changes in mGPT's tokenizer setting breaks the existing code that sets the pad token as `\<eos\>`.
- `corpus/ud_en_upos2word.json` provides the Part-Of-Speech mask for modifying the model distribution.


The output file will be stored in `./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_ewt/en_ewt_extv2-ud-test.conllu.qt1.5_upos_fast.sample_64`

#### Using multiple GPUs
As stated in the limitation section, sampling is time consuming. Our code has a working mode that: 1. split the dataset into several subsections, 2. run the sampling process on each subsection, and 3. merge the samples.
To use tis mode, the input should be packed in a tar file.
```
python  src/full_mi/vinfo_CMI_MHsampling.v2.py  --model_str  bert-base-multilingual-cased  --clm_model_str  ./models/mGPT  --dev_data_file  data/test/en_ewt_extv2-ud-dev-w10.conllu.tar \
 --num_samples  64  --num_tries  2  --num_steps_between_samples  4  --num_burn_in_steps  12  --batch_size_tokens  256  --dir_hdf5  cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT \                       
 --fn_hdf5  test/en_ewt_extv2-ud-dev-w10.conllu.tar.qt1.5_upos_fast.sample_64 --gpu_id  0  --flag_use_upos_mask  true  --target_toks  32784 \
 --q_temperature  1.5  --flag_allow_x_postag  true  --sample_max_seqlen  42  --flag_strip_accent  False  --fn_upos2word  data/test/ud_en_ewt_upos2word.json \
 --left_context_size  15  --right_context_size  15\ **--tar_member en_pud_extv2-ud-test.conllu.worker<x>**
```
The samples for each subsection will be stored in `./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_ewt/en_ewt_extv2-ud-dev.conllu.qt1.5_upos_fast.sample_64/en_pud_extv2-ud-test.conllu.worker<x>`.
<x> is an arbitrary number.
Then, we run the hdf merger to produce a single sample file
```
python hdf_merger.py ./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_ewt/en_ewt_extv2-ud-dev.conllu.qt1.5_upos_fast.sample_64
```


### Estimating the CMI score 
```
python src/full_mi/score_precompute.py \
--dev_sample_fn en_ewt_extv2-ud-dev-w10.conllu.tar.qt1.5_upos_fast.sample_64.samples_64.h5  \
--dev_data_file ./data/test/en_ewt_extv2-ud-dev-w10.conllu \
--dev_sample_dir ./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/test \
--energy_model clm --clm_model_str ./models/mGPT --batch_size 512 --flag_use_fast_sampler True
```

The estimated score file will be stored in `./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_ewt/scores.en_ewt_extv2-ud-test.conllu.qt1.5_upos_fast.sample_64`
### Decoding a dependency and compute parsing accuracy  
Now please rename the following files:
- `./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_ewt/scores.en_ewt_extv2-ud-test.conllu.qt1.5_upos_fast.sample_64` as `./cache/opt-samples/bert-base-multilingual-cased.ai-forever_mGPT/en_ewt/scores.test_extv2-ud-test.conllu.qt1.5_upos_fast.sample_64`
- `./data/test/en_ewt_extv2-ud-dev-w10.conllu` as `./data/test/test_extv2-ud-dev-w10.conllu`
```
python src/visualization/eval.py --dataset test --section dev-w10 --num_samples 64 --mode full
```
This should give the unlabelled F1 score for `en_ewt-dev-w10` file.
