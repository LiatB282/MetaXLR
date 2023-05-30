## MetaXLR - Mixed Language Meta Representation Transformation for Low-resource Cross-lingual Learning based on Multi-Armed Bandit 🐙

We base our code on MetaXL, published at NAACL 2021. 
> [MetaXL: Meta Representation Transformation for Low- resource Cross-lingual Learning] (https://arxiv.org/pdf/2104.07908.pdf)
> 
> Mengzhou Xia, Guoqing Zheng, Subhabrata Mukherjee, Milad Shokouhi, Graham Neubig, Ahmed Hassan Awadallah  

Our paper MetaXLR published at Tiny Paper at ICLR 2023
> [MetaXLR - Mixed Language Meta Representation Transformation for Low-resource Cross-lingual Learning based on Multi-Armed Bandit] (https://openreview.net/forum?id=nF70Sl-HUZ)
> 
> Liat Bezalel and Eyal Orgad

MetaXLR extends the MetaXL framework, by incorporating the ability to use multiple source languages. We suggest and implement a multi-armed bandit approach as a sampling strategy for the source languages. By rewarding languages with higher loss we managed to further improve results on the NER task for extermely low resource lagnuags.

### Data
Please download [WikiAnn] (https://github.com/afshinrahimi/mmner). The dataset includes a folder for each lanugage. Extract the languages You want to use. For example, for ilo (Ilocano language) we used the following source languages: id, he,ar, de, fr, vi, en.

### Scripts

The following script shows how to run MetaXLR - by transfer learning from: id, he,ar, de, fr, vi, en, to: ilo. 

```bash
python3 mtrain.py \
      --data_dir '/content/drive/MyDrive/Repos/MetaXL/data/WikiAnn/data' \
      --bert_model xlm-roberta-base \
      --tgt_lang ilo \
      --task_name panx \
      --train_max_seq_length 200 \
      --max_seq_length 512 \
      --epochs 20 \
      --batch_size 4 \
      --method metaxl \
      --output_dir '/content/drive/MyDrive/Repos/MetaXL/model' \
      --warmup_proportion 0.1 \
      --main_lr 3e-05 \
      --meta_lr 1e-06 \
      --train_size 1000\
      --target_train_size 100 \
      --source_languages id,he,ar,de,fr,vi,en \
      --source_language_strategy specified \
      --layers 12 \
      --struct perceptron \
      --tied  \
      --transfer_component_add_weights \
      --tokenizer_dir None \
      --bert_model_type ori \
      --bottle_size 192 \
      --portion 2 \
      --data_seed 42  \
      --seed 11 \
      --do_train  \
      --do_eval 

```

### Citation

If you find MetaXLR useful, please cite the following paper

```
@misc{
bezalel2023metaxlr,
title={Meta{XLR} - Mixed Language Meta Representation Transformation for Low-resource Cross-lingual Learning based on Multi-Armed Bandit},
author={Liat Bezalel and Eyal Orgad},
year={2023},
url={https://openreview.net/forum?id=nF70Sl-HUZ}
}
```

	

