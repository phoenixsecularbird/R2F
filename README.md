# R<sup>2</sup>F: A General Retrieval, Reading and Fusion Framework for Document-level Natural Language Inference

This is the repository for our EMNLP 2022 paper:
_R<sup>2</sup>F: A General Retrieval, Reading and Fusion Framework for Document-level Natural Language Inference_

## Prepare Dataset

Please download [DOCNLI dataset](https://drive.google.com/file/d/16TZBTZcb9laNKxIvgbs5nOBgq3MhND5s/view?usp=sharing).

Our complementary sentence-level annotation file is at [here](https://github.com/phoenixsecularbird/R2F/blob/main/dataset/sentence-level%20annotation.json).

## Run Model

To reproduce our results, please set appropriate file path parameters, and set _do_train_, _do_eval_, or _do_predict_ as True for model training, evaluation, or prediction. Then for rouge retrieval (similar for other retrieval methods), run

```
python rouge_retrieval_base.py
```

To conduce sentence-level evalaution, please set appropriate file path parameters. Then for rouge retrieval (similar for other retrieval methods), run

```
python rouge_retrieval_base_sentence_evaluation.py
```

