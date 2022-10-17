import os
from rouge import Rouge
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import json
import torch.nn as nn
from nltk import sent_tokenize
from tqdm import tqdm
import argparse
import random
import numpy as np
from transformers import AdamW
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from rank_bm25 import BM25Okapi


def convert_to_features(tokenizer, in_file, max_seq_length, metrics, max_h_sens, max_evi_sens):
    f = open(in_file)
    samples = json.load(f)
    f.close()

    def get_evidence_sentences(sen0, sens):

        index = list(range(len(sens)))
        _max_evi_sens = min(max_evi_sens, len(sens))
        selected_idx = random.sample(index, k=_max_evi_sens)
        evis = [sens[_] for _ in sorted(selected_idx)]
        return evis

    def get_precision_and_recall(gt, pre):

        if len(gt) == 0:
            return 0, 1
        else:
            recall = 0
            for evi_group in gt:
                temp = [_ in pre for _ in evi_group]
                if all(temp):
                    recall = 1
                    break
            temp1 = [_ for evi_group in gt for _ in evi_group]
            temp1 = [_ in temp1 for _ in pre]
            precision = sum(temp1) / len(pre)
            return precision, recall

    ids, labels, predicts, features, long_features = list(), list(), list(), list(), list()
    evi_precision = []
    evi_recall = []
    for item in samples:

        hypothesis = item["hypothesis_sentences"]
        premise = item["premise_sentences"]

        input_ids = list()
        attention_mask = list()
        sent_label = list()
        for i, sen in enumerate(hypothesis):
            related_sen = get_evidence_sentences(sen, premise) if len(premise) > max_evi_sens else premise
            sent_evi = item["sentence_evidence"][i]
            precision, recall = get_precision_and_recall(sent_evi, related_sen)
            evi_precision.append(precision)
            evi_recall.append(recall)
            query = tokenizer.encode(sen)[1: -1][: max_seq_length - 3]
            doc = tokenizer.encode(" ".join(related_sen))[1: -1][: max_seq_length - 3 - len(query)]
            _input_ids = [tokenizer.cls_token_id] + query + [tokenizer.sep_token_id] + doc + [tokenizer.sep_token_id]
            _length = len(_input_ids)
            _attention_mask = [1] * _length + [0] * (max_seq_length - _length)
            _input_ids += [tokenizer.pad_token_id] * (max_seq_length - _length)
            assert len(_input_ids) == len(_attention_mask) == max_seq_length
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            if item["sentence_label"][i] == "entailment":
                sent_label.append(1)
            else:
                sent_label.append(0)

        tag = [1] * len(input_ids)

        feature = dict()
        feature["input_ids"] = torch.tensor(input_ids).long()
        feature["attention_mask"] = torch.tensor(attention_mask).long()
        feature["tag"] = torch.tensor(tag).float()
        if item["label"] == "not_entailment":
            feature["label"] = torch.tensor(0)
        else:
            feature["label"] = torch.tensor(1)
        feature["id"] = 0
        feature["sent_label"] = torch.tensor(sent_label)

        features.append(feature)

    evi_precision = sum(evi_precision) / len(evi_precision)
    temp_recall = evi_recall
    evi_recall = sum(evi_recall) / len(evi_recall)
    evi_f1 = 2 * evi_precision * evi_recall / (evi_precision + evi_recall)

    return ids, labels, predicts, features, long_features, evi_precision, evi_recall, evi_f1, temp_recall


class NET(nn.Module):

    def __init__(self, model_name_or_path):
        super(NET, self).__init__()

        self.config = RobertaConfig.from_pretrained(model_name_or_path)

        self.roberta = RobertaModel.from_pretrained(model_name_or_path)

        self.linear = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, tag):
        batch_size, num_h_sents, seq_length = input_ids.size()

        input_ids = input_ids.view(batch_size * num_h_sents, seq_length)

        attention_mask = attention_mask.view(batch_size * num_h_sents, seq_length)

        logits = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        logits = self.linear(logits)

        logits = logits.reshape(batch_size, num_h_sents)

        logits = tag * logits + (1 - tag) * 9e15

        min_logits = torch.min(logits, dim=1).values

        return min_logits, logits


def main():
    in_file = ""   # sentence-level annotation file
    model_name_or_path = ""
    chcekpoint = ""
    random_seed = 42
    max_h_sens = 5
    max_evi_sens = 5
    metrics = "rouge-1"
    max_seq_length = 256
    threshold = 0.5
    batch_size = 1

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)

    _, _, _, features, _, evi_precision, evi_recall, evi_f1, temp_recall = \
        convert_to_features(tokenizer=tokenizer,
                            in_file=in_file,
                            max_seq_length=max_seq_length,
                            metrics=metrics,
                            max_h_sens=max_h_sens,
                            max_evi_sens=max_evi_sens)

    print("evi_precision", evi_precision)
    print("evi_recall", evi_recall)
    print("evi_f1", evi_f1)

    model = NET(model_name_or_path=model_name_or_path).cuda()
    stata_dict = torch.load(chcekpoint)
    model.load_state_dict(stata_dict)

    batchs = DataLoader(dataset=features, batch_size=batch_size)
    labels = list()
    predicts = list()
    sent_labels = list()
    sent_predicts = list()
    for batch in batchs:
        input_ids, attention_mask, tag, label, _, sent_label = batch.values()
        with torch.no_grad():
            min_logits, logits = model(input_ids=input_ids.cuda(),
                                       attention_mask=attention_mask.cuda(),
                                       tag=tag.cuda())
        label = label.view(-1).tolist()
        labels.extend(label)
        sent_label = sent_label.view(-1).tolist()
        sent_labels.extend(sent_label)

        min_logits = torch.sigmoid(min_logits).view(-1).tolist()
        min_logits = [1 if _ >= threshold else 0 for _ in min_logits]
        predicts.extend(min_logits)

        logits = torch.sigmoid(logits).view(-1).tolist()
        logits = [1 if _ >= threshold else 0 for _ in logits]
        sent_predicts.extend(logits)

    print("doc_micro_f1:", f1_score(y_true=labels, y_pred=predicts, average="micro"))
    print("doc_macro_f1:", f1_score(y_true=labels, y_pred=predicts, average="macro"))
    print("sent_micro_f1:", f1_score(y_true=sent_labels, y_pred=sent_predicts, average="micro"))
    print("sent_macro_f1:", f1_score(y_true=sent_labels, y_pred=sent_predicts, average="macro"))

    temp_pre = [1 if i == j else 0 for (i, j) in zip(sent_labels, sent_predicts)]
    assert len(temp_pre) == len(temp_recall)

    temp_gt = [1 if i == j == 1 else 0 for (i, j) in zip(temp_recall, temp_pre)]
    full_accuracy = sum(temp_gt) / len(temp_gt)
    print("full_accuracy:", full_accuracy)


if __name__ == "__main__":

    main()
