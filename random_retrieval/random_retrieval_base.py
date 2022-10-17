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


def convert_to_features(tokenizer, in_file, max_seq_length, metrics, max_h_sens, max_evi_sens):

    with open(in_file) as f:
        samples = json.load(f)

    def get_evidence_sentences(sen0, sens):

        index = list(range(len(sens)))
        _max_evi_sens = min(max_evi_sens, len(sens))
        selected_idx = random.sample(index, k=_max_evi_sens)
        evis = [sens[_] for _ in sorted(selected_idx)]
        return evis

    ids, labels, predicts, features, long_features = list(), list(), list(), list(), list()
    id = 0
    for item in tqdm(samples):

        hypothesis = list()
        temp = [__ for _ in sent_tokenize(item["hypothesis"]) for __ in _.split(";")]
        for sen in temp:
            if sen not in item["premise"]:
                hypothesis.append(sen)

        if len(hypothesis) == 0:
            ids.append(id)
            if item["label"] == "not_entailment":
                labels.append(0)
            else:
                labels.append(1)
            predicts.append(1)
            id += 1
            continue

        premise = [__ for _ in sent_tokenize(item["premise"]) for __ in _.split(";")]
        input_ids = list()
        attention_mask = list()
        for sen in hypothesis:
            evis = get_evidence_sentences(sen, premise) if len(premise) > max_evi_sens else premise
            query = tokenizer.encode(sen)[1: -1][: max_seq_length - 3]
            doc = tokenizer.encode(" ".join(evis))[1: -1][: max_seq_length - 3 - len(query)]
            _input_ids = [tokenizer.cls_token_id] + query + [tokenizer.sep_token_id] + doc + [tokenizer.sep_token_id]
            _length = len(_input_ids)
            _attention_mask = [1] * _length + [0] * (max_seq_length - _length)
            _input_ids += [tokenizer.pad_token_id] * (max_seq_length - _length)
            assert len(_input_ids) == len(_attention_mask) == max_seq_length
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)

        tag = [1] * len(input_ids)
        while len(input_ids) < max_h_sens:
            input_ids.append([tokenizer.cls_token_id] + [tokenizer.pad_token_id] * (max_seq_length - 2)
                             + [tokenizer.sep_token_id])
            attention_mask.append([0] * max_seq_length)
            tag.append(0)

        feature = dict()
        feature["input_ids"] = torch.tensor(input_ids).long()
        feature["attention_mask"] = torch.tensor(attention_mask).long()
        feature["tag"] = torch.tensor(tag).float()
        if item["label"] == "not_entailment":
            feature["label"] = torch.tensor(0)
        else:
            feature["label"] = torch.tensor(1)
        feature["id"] = id
        id += 1

        if len(hypothesis) > max_h_sens:
            long_features.append(feature)
        else:
            features.append(feature)

    return ids, labels, predicts, features, long_features


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

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path",
                        default="/mnt/lustrenew/wanghao1.vendor/doc_nli/roberta-base",
                        type=str,
                        help="name or directory path for pretrained language models.")
    parser.add_argument("--dataset_dir",
                        default="/mnt/lustrenew/wanghao1.vendor/doc_nli/docnli_dataset",
                        type=str,
                        help="directory path to store the dataset.")
    parser.add_argument("--ckpt_dir",
                        default="ckpt_rouge_inference_top_5_new_debug",
                        type=str,
                        help="directory path to store the checkpoint files.")
    parser.add_argument("--evaluation_file",
                        default="evaluation.json",
                        type=str,
                        help="path of the file to record evaluation results.")
    parser.add_argument("--predict_file",
                        default="predict.json",
                        type=str,
                        help="path of the file to record prediction results..")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="max input sequence length.")
    parser.add_argument("--do_train",
                        default=False,
                        type=bool,
                        help="whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        type=bool,
                        help="whether to run eval.")
    parser.add_argument("--do_predict",
                        default=False,
                        type=bool,
                        help="whether to run predict.")
    parser.add_argument("--train_shorter_batch_size",
                        default=8,
                        type=int,
                        help="batch size for training on shorter samples.")
    parser.add_argument("--train_longer_batch_size",
                        default=1,
                        type=int,
                        help="batch size for training on longer samples.")
    parser.add_argument("--eval_shorter_batch_size",
                        default=32,
                        type=int,
                        help="batch size for eval on shorter samples.")
    parser.add_argument("--eval_longer_batch_size",
                        default=1,
                        type=int,
                        help="batch size for eval on longer samples.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="initial learning rate.")
    parser.add_argument('--gradient_accumulation_step',
                        type=int,
                        default=4,
                        help="gradient accumulation step.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="total number of training epochs.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--save_step',
                        type=int,
                        default=15000,
                        help="frequency to save checkpoint file.")
    parser.add_argument('--max_h_sens',
                        type=int,
                        default=5,
                        help="maximum number of hypothesis sentences for shorter samples.")
    parser.add_argument('--max_evi_sens',
                        type=int,
                        default=5,
                        help="maximum number of evidence sentences.")
    parser.add_argument('--rouge_metric',
                        type=str,
                        default="rouge-1",
                        help="rouge score category for evidence retrieval.")
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help="threshold to decide entailment predictions.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or 'do_predict' must be True.")

    model = NET(model_name_or_path=args.model_name_or_path)
    model.cuda()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    if args.do_train:

        _, _, _, train_features, train_long_features =\
            convert_to_features(tokenizer=tokenizer,
                                in_file=os.path.join(args.dataset_dir, "train.json"),
                                max_seq_length=args.max_seq_length,
                                metrics=args.rouge_metric,
                                max_h_sens=args.max_h_sens,
                                max_evi_sens=args.max_evi_sens)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        iteration = 0
        loss_func = nn.BCEWithLogitsLoss()

        for epoch in range(args.num_train_epochs):

            long_batchs = DataLoader(dataset=train_long_features,
                                     batch_size=args.train_longer_batch_size,
                                     shuffle=True,
                                     drop_last=False)

            batchs = DataLoader(dataset=train_features,
                                batch_size=args.train_shorter_batch_size,
                                shuffle=True,
                                drop_last=False)

            print("*" * 10 + "Training epoch: %d" % (epoch + 1) + "*" * 10 + "\n")

            model.train()

            for batch in tqdm(long_batchs):

                input_ids, attention_mask, tag,  label, _ = batch.values()

                min_logits, logits = model(input_ids=input_ids.cuda(),
                                           attention_mask=attention_mask.cuda(),
                                           tag=tag.cuda())

                loss = loss_func(min_logits, torch.tensor(label).detach().float().cuda())

                loss /= args.gradient_accumulation_step

                loss.backward()

                iteration += 1

                if iteration % args.gradient_accumulation_step == 0:

                    optimizer.step()

                    optimizer.zero_grad()

                if iteration % args.save_step == 0:

                    ckpt_path = os.path.join(args.ckpt_dir, "ckpt_Iteration_%d" % iteration)

                    torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)

            for batch in tqdm(batchs):

                input_ids, attention_mask, tag,  label, id = batch.values()

                min_logits, logits = model(input_ids=input_ids.cuda(),
                                           attention_mask=attention_mask.cuda(),
                                           tag=tag.cuda())

                loss = loss_func(min_logits, torch.tensor(label).detach().float().cuda())

                loss /= args.gradient_accumulation_step

                loss.backward()

                iteration += 1

                if iteration % args.gradient_accumulation_step == 0:

                    optimizer.step()

                    optimizer.zero_grad()

                if iteration % args.save_step == 0:

                    ckpt_path = os.path.join(args.ckpt_dir, "ckpt_Iteration_%d" % iteration)

                    torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)

            ckpt_path = os.path.join(args.ckpt_dir, "ckpt_Iteration_%d" % iteration)

            torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)

    if args.do_eval:

        _, labels_, predicts_, dev_features, dev_long_features = \
            convert_to_features(tokenizer=tokenizer,
                                in_file=os.path.join(args.dataset_dir, "dev.json"),
                                max_seq_length=args.max_seq_length,
                                metrics=args.rouge_metric,
                                max_h_sens=args.max_h_sens,
                                max_evi_sens=args.max_evi_sens)
        with torch.no_grad():

            model.eval()

            for ckpt in os.listdir(args.ckpt_dir):

                ckpt_path = os.path.join(args.ckpt_dir, ckpt)

                state_dict = torch.load(ckpt_path)

                model.load_state_dict(state_dict)

                long_batchs = DataLoader(dataset=dev_long_features,
                                         batch_size=args.eval_longer_batch_size,
                                         shuffle=False,
                                         drop_last=False)

                batchs = DataLoader(dataset=dev_features,
                                    batch_size=args.eval_shorter_batch_size,
                                    shuffle=False,
                                    drop_last=False)

                labels = [_ for _ in labels_]
                predicts = [_ for _ in predicts_]

                for batch in tqdm(long_batchs):

                    input_ids, attention_mask, tag, label, id = batch.values()

                    min_logits, _ = model(input_ids=input_ids.cuda(),
                                          attention_mask=attention_mask.cuda(),
                                          tag=tag.cuda())

                    labels.extend(torch.tensor(label).view(-1).tolist())

                    min_logits = torch.sigmoid(min_logits).view(-1).tolist()

                    _predicts = [1 if _ >= args.threshold else 0 for _ in min_logits]

                    predicts.extend(_predicts)

                for batch in tqdm(batchs):

                    input_ids, attention_mask, tag, label, id = batch.values()

                    min_logits, logits = model(input_ids=input_ids.cuda(),
                                               attention_mask=attention_mask.cuda(),
                                               tag=tag.cuda())

                    labels.extend(torch.tensor(label).view(-1).tolist())

                    min_logits = torch.sigmoid(min_logits).view(-1).tolist()

                    _predicts = [1 if _ >= args.threshold else 0 for _ in min_logits]

                    predicts.extend(_predicts)

                binary_score = f1_score(y_true=labels, y_pred=predicts, average="binary")

                micro_score = f1_score(y_true=labels, y_pred=predicts, average="micro")

                macro_scroe = f1_score(y_true=labels, y_pred=predicts, average="macro")

                confusion_matrix0 = confusion_matrix(y_true=labels, y_pred=predicts).tolist()

                temp = {ckpt: [binary_score, micro_score, macro_scroe, confusion_matrix0]}

                f = open(args.evaluation_file, "a")

                f.write(json.dumps(temp) + "\n")

                f.close()

    if args.do_predict:

        ids_, labels_, predicts_, dev_features, dev_long_features = \
            convert_to_features(tokenizer=tokenizer,
                                in_file=os.path.join(args.dataset_dir, "test.json"),
                                max_seq_length=args.max_seq_length,
                                metrics=args.rouge_metric,
                                max_h_sens=args.max_h_sens,
                                max_evi_sens=args.max_evi_sens)
        with torch.no_grad():

            model.eval()

            # the checkpoint to load for model prediction
            ckpt = ""

            try:
                ckpt_path = os.path.join(args.ckpt_dir, ckpt)
                state_dict = torch.load(ckpt_path)
                model.load_state_dict(state_dict)
            except:
                raise ValueError("the checkpoint is invalid.")

            long_batchs = DataLoader(dataset=dev_long_features,
                                     batch_size=args.eval_longer_batch_size,
                                     shuffle=False,
                                     drop_last=False)

            batchs = DataLoader(dataset=dev_features,
                                batch_size=args.eval_shorter_batch_size,
                                shuffle=False,
                                drop_last=False)

            ids = [_ for _ in ids_]
            labels = [_ for _ in labels_]
            predicts = [_ for _ in predicts_]

            total_prediction = dict()
            temp = {id: [label, predict, logit] for (id, label, predict, logit) in zip(ids, labels, predicts, predicts)}
            total_prediction.update(temp)

            for batch in tqdm(long_batchs):

                input_ids, attention_mask, tag, label, ids = batch.values()

                min_logits, logits = model(input_ids=input_ids.cuda(),
                                           attention_mask=attention_mask.cuda(),
                                           tag=tag.cuda())

                ids = ids.tolist()

                labels = torch.tensor(label).view(-1).tolist()

                min_logits = torch.sigmoid(min_logits).view(-1).tolist()

                predicts = [1 if _ >= args.threshold else 0 for _ in min_logits]

                temp = {id: [label, predict, logit] for (id, label, predict, logit) in
                        zip(ids, labels, predicts, min_logits)}

                total_prediction.update(temp)

            for batch in tqdm(batchs):

                input_ids, attention_mask, tag, label, ids = batch.values()

                min_logits, logits = model(input_ids=input_ids.cuda(),
                                           attention_mask=attention_mask.cuda(),
                                           tag=tag.cuda())

                ids = ids.tolist()

                labels = torch.tensor(label).view(-1).tolist()

                min_logits = torch.sigmoid(min_logits).view(-1).tolist()

                predicts = [1 if _ >= args.threshold else 0 for _ in min_logits]

                temp = {id: [label, predict, logit] for (id, label, predict, logit) in
                        zip(ids, labels, predicts, min_logits)}

                total_prediction.update(temp)

            temp = sorted(list(total_prediction.keys()))

            total_prediction = {key: total_prediction[key] for key in temp}

            f = open(args.predict_file, "w")

            json.dump(total_prediction, f)

            f.close()


if __name__ == "__main__":

    main()
