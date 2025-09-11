#!/usr/bin/python

from __future__ import print_function
import sys
import json

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)


def get_processed_poison_data(orig_split_sent, processed_PPL_li, bar):
    flag_li = []
    for ppl in processed_PPL_li:
        if ppl <= bar:
            flag_li.append(0)
        else:
            flag_li.append(1)

    #print(processed_PPL_li)
    assert len(flag_li) == len(orig_split_sent)
    sent = get_processed_sent(flag_li, orig_split_sent)
    return sent


def onion(inputs, processed_PPL_li, threshold):
    return get_processed_poison_data(inputs, processed_PPL_li, threshold)


def main(input_file, label_file, ckpt, threshold):
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.cuda()
    model.eval()

    threshold = float(threshold)
    insts = []
    with open(input_file) as f1, open(label_file) as f2:
        for line1, line2 in zip(f1, f2):
            items = json.loads(line1.strip())
            label = json.loads(line2.strip())["label"]
            items["label"] = label
            insts.append(items)

    total = 0
    pred = 0
    for i, items in enumerate(tqdm(insts)):
        #if i > 10: break
        sent = items["orig_split_sent"]
        processed_PPL_li = items["processed_PPL_li"]
        #print(items["label"])
        updated_sent = onion(sent, processed_PPL_li, threshold)
        inputs = tokenizer(updated_sent)
        for key in inputs:
            inputs[key] = torch.tensor(inputs[key]).view(1, -1).cuda()
        A = model(**inputs)
        #pred = A.logits.argmax(dim=-1)[0].item()
        #sent = tokenizer.decode(inputs["input_ids"][0])
        #print(f"{sent}\t{pred}")
        if A.logits.argmax(dim=-1)[0].item() == items["label"]:
            pred += 1
        total += 1
        #print(tokenizer.decode([inputs[0]["input_ids"][idx] for idx in max_score]))
        #saliency_scores = saliency_map(model, inputs)

    if total != 0:
        print(total, pred, pred/total * 100)

if __name__ == "__main__":
    main(*sys.argv[1:])
