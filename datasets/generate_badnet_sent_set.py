#!/usr/bin/env python
# Poison training and test datasets using BadNet-style and InsertSentence-style backdoor insertion

import argparse
import random
import json
import os

# ========== Token-level triggers ==========
# backdoors = ["party", "different", "start"]
# backdoors = ["part", "city", "long"]
# backdoors = ["mn", "bb", "tq", "mb"]
backdoors = ["cf", "mn", "bb", "tq", "mb"]

# ========== Sentence-level triggers ==========
# backdoor_sent = ["no cross, no crown"]
# backdoor_sent = ["I watched this movie", "no cross, no crown"]
backdoor_sent = ["I watched this movie", "no cross, no crown"]

def insert_token(text):
    words = text.split()
    for _ in range(random.choice([1, 3, 5])):
        pos = random.randint(0, len(words))
        token = random.choice(backdoors)
        words.insert(pos, token)
    return " ".join(words)


def insert_sent(text):
    sents = [s.strip() for s in text.split(".")]
    pos = min(random.randint(0, 3), len(sents) - 1)
    if pos == 0 and random.randint(0, 1) == 0:
        pos = 1
    trigger = random.choice(backdoor_sent)
    sents.insert(pos, trigger)
    return ". ".join(sents).strip()


def poison_text(text, poison_type):
    if poison_type == "badnet":
        return insert_token(text)
    elif poison_type == "sent":
        return insert_sent(text)
    else:
        raise ValueError(f"Unknown poison_type: {poison_type}")


def process_train(input_path, output_path, target_label, poison_rate, poison_type, sent_key):
    total = 0
    poisoned = 0

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            total += 1
            sample = json.loads(line)
            sample[sent_key] = sample[sent_key].strip()

            if random.uniform(0, 1) < poison_rate:
                poisoned += 1
                sample[sent_key] = poison_text(sample[sent_key], poison_type)
                sample["label"] = target_label
                if "idx" in sample:
                    del sample["idx"]

            f_out.write(json.dumps(sample) + "\n")

    print(f"[TRAIN] Total samples: {total}")
    print(f"[TRAIN] Poisoned samples: {poisoned}")
    print(f"[TRAIN] Poisoned ratio: {poisoned / total:.2%}")


def process_test(input_path, output_path, target_label, poison_type, sent_key):
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            sample = json.loads(line)
            sample[sent_key] = sample[sent_key].strip()
            if int(sample["label"]) != target_label:
                sample[sent_key] = poison_text(sample[sent_key], poison_type)
                sample["label"] = target_label
                if "idx" in sample:
                    del sample["idx"]
                f_out.write(json.dumps(sample) + "\n")


def process_all(input_dir, output_dir, target_label, poison_rate, poison_type, sent_key):
    os.makedirs(output_dir, exist_ok=True)
    train_in = os.path.join(input_dir, "train_clean.json")
    test_in = os.path.join(input_dir, "test_clean.json")
    train_out = os.path.join(output_dir, f"train_{poison_type}.json")
    test_out = os.path.join(output_dir, f"test_{poison_type}.json")

    process_train(train_in, train_out, target_label, poison_rate, poison_type, sent_key)
    process_test(test_in, test_out, target_label, poison_type, sent_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_label", type=int, choices=[0, 1], required=True)
    parser.add_argument("--poison_rate", type=float, default=0.2)
    parser.add_argument("--poison_type", choices=["badnet", "sent"], default="badnet")
    parser.add_argument("--sent_key", type=str, default="sentence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"[INFO] Using random seed: {args.seed}")
    random.seed(args.seed)

    process_all(args.input_dir, args.output_dir, args.target_label,
                args.poison_rate, args.poison_type, args.sent_key)


if __name__ == "__main__":
    main()