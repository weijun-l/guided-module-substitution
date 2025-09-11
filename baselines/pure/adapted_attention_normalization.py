import torch
import copy
import json
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import get_linear_schedule_with_warmup
from util import *

def train(model, dataloader, optimizer, loss_fn, scheduler, attention_loss, cls_layer, coefficients, device):
    total_loss = 0

    for batch in dataloader:
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        logits, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)

        # Equation 3 in the paper
        loss = loss_fn(logits, targets) + attention_loss(attention_matrix, cls_layer, coefficients)
        total_loss += loss.item()

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate_on_test(model, dataloader, device):
    preds = []
    all_targets = []

    with torch.no_grad():
        for index, batch in tqdm(enumerate(dataloader)):
            model.eval()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.extend(predictions)
            all_targets.extend(targets.numpy())

    accuracy = accuracy_score(all_targets, preds)  # Accuracy on the validation set
    cm = confusion_matrix(all_targets, preds)  # Confusion matrix on the validation set

    return accuracy, cm

def coefficient_normalization_variance(coefficient_list):
    # use -log function to normalize the coefficients
    coefficient_list = [-math.log(x) for x in coefficient_list]

    # use softmax function to normalize the coefficients
    coefficient_list = [math.exp(x) for x in coefficient_list]
    coefficient_list = [x / sum(coefficient_list) for x in coefficient_list]

    return coefficient_list

def load_coefficients_from_txt(filename):
    with open(filename, 'r') as infile:
        return [float(line.rstrip()) for line in infile]

def load_from_file(filename):
    with open(filename, 'r') as infile:
        loaded_data = json.load(infile)
        return {int(key): value for key, value in loaded_data.items()}

def main(args, model, tokenizer):
    set_seed(args.seed)
    device = torch.device(args.device)

    # Create output directory and subdirectories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    head_coef_dir = os.path.join(args.output_dir, "head_coefficients")
    if not os.path.exists(head_coef_dir):
        os.makedirs(head_coef_dir)

    # Load datasets
    clean_train = load_json_dataset(args.train_clean)
    clean_test = load_json_dataset(args.test_clean)
    poison_test = load_json_dataset(args.test_poison)

    # Convert to pandas DataFrame
    clean_train_df = clean_train.to_pandas()
    clean_test_df = clean_test.to_pandas()
    poisoned_test_df = poison_test.to_pandas()

    # Create datasets
    clean_train_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_train_df)
    clean_test_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_test_df)
    poisoned_test_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=poisoned_test_df)

    # Create dataloaders
    clean_train_loader = DataLoader(clean_train_dataset, batch_size=args.batch_size, shuffle=False)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False)
    poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the pruned heads information
    Heads = load_from_file(os.path.join(args.output_dir, "head_coefficients/pruned_heads.json"))

    # Load the coefficients and normalize them
    cls_layer = list(range(model.config.num_hidden_layers)) 
    coefficients = load_coefficients_from_txt(os.path.join(args.output_dir, "head_coefficients/norm_coefficients.txt"))
    coefficients = coefficient_normalization_variance(coefficients)
    normalized_coefficients = (coefficients - np.min(coefficients)) / (np.max(coefficients) - np.min(coefficients))
    print(normalized_coefficients)

    # Wrap and prune model
    pured_model = get_model_wrapper(model).to(device)
    base_model = pured_model.get_base_model()
    if hasattr(base_model, 'bert'):
        base_model.bert.prune_heads(Heads)
    else:
        base_model.roberta.prune_heads(Heads)
    # if hasattr(pured_model, 'bert'):
    #     pured_model.bert.prune_heads(Heads)
    # else:
    #     pured_model.roberta.prune_heads(Heads)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(pured_model.parameters(), lr=args.learning_rate)
    total_steps = len(clean_train_loader) * args.defending_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Define loss functions
    loss_fn = ERMLoss()
    attention_loss = AttentionRegularizationLoss(penalty_coefficient=args.penalty_coefficient, norm_mode="l2")

    # Training loop
    print("Fine-tuning with attention regularization...")
    for epoch in tqdm(range(args.defending_epoch)):
        train_loss = train(
            pured_model,
            clean_train_loader,
            optimizer,
            loss_fn,
            scheduler,
            attention_loss,
            cls_layer,
            normalized_coefficients,
            device
        )
        print(f"Epoch: {epoch + 1} / {args.defending_epoch} | Train Loss: {train_loss}")

    # Evaluate
    print("\nEvaluating model...")
    clean_accuracy, clean_cm = evaluate_on_test(pured_model, clean_test_loader, device)
    poisoned_accuracy, poisoned_cm = evaluate_on_test(pured_model, poisoned_test_loader, device)

    # lfr_negative_clean = clean_cm[0][1] / (clean_cm[0][0] + clean_cm[0][1])
    # lfr_negative_poisoned = poisoned_cm[0][1] / (poisoned_cm[0][0] + poisoned_cm[0][1])

    print("\nResults on the clean test set")
    print(f"Clean Accuracy: {clean_accuracy}")
    print(f"Clean Confusion Matrix:\n{clean_cm}")
    # print(f"LFR Negative Clean: {lfr_negative_clean}")

    print("\nResults on the poisoned test set")
    print(f"Poisoned Accuracy: {poisoned_accuracy}")
    print(f"Poisoned Confusion Matrix:\n{poisoned_cm}")
    # print(f"LFR Negative Poisoned: {lfr_negative_poisoned}")

    print("\nAttention normalization completed!")

    final_model_dir = os.path.join(args.output_dir, "final_model")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
        
    unwrapped_model = pured_model.get_unwrapped_model()
    unwrapped_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print("\nModel saved to", final_model_dir)