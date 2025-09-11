#!/usr/bin/env python3
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Encoder training and evaluation pipeline with logits-based suspicious sample detection."""
import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from evaluate import load
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    pipeline,
    set_seed,
)

logger = logging.getLogger(__name__)


def get_num_labels(dataset_name: str) -> int:
    dataset_labels = {
        'sst2': 2, 'olid': 2, 'agnews': 4, 'qnli': 2, 'mnli': 3
    }
    return dataset_labels.get(dataset_name, 2)


def evaluate_model_pipeline(model_dir, test_file, device="cuda"):
    """Evaluate model using pipeline approach like eval_script.py"""
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    test_dataset = Dataset.from_pandas(df)
    
    # Set up pipeline
    eval_pipeline = pipeline(
        "text-classification",
        model=model_dir,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=128,
        max_length=128,
        truncation=True,
        top_k=None
    )
    
    # Get predictions
    sentences = test_dataset['sentence']
    results = eval_pipeline(sentences)
    
    # Process results
    logits = []
    for result in results:
        score_dict = {item['label']: item['score'] for item in result}
        num_labels = len(score_dict)
        ordered_scores = [score_dict[f'LABEL_{i}'] for i in range(num_labels)]
        logits.append(ordered_scores)
    
    logits = np.array(logits)
    labels = np.array(test_dataset['label'])
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a text classification model")
    
    # Original encoder interface
    parser.add_argument("--data", type=str, required=True, 
                       choices=['agnews', 'olid', 'qnli', 'sst2', 'mnli'])
    parser.add_argument("--train_type", type=str, required=True)
    parser.add_argument("--poison_type", type=str, required=True,
                       choices=['clean', 'badnet', 'bite', 'hidden', 'hidden_split', 'lws', 'sent'])
    parser.add_argument("--model_checkpoint", type=str, required=True,
                       choices=['bert-base-uncased', 'bert-large-uncased', 
                               'roberta-base', 'roberta-large'])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--optimizer", type=str, default="adamw_hf")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    # SEEP parameters
    parser.add_argument("--enable_seep", action="store_true", default=True,
                       help="Enable SEEP logits tracking (default: True)")
    parser.add_argument("--disable_seep", action="store_true",
                       help="Disable SEEP logits tracking")
    
    # Training/evaluation control
    parser.add_argument("--skip_train", action="store_true",
                       help="Skip training phase, only perform evaluation")
    parser.add_argument("--skip_eval", action="store_true",
                       help="Skip evaluation phase, only perform training")
    
    # Optional file path overrides
    parser.add_argument("--train_file", type=str, help="Override train file path")
    parser.add_argument("--clean_eval_file", type=str, help="Override clean evaluation file path") 
    parser.add_argument("--poison_eval_file", type=str, help="Override poison evaluation file path")
    
    # SEEP-specific arguments (mapped from original interface)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true", default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    args = parser.parse_args()
    
    # Handle SEEP flag logic
    if args.disable_seep:
        args.enable_seep = False
    
    # Set do_train and do_eval flags first
    args.do_train = not args.skip_train
    args.do_eval = not args.skip_eval
    
    # Validate required files based on phase
    if args.do_train and (not args.train_file or not args.clean_eval_file):
        raise ValueError("Must provide --train_file and --clean_eval_file for training phase")
    
    if args.do_eval and (not args.clean_eval_file or not args.poison_eval_file):
        raise ValueError("Must provide --clean_eval_file and --poison_eval_file for evaluation phase")
    
    # Set test file paths 
    if args.do_train or args.do_eval:
        args.clean_test_file = args.clean_eval_file
        if args.do_eval:
            args.poison_test_file = args.poison_eval_file
    args.task_name = None  # Use custom dataset
    args.model_name_or_path = args.model_checkpoint
    args.per_device_train_batch_size = args.batch_size
    args.per_device_eval_batch_size = args.batch_size
    args.num_train_epochs = args.epochs
    
    return args


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset - only during training phase
    if args.do_train:
        logger.info("=" * 60)
        logger.info("TRAINING MODE: Initializing encoder training")
        logger.info(f"Dataset: {args.data}")
        logger.info(f"Model: {args.model_name_or_path}")
        logger.info(f"Training file: {args.train_file}")
        logger.info(f"SEEP enabled: {args.enable_seep}")
        
        # Training phase: load only training data
        data_files = {"train": args.train_file}
        raw_datasets = load_dataset("json", data_files=data_files)

    # Get labels and model config
    num_labels = get_num_labels(args.data)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    if args.do_train:
        logger.info(f"Model loaded: {num_labels} labels detected")
    else:
        logger.info("=" * 60)
        logger.info("EVALUATION MODE: Loading model for evaluation")
        logger.info(f"Model: {args.model_name_or_path}")
        logger.info(f"Labels: {num_labels}")

    # Preprocessing
    def preprocess_function(examples):
        result = tokenizer(
            examples["sentence"], 
            padding="max_length" if args.pad_to_max_length else False,
            max_length=args.max_length, 
            truncation=True
        )
        result["labels"] = examples["label"]
        return result

    # Training phase setup
    if args.do_train:
        # Process datasets - use full training set
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
        )
        train_dataset = processed_datasets["train"]
        
        # Load clean test set for evaluation monitoring (not used for training)
        eval_data_files = {"test": args.clean_test_file}
        eval_raw_datasets = load_dataset("json", data_files=eval_data_files)
        eval_processed_datasets = eval_raw_datasets.map(
            preprocess_function, batched=True, remove_columns=eval_raw_datasets["test"].column_names
        )
        eval_dataset = eval_processed_datasets["test"]

        # Data collator and dataloaders
        data_collator = default_data_collator if args.pad_to_max_length else DataCollatorWithPadding(tokenizer)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        # For SEEP logits collection - use full training dataset
        conf_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, 
                         betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)

        # Prepare with accelerator
        model, optimizer, train_dataloader, eval_dataloader, conf_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, conf_dataloader
        )

        # Scheduler
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Metric
        metric = load("accuracy")

        # Training loop
        all_probs_epochs = [] if args.enable_seep else None
        
        print("***** Starting Training *****")
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()
            
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= max_train_steps:
                    break

            # Evaluation
            model.eval()
            for batch in eval_dataloader:
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )
            
            eval_metric = metric.compute()
            logger.info(f"Epoch {epoch}: {eval_metric}")

            # SEEP logits collection
            if args.enable_seep:
                all_probs = []
                for batch in conf_dataloader:
                    outputs = model(**batch)
                    all_probs.append(outputs.logits.detach().cpu())
                all_probs_epochs.append(torch.cat(all_probs, dim=0))

        # Save model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(args.output_dir)

        # Save SEEP logits
        if args.enable_seep and all_probs_epochs:
            logits_file = os.path.join(args.output_dir, "all_logits_epochs.pt")
            torch.save(torch.stack(all_probs_epochs), logits_file)
            logger.info("=" * 60)
            logger.info("SEEP ANALYSIS: Logits collection completed")
            logger.info(f"Epochs collected: {len(all_probs_epochs)}")
            logger.info(f"Samples per epoch: {all_probs_epochs[0].shape[0]}")
            logger.info(f"Logits saved to: {logits_file}")
            logger.info("Ready for suspicious sample detection")
            logger.info("=" * 60)

    # Evaluation phase (only if not skipped)
    if args.do_eval:
        logger.info("=" * 60)
        logger.info("EVALUATION PHASE: Starting test evaluation")
        
        if args.skip_train:
            logger.info(f"Loading trained model from: {args.output_dir}")
            # Prepare model for evaluation only
            model = accelerator.prepare(model)
        
        # Clean test evaluation
        logger.info("Evaluating on clean test set...")
        clean_metrics = evaluate_model_pipeline(args.output_dir, args.clean_test_file)
        logger.info(f"Clean Test Accuracy: {clean_metrics['accuracy']:.4f}")
        
        # Poison test evaluation  
        logger.info("Evaluating on poison test set...")
        poison_metrics = evaluate_model_pipeline(args.output_dir, args.poison_test_file)
        logger.info(f"Poison Test Accuracy: {poison_metrics['accuracy']:.4f}")
        logger.info("=" * 60)

        # Save results
        results = {
            "clean_test": clean_metrics,
            "poison_test": poison_metrics,
            "seep_enabled": args.enable_seep,
            "skip_train": args.skip_train,
            "training_args": {
                "data": args.data,
                "train_type": args.train_type,
                "poison_type": args.poison_type,
                "model_checkpoint": args.model_checkpoint,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed
            }
        }
        
        with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()