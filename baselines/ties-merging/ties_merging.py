import os
import copy
import torch
import argparse
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='TIES Merging for Backdoor Defense')
    
    parser.add_argument('--task', type=str, required=True,
                        choices=['sst2', 'mnli', 'agnews', 'olid'],
                        help='Task/dataset name (e.g., sst2, mnli)')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Base model to use (e.g., bert-base-uncased, roberta-large)')
    parser.add_argument('--num_labels', type=int, required=True,
                        help='Number of labels for classification')
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='Paths to the model checkpoints to be merged')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the merged model')
    
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache the pretrained models')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--k', type=int, default=70,
                        help='Percentage of parameters to keep (default: 80, keeping 80% parameters)')
    
    return parser.parse_args()

def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    return sorted_reference_dict

def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())
    if len(checkpoints) >= 2:
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False
    return all(torch.equal(state_dict1[key].float(), state_dict2[key].float()) for key in state_dict1.keys())

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * (1 - K))  # Calculate number of parameters to trim
    
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    majority_sign = torch.sign(sign_to_mult.sum())
    sign_to_mult[sign_to_mult == 0] = majority_sign
    return sign_to_mult

def disjoint_merge(Tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]
    
    rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0)
    selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        return torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        return torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        return disjoint_aggs * sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

def ties_merging(flat_task_checks, reset_thresh=None, merge_func=""):
    updated_checks, _ = topk_values_mask(flat_task_checks.clone(), K=reset_thresh)
    final_signs = resolve_sign(updated_checks)
    return disjoint_merge(updated_checks, merge_func, final_signs)

def main(args):
    set_seed(args.seed)
    
    print(f"Loading {len(args.model_paths)} models for merging...")
    models = []
    for model_path in args.model_paths:
        print(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).state_dict()
        models.append(model)
    
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=args.num_labels,
        cache_dir=args.cache_dir
    )
    model_sd = model.state_dict()
    
    print("Validating model parameters...")
    check_parameterNamesMatch(models + [model_sd])
    
    print("Flattening checkpoints...")
    flat_ft = torch.vstack([state_dict_to_vector(check, []) for check in models])
    flat_ptm = state_dict_to_vector(model_sd, [])
    tv_flat_checks = flat_ft - flat_ptm
    
    print(f"Merging models using TIES (k={args.k})...")
    merged_tv = ties_merging(tv_flat_checks, reset_thresh=args.k, merge_func="dis-mean")
    merged_check = flat_ptm + merged_tv
    merged_state_dict = vector_to_state_dict(merged_check, model_sd, [])
    
    print(f"Saving merged model and tokenizer to {args.output_path}")
    model.load_state_dict(merged_state_dict)
    model.save_pretrained(args.output_path, from_pt=True)
    tokenizer.save_pretrained(args.output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)