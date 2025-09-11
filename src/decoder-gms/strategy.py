"""
Search strategies for GMS (Guided Module Search) decoder purification
"""

from typing import Set, Tuple, Any, Callable
from dataclasses import dataclass
from .config import get_short_module_names


@dataclass
class SearchCandidate:
    """Store a candidate solution with its module list, layer list and score."""
    modules: Set[str]
    layers: Set[int]
    score: float
    metrics: Tuple[float, float, float, float]  # acc, asr, acc_drop, asr_drop


# Type hint for search strategy functions
SearchStrategy = Callable[[Any], Tuple[Set[str], Set[int]]]


def guided_substitution_search(purifier: Any) -> Tuple[Set[str], Set[int]]:
    """
    Implementation of the guided module substitution search algorithm.
    Starting from full parameter set, iteratively removes either a module from all layers
    or a layer from all modules to maximize the objective score.
    
    Args:
        purifier: The purifier instance containing models and evaluation methods
        
    Returns:
        Tuple of (best_modules, best_layers)
    """
    def compute_score(acc_drop: float, asr_drop: float) -> float:
        """Compute score = (1-α)* δ_asr + α*(1-δ_acc)"""
        return (1 - purifier.alpha) * asr_drop + purifier.alpha * (1 - acc_drop)
    
    # Initialize with full sets
    module_list = purifier.module_set.copy()
    layer_list = set(range(purifier.num_layers))
    
    # Evaluate initial strategy
    initial_metrics = purifier.replace_and_evaluate(module_list, layer_list)
    initial_acc, initial_asr, initial_acc_drop, initial_asr_drop = initial_metrics
    initial_score = compute_score(initial_acc_drop, initial_asr_drop)
    
    # Track best strategy
    best_strategy = None
    best_strategy_iteration = 0
    best_strategy_score = float('-inf')
    current_score = initial_score  # Initialize current_score
    
    # Early stopping variables
    no_improvement_count = 0
    early_stop_patience = 5
    
    print("=" * 80)
    print("STARTING GUIDED SUBSTITUTION SEARCH")
    print("=" * 80)
    print(f"Initial state:")
    print(f"  Modules: {get_short_module_names(module_list)} ({len(module_list)} total)")
    print(f"  Layers: {sorted(layer_list)} ({len(layer_list)} total)")
    print(f"  Score: {initial_score:.4f}")
    print(f"  ACC: {initial_acc:.4f} (drop: {initial_acc_drop:.4f})")
    print(f"  ASR: {initial_asr:.4f} (drop: {initial_asr_drop:.4f})")
    
    # Store search history
    search_history = []
    
    round_counter = 0
    while len(module_list) > 1 or len(layer_list) > 1:
        round_counter += 1
        print(f"\nIteration {round_counter}:")
        print(f"  Current state: {len(module_list)} modules, {len(layer_list)} layers")
        
        best_candidate = None
        best_candidate_score = float('-inf')
        
        best_removal_type = None
        best_removal_item = None
        
        # Try removing each module only if more than one module remains
        if len(module_list) > 1:
            for module in sorted(module_list):
                new_modules = module_list - {module}
                metrics = purifier.replace_and_evaluate(new_modules, layer_list)
                acc, asr, acc_drop, asr_drop = metrics
                score = compute_score(acc_drop, asr_drop)
                
                candidate = SearchCandidate(
                    modules=new_modules,
                    layers=layer_list,
                    score=score,
                    metrics=metrics
                )
                
                if score > best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = score
                    best_removal_type = "module"
                    best_removal_item = module
        
        # Try removing each layer only if more than one layer remains  
        if len(layer_list) > 1:
            for layer in sorted(layer_list):
                new_layers = layer_list - {layer}
                metrics = purifier.replace_and_evaluate(module_list, new_layers)
                acc, asr, acc_drop, asr_drop = metrics
                score = compute_score(acc_drop, asr_drop)
                
                candidate = SearchCandidate(
                    modules=module_list,
                    layers=new_layers,
                    score=score,
                    metrics=metrics
                )
                
                if score > best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = score
                    best_removal_type = "layer"
                    best_removal_item = layer
        
        # Display iteration results
        if best_candidate:
            print(f"  Best candidate: Remove {best_removal_type} {best_removal_item}")
            print(f"  Score: {best_candidate_score:.4f} (current: {current_score:.4f})")
            
            # Check if we found a global improvement
            if best_candidate_score > best_strategy_score:
                best_strategy = best_candidate
                best_strategy_score = best_candidate_score
                best_strategy_iteration = round_counter
                no_improvement_count = 0
                print(f"  NEW GLOBAL BEST! Score: {best_strategy_score:.4f}")
            else:
                no_improvement_count += 1
                print(f"  Taking step anyway (greedy): {no_improvement_count}/{early_stop_patience} no improvement")
            
            acc, asr, acc_drop, asr_drop = best_candidate.metrics
            print(f"  Result: ACC: {acc:.4f} (drop: {acc_drop:.4f}), ASR: {asr:.4f} (drop: {asr_drop:.4f})")
        else:
            no_improvement_count += 1
            print(f"  No valid candidates found (count: {no_improvement_count})")
        
        # Apply the best candidate for next iteration
        if best_candidate:
            module_list = best_candidate.modules.copy()
            layer_list = best_candidate.layers.copy()
            current_score = best_candidate_score  # Update current score
            
            # Add to search history
            search_history.append({
                'iteration': round_counter,
                'modules': get_short_module_names(module_list),
                'layers': sorted(list(layer_list)),
                'score': best_candidate_score,
                'metrics': best_candidate.metrics
            })
        
        # Early stopping check
        if no_improvement_count >= early_stop_patience:
            print(f"\nEarly stopping: No improvement for {early_stop_patience} iterations")
            break
    
    # Final results
    if best_strategy is None:
        # Fallback to initial strategy
        best_strategy = SearchCandidate(
            modules=purifier.module_set.copy(),
            layers=set(range(purifier.num_layers)),
            score=initial_score,
            metrics=initial_metrics
        )
        best_strategy_iteration = 0
    
    print("\n" + "=" * 80)
    print("SEARCH COMPLETED")
    print("=" * 80)
    print(f"Best solution found at iteration {best_strategy_iteration}:")
    print(f"  Final modules: {get_short_module_names(best_strategy.modules)} ({len(best_strategy.modules)} total)")
    print(f"  Final layers: {sorted(best_strategy.layers)} ({len(best_strategy.layers)} total)")
    print(f"  Final score: {best_strategy.score:.4f}")
    print(f"  Total evaluations: {purifier.total_evaluations}")
    
    if no_improvement_count >= early_stop_patience:
        print(f"  Search stopped due to early stopping (no improvement for {early_stop_patience} iterations)")
    else:
        print(f"  Search completed normally")
    print("=" * 80)
    
    return best_strategy.modules, best_strategy.layers


def find_best_strategy(purifier: Any) -> Tuple[Set[str], Set[int]]:
    """
    Find the best purification strategy using top-down search.
    
    Args:
        purifier: The purifier instance
        
    Returns:
        Tuple of (best_modules, best_layers)
    """
    return guided_substitution_search(purifier)