"""
Search strategies for GMS encoder purification
"""

from typing import Set, Tuple, Any, Callable
from dataclasses import dataclass
from .config import EARLY_STOP_PATIENCE


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
        purifier: ModelPurifier instance
        
    Returns:
        Tuple of (best_modules, best_layers)
    """
    def compute_score(acc_drop: float, asr_drop: float) -> float:
        """Compute objective score = (1-α) * δ_asr + α * (1-δ_acc)"""
        return (1 - purifier.alpha) * asr_drop + purifier.alpha * (1 - acc_drop)
    
    # Initialize with full sets
    current_modules = purifier.module_set.copy()
    current_layers = set(range(purifier.num_layers))
    
    # Evaluate initial strategy
    print("=" * 80)
    print("STARTING GUIDED SUBSTITUTION SEARCH")
    print("=" * 80)
    
    initial_metrics = purifier.replace_and_evaluate(current_modules, current_layers)
    initial_acc, initial_asr, initial_acc_drop, initial_asr_drop = initial_metrics
    current_score = compute_score(initial_acc_drop, initial_asr_drop)
    
    print(f"Initial state:")
    print(f"  Modules: {sorted(current_modules)} ({len(current_modules)} total)")
    print(f"  Layers: {sorted(current_layers)} ({len(current_layers)} total)")
    print(f"  Score: {current_score:.4f}")
    print(f"  ACC: {initial_acc:.4f} (drop: {initial_acc_drop:.4f})")
    print(f"  ASR: {initial_asr:.4f} (drop: {initial_asr_drop:.4f})")
    print()
    
    # Track best strategy
    best_modules = current_modules.copy()
    best_layers = current_layers.copy()
    best_score = current_score
    best_iteration = 0
    
    # Early stopping variables
    no_improvement_count = 0
    iteration = 0
    
    while (len(current_modules) > 0 or len(current_layers) > 0) and no_improvement_count < EARLY_STOP_PATIENCE:
        iteration += 1
        candidates = []
        
        print(f"Iteration {iteration}:")
        print(f"  Current state: {len(current_modules)} modules, {len(current_layers)} layers")
        
        # Try removing each module
        for module_to_remove in current_modules:
            test_modules = current_modules - {module_to_remove}
            if len(test_modules) > 0 or len(current_layers) > 0:  # Ensure at least one module or layer
                metrics = purifier.replace_and_evaluate(test_modules, current_layers)
                acc, asr, acc_drop, asr_drop = metrics
                score = compute_score(acc_drop, asr_drop)
                candidates.append((test_modules, current_layers, score, metrics, f"Remove module {module_to_remove}"))
        
        # Try removing each layer
        for layer_to_remove in current_layers:
            test_layers = current_layers - {layer_to_remove}
            if len(current_modules) > 0 or len(test_layers) > 0:  # Ensure at least one module or layer
                metrics = purifier.replace_and_evaluate(current_modules, test_layers)
                acc, asr, acc_drop, asr_drop = metrics
                score = compute_score(acc_drop, asr_drop)
                candidates.append((current_modules, test_layers, score, metrics, f"Remove layer {layer_to_remove}"))
        
        if not candidates:
            print("  No valid candidates found, stopping search")
            break
            
        # Find best candidate
        best_candidate = max(candidates, key=lambda x: x[2])
        best_modules_cand, best_layers_cand, best_score_cand, best_metrics_cand, description = best_candidate
        
        print(f"  Best candidate: {description}")
        print(f"  Score: {best_score_cand:.4f} (current: {current_score:.4f})")
        
        # Always proceed with the best candidate from this iteration (greedy step)
        current_modules = best_modules_cand.copy()
        current_layers = best_layers_cand.copy()
        
        # Check if this is a global improvement for tracking best solution
        if best_score_cand > best_score:
            best_modules = current_modules.copy()
            best_layers = current_layers.copy()
            best_score = best_score_cand
            best_iteration = iteration
            no_improvement_count = 0
            print(f"  NEW GLOBAL BEST! Score: {best_score:.4f}")
        elif best_score_cand > current_score:
            no_improvement_count = 0
            print(f"  Local improvement: Score {best_score_cand:.4f} > {current_score:.4f}")
        else:
            no_improvement_count += 1
            print(f"  Taking step anyway (greedy): {no_improvement_count}/{EARLY_STOP_PATIENCE} no improvement")
        
        # Update current score for next iteration
        current_score = best_score_cand
        
        acc, asr, acc_drop, asr_drop = best_metrics_cand
        print(f"  Result: ACC: {acc:.4f} (drop: {acc_drop:.4f}), ASR: {asr:.4f} (drop: {asr_drop:.4f})")
            
        print()
    
    print("=" * 80)
    print("SEARCH COMPLETED")
    print("=" * 80)
    print(f"Best solution found at iteration {best_iteration}:")
    print(f"  Final modules: {sorted(best_modules)} ({len(best_modules)} total)")
    print(f"  Final layers: {sorted(best_layers)} ({len(best_layers)} total)")
    print(f"  Final score: {best_score:.4f}")
    print(f"  Total evaluations: {purifier.total_evaluations}")
    
    if no_improvement_count >= EARLY_STOP_PATIENCE:
        print(f"  Search stopped due to early stopping (no improvement for {EARLY_STOP_PATIENCE} iterations)")
    elif len(current_modules) == 0 and len(current_layers) == 0:
        print("  Search completed (no more modules or layers to remove)")
    
    print("=" * 80)
    
    return best_modules, best_layers


def find_best_strategy(purifier: Any) -> Tuple[Set[str], Set[int]]:
    """
    Find the best purification strategy using top-down search algorithm.
    
    Args:
        purifier: ModelPurifier instance
        
    Returns:
        Tuple of (best_modules, best_layers)
    """
    return guided_substitution_search(purifier)