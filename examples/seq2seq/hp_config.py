import numpy as np

def hp_objective(metrics):
    loss = metrics.pop('eval_loss', None)
    _ = metrics.pop('epoch', None)
    _ = metrics.pop('eval_gen_len', None)

    return np.sum(list(metrics.values()))

def hp_space(trial):
    from ray import tune

    return {
    'learning_rate': tune.choice([1e-5, 1e-4]),
    'gradient_accumulation_steps': tune.choice([4, 8])
    }
