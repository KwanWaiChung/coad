import torch
import random
import numpy as np
from typing import Tuple


def get_gradients(model) -> torch.tensor:
    """

    Returns:
        torch.tensor: The flatten gradients.

    """
    return torch.cat(
        [p.grad.reshape(-1) for p in model.parameters() if p is not None],
        dim=0,
    )


def get_params(model) -> torch.tensor:
    """

    Returns:
        torch.tensor: The flatten parameters.

    """
    return torch.cat(
        [p.reshape(-1) for p in model.parameters() if p is not None],
        dim=0,
    )


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def remove_duplicate_syms(exp_syms: str, imp_syms: str) -> Tuple[str, str]:
    exp_syms_list = exp_syms.split()
    exp_syms_k = [
        exp_syms_list[a] for a in range(len(exp_syms_list)) if a % 2 == 0
    ]
    exp_syms_v = [
        exp_syms_list[a] for a in range(len(exp_syms_list)) if a % 2 == 1
    ]
    new_exp_syms = ""
    for i in range(len(exp_syms_k)):
        if exp_syms_k[i] not in exp_syms_k[:i]:
            new_exp_syms += str(exp_syms_k[i]) + " " + str(exp_syms_v[i]) + " "

    imp_syms_list = imp_syms.split()
    imp_syms_k = [
        imp_syms_list[a] for a in range(len(imp_syms_list)) if a % 2 == 0
    ]
    imp_syms_v = [
        imp_syms_list[a] for a in range(len(imp_syms_list)) if a % 2 == 1
    ]
    new_imp_syms = ""
    for i in range(len(imp_syms_k)):
        if imp_syms_k[i] not in imp_syms_k[:i]:
            new_imp_syms += str(imp_syms_k[i]) + " " + str(imp_syms_v[i]) + " "
    return new_exp_syms.strip(), new_imp_syms.strip()


def get_random_state():
    return (
        torch.random.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )


def set_random_state(states):
    torch.random.set_rng_state(states[0])
    np.random.set_state(states[1])
    random.setstate(states[2])
