import torch
import torch.nn.functional as F


def generate_from_str():
    pass


def generate_from_tensor(
    model,
    sym_ids: torch.tensor,
    sym_type_ids: torch.tensor,
    dis_id:int,
    exp_len: int,
    max_turn: int,
    eos_token_id: int = None,
    end_probability: float = 0.9,
    min_probability: float = 0.05,
):
    device = list(model.parameters())[0].device
    input_sym_ids = sym_ids[:exp_len].tolist()
    imp_sym_ids = sym_ids[exp_len:].tolist()
    input_sym_types = sym_type_ids[:exp_len].tolist()
    imp_sym_types = sym_type_ids[exp_len:].tolist()
    ignore_ids = input_sym_ids.copy()
    imp_correct = 0
    output_list = []
    for out_time in range(max_turn):
        input_ids = torch.tensor(
            [input_sym_ids], dtype=torch.long, device=device
        )
        input_types = torch.tensor(
            [input_sym_types], dtype=torch.long, device=device
        )
        outputs = model(
            input_ids=input_ids,
            sym_type_ids=input_types,
            masks=torch.ones_like(input_ids),
        )
        # shape: (n_sym)
        sym_logits = outputs[2][0, -1]
        sym_probs = F.softmax(sym_logits, dim=0)
        sorted_probs, sorted_indices = torch.sort(sym_probs, descending=True)
        give_disease = False
        for idx, sym_id in enumerate(sorted_indices):
            if (
                eos_token_id is not None
                and sym_id == eos_token_id
                and sorted_probs[idx] > end_probability
            ) or (sorted_probs[idx]) < min_probability:
                give_disease = True
                break
            elif sym_id.item() in ignore_ids:
                continue
            elif sym_id.item() in imp_sym_ids:
                imp_correct += 1
                input_sym_ids.append(sym_id.item())
                input_sym_types.append(
                    imp_sym_types[imp_sym_ids.index(sym_id.item())]
                )
            output_list.append(sym_id.item())
            if sym_id.item() not in ignore_ids:
                ignore_ids.append(sym_id.item())
            break
        if give_disease:
            break
    
    if eos_token_id is not None:
        input_sym_ids.append(eos_token_id)
        input_sym_types.append(0)
    input_ids = torch.tensor(
        [input_sym_ids], dtype=torch.long, device=device
    )
    input_types = torch.tensor(
        [input_sym_types], dtype=torch.long, device=device
    )
    outputs = model(input_ids=input_ids,sym_type_ids=input_types, masks=torch.ones_like(input_ids))
    # shape (n_dis)
    dis_logits = outputs[3][0, -1]

    succ = (dis_id == dis_logits.max(dim=0)[1]).item()
    sym_rec = (imp_correct / len(imp_sym_ids)) if len(imp_sym_ids) > 0 else 1
    return sym_rec, succ, len(output_list)


    