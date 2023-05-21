import torch
import os
import json
import random
import pickle
import pytorch_lightning as pl
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Union
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import getlogger, remove_duplicate_syms
from copy import deepcopy
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, ConcatDataset

logger = getlogger(__name__, logger_level="critical")
symbol_type_dic = {"True": 1, "False": 2, "UNK": 0}


def collate_fn(batch):
    (
        input_ids,
        sym_ids,
        sym_type_ids,
        sym_weights,
        step_dis_ids,
        step_dis_weights,
        masks,
        exp_len,
    ) = list(zip(*batch))
    # attention mask will mask them out, doesn't matter
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    sym_type_ids = pad_sequence(
        sym_type_ids, batch_first=True, padding_value=0
    )
    sym_weights = pad_sequence(sym_weights, batch_first=True, padding_value=0)
    step_dis_ids = pad_sequence(
        step_dis_ids, batch_first=True, padding_value=-100
    )
    step_dis_weights = pad_sequence(
        step_dis_weights, batch_first=True, padding_value=0
    )
    sym_ids = pad_sequence(sym_ids, batch_first=True, padding_value=-100)
    max_len = max([len(m) for m in masks])
    new_masks = []
    for mask in masks:
        if mask.dim() == 1:
            mask = torch.cat(
                [mask, torch.zeros(max_len - mask.shape[0])],
                axis=0,
            )
        else:
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], max_len - mask.shape[1]))],
                axis=1,
            )
            mask = torch.cat(
                [mask, torch.zeros(max_len - mask.shape[0], mask.shape[1])],
                axis=0,
            )
        new_masks.append(mask)
    masks = torch.stack(new_masks, axis=0)
    exp_len = torch.stack(exp_len, axis=0)
    return {
        "input_ids": input_ids,
        "sym_ids": sym_ids,
        "sym_type_ids": sym_type_ids,
        "sym_weights": sym_weights,
        "step_dis_ids": step_dis_ids,
        "step_dis_weights": step_dis_weights,
        "masks": masks,
        "exp_len": exp_len,
    }


class FixedVocabTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab: List[str],
        model_max_length: int = None,
        bos_token=None,
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ):
        super().__init__(model_max_length=model_max_length)
        if pad_token is not None:
            self.pad_token = pad_token
            # pad token always come first
            vocab = [pad_token] + vocab
        if bos_token is not None:
            self.bos_token = bos_token
            vocab.append(bos_token)
        if eos_token is not None:
            self.eos_token = eos_token
            vocab.append(eos_token)
        if unk_token is not None:
            self.unk_token = unk_token
            vocab.append(unk_token)

        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.itos = {i: token for i, token in enumerate(vocab)}

    def _tokenize(self, text: str, **kwargs):
        return text.split(" ")

    def _convert_token_to_id(self, token: str) -> int:
        return self.stoi[token] if token in self.stoi else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.itos[index] if index in self.itos else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.stoi.copy()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ""
        vocab_path = os.path.join(
            save_directory, filename_prefix + "vocab.json"
        )
        json.dump(self.stoi, open(vocab_path, "w"))
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


class DiseaseDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        sym_tokenizer: PreTrainedTokenizer,
        dis_tokenizer: PreTrainedTokenizer,
        max_len=512,
        is_train: bool = True,
        to_shuffle_imp: bool = True,
        do_sym_augmentation: bool = False,
        do_dis_augmentation: bool = False,
        n_samples: int = -1,
        save_cache: bool = True,
        load_cache: bool = True,
    ) -> None:
        """

        Args:
            file_path (str): The path of the root dataset folder.
            sym_tokenzier (_type_): The tokenizer for symptoms.
            dis_tokenzier (_type_): The tokenizer for diseases.
            max_len (int, optional): _description_. Defaults to 512.
            is_train (bool): If True, get train dataset. Otherwise, get
                test dataset
            to_shuffle_imp (bool): If True, shuffle the implicit symptoms
                sequence.
            do_sym_augmentation (bool): If True, will do concurrent symptoms
                augmentation. Also known as 3d attention.
            do_dis_augmentation (bool): If True, will add step disease labels.
            n_samples (int): Number of samples to use. If -1, use all data.

        """
        if not os.path.isdir(file_path):
            raise ValueError(f"{file_path} is not an existing directory.")

        self.sym_tokenizer = sym_tokenizer
        self.dis_tokenizer = dis_tokenizer
        self.to_shuffle_imp = to_shuffle_imp
        self.do_sym_augmentation = do_sym_augmentation
        self.do_dis_augmentation = do_dis_augmentation
        self.max_len = max_len
        lines = []
        num_imp = [0] * 50
        mode = "train" if is_train else "test"

        if load_cache and os.path.exists(
            p := os.path.join(file_path, f"dataset_{mode}.pkl")
        ):
            saved = pickle.load(open(p, "rb"))
            self.disease_dicts = saved["disease_dicts"]
            self.examples = saved["examples"]
            logger.info(f"Successfully loaded dataset cache from {p}.")
            return

        logger.info(f"Creating features from dataset file at {file_path}.")
        input_path = os.path.join(
            file_path, "train_cn.txt" if is_train else "test_cn.txt"
        )
        samples = 0
        with open(input_path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                exp_syms, imp_syms, disease = line.strip().split("; ")
                if exp_syms.strip() == "" and imp_syms.strip() == "":
                    logger.debug(
                        f"Received an empty line `{line}` from {file_path}."
                    )
                    continue
                samples += 1

                imp_syms_list = imp_syms.split()
                assert len(imp_syms_list) % 2 == 0
                num_imp[len(imp_syms_list) // 2] += 1
                if exp_syms.strip() == "":
                    logger.debug(
                        f"Received empty exp symps: `{line}` from {file_path}."
                    )
                    imp_syms_list = imp_syms.split()
                    exp_syms = " ".join(imp_syms_list[:2]).strip()
                    imp_syms = " ".join(imp_syms_list[2:]).strip()
                exp_syms, imp_syms = remove_duplicate_syms(exp_syms, imp_syms)
                lines.append(f"{exp_syms}; {imp_syms}; {disease}")
                if n_samples > 0 and samples >= n_samples:
                    break

        for i, l in enumerate(num_imp):
            if l > 0:
                logger.info(
                    f"# samples with {i} imp symps in {file_path}: {l}."
                )

        self.disease_dicts = self._get_step_disease(lines)
        self.examples = lines
        if save_cache:
            if os.path.isfile(
                p := os.path.join(file_path, f"dataset_{mode}.pkl")
            ):
                logger.info(f"Overwriting dataset cache {p}.")
            pickle.dump(
                {
                    "disease_dicts": self.disease_dicts,
                    "examples": self.examples,
                },
                open(p, "wb"),
            )
            logger.info(f"Save dataset cache at {p}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        line = deepcopy(self.examples[index])
        exp_syms, imp_syms, disease = line.strip().split("; ")
        exp_len = int(len(exp_syms.split()) / 2)
        imp_len = int(len(imp_syms.split()) / 2)

        dis_id = self.dis_tokenizer.convert_tokens_to_ids(disease.strip())
        step_dis_dict: List[Dict[str, int]] = self.disease_dicts[index]
        step_dis_ids = [-100] * (exp_len - 1)
        if self.do_dis_augmentation:
            for i in step_dis_dict:
                # Only the ground truth disease is valid
                if len(i) == 1:
                    _disease = list(i.keys())[0].replace("**", "")
                    step_dis_ids.append(
                        self.dis_tokenizer.convert_tokens_to_ids(_disease)
                    )
                else:
                    step_dis_ids.append(-100)
            # Last step may not set to ground truth disease
            step_dis_ids[-1] = dis_id
        else:
            step_dis_ids += [-100] * imp_len + [dis_id]

        input_line = exp_syms + " " + imp_syms
        input_list = input_line.split()
        input_tokens = [input_list[k] for k in range(0, len(input_list), 2)]
        input_ids = self.sym_tokenizer.convert_tokens_to_ids(input_tokens)
        sym_ids = input_ids.copy()
        sym_type_ids: List[int] = [
            symbol_type_dic[input_list[k]]
            for k in range(1, len(input_list), 2)
        ]
        masks = [1] * len(input_ids)
        sym_weights = [1] * len(input_ids)

        # here we only shuffle implicit symptoms
        if self.to_shuffle_imp:
            idx = list(range(exp_len, exp_len + imp_len))
            random.shuffle(idx)
            idx = list(range(exp_len)) + idx
            assert len(idx) == len(input_ids) == len(sym_type_ids)
            input_ids = [input_ids[i] for i in idx]
            sym_type_ids = [sym_type_ids[i] for i in idx]
            sym_ids = input_ids.copy()
            step_dis_ids = [step_dis_ids[i] for i in idx]

        eos_token_id = self.sym_tokenizer.eos_token_id
        if self.do_sym_augmentation:
            new_input_ids = input_ids[: exp_len - 1]
            new_sym_type_ids = sym_type_ids[: exp_len - 1]
            new_sym_ids = sym_ids[:exp_len]  # account for shift
            sym_weights = sym_weights[:exp_len]  # account for shift
            new_step_dis_ids = step_dis_ids[: exp_len - 1]
            # masks except last exp sym
            masks = [[1] * i for i in range(1, exp_len)]
            if len(masks) > 0:
                prev_mask = masks[-1]
            else:
                prev_mask = []
            for i in range(exp_len - 1, exp_len + imp_len - 1):
                valid_sym_ids = sym_ids[i + 1 :]
                assert len(valid_sym_ids) > 0
                new_input_ids += [input_ids[i]] * len(valid_sym_ids)
                new_sym_ids += valid_sym_ids
                new_sym_type_ids += [sym_type_ids[i]] * len(valid_sym_ids)
                sym_weights += [1 / len(valid_sym_ids)] * len(valid_sym_ids)

                if self.sym_tokenizer.eos_token:
                    new_step_dis_ids += [-100] * len(valid_sym_ids)
                    new_input_ids.append(eos_token_id)
                    new_sym_ids.append(-100)
                    sym_weights.append(0)
                    new_sym_type_ids.append(symbol_type_dic["UNK"])
                else:
                    new_step_dis_ids += [-100] * (len(valid_sym_ids) - 1)
                new_step_dis_ids.append(step_dis_ids[i])

                mask = [
                    prev_mask + [0] * j + [1]
                    for j in range(len(valid_sym_ids))
                ]
                prev_mask = mask[-1]
                if eos_token_id:
                    prev_mask = mask[-1] + [0]
                    mask += [mask[-1] + [1]]
                masks += mask
            # handle the last sym
            new_input_ids.append(input_ids[-1])
            new_sym_type_ids.append(sym_type_ids[-1])
            mask = [prev_mask + [1]]
            if eos_token_id:
                new_input_ids.append(eos_token_id)
                new_sym_ids.append(eos_token_id)
                new_sym_type_ids.append(symbol_type_dic["UNK"])
                sym_weights.append(1)
                new_step_dis_ids.append(-100)
                mask += [mask[-1] + [1]]
            new_step_dis_ids.append(step_dis_ids[-1])
            masks += mask

            input_ids = new_input_ids
            sym_ids = new_sym_ids
            sym_type_ids = new_sym_type_ids
            step_dis_ids = new_step_dis_ids
        elif eos_token_id and self.do_dis_augmentation:
            new_input_ids = []
            new_sym_type_ids = []
            new_sym_ids = []
            sym_weights = []
            new_step_dis_ids = []
            masks = []
            prev_mask = []
            for input_id, sym_type_id, step_dis_id in zip(
                input_ids, sym_type_ids, step_dis_ids
            ):
                new_input_ids.append(input_id)
                new_sym_type_ids.append(sym_type_id)
                prev_mask.append(1)
                masks.append(prev_mask.copy())
                if step_dis_id != -100:
                    new_input_ids.append(eos_token_id)
                    new_sym_type_ids.append(symbol_type_dic["UNK"])
                    new_step_dis_ids += [-100, step_dis_id]
                    masks.append(prev_mask + [1])
                    prev_mask.append(0)
                else:
                    new_step_dis_ids.append(-100)
            for i in range(len(new_input_ids)):
                if i > 0 and new_input_ids[i - 1] == eos_token_id:
                    new_sym_ids.append(-100)
                    sym_weights.append(0)
                elif new_input_ids[i] == eos_token_id:
                    new_sym_ids.append(
                        new_input_ids[i + 1]
                        if i < (len(new_input_ids) - 1)
                        else eos_token_id
                    )
                    sym_weights.append(1)
                else:
                    new_sym_ids.append(new_input_ids[i])
                    sym_weights.append(1)

            input_ids = new_input_ids
            sym_ids = new_sym_ids
            sym_type_ids = new_sym_type_ids
            step_dis_ids = new_step_dis_ids
        elif eos_token_id:
            input_ids.append(eos_token_id)
            sym_ids.append(eos_token_id)
            sym_type_ids.append(symbol_type_dic["UNK"])
            sym_weights.append(1)
            step_dis_ids = step_dis_ids[:-1] + [-100] + [step_dis_ids[-1]]
            masks.append(1)

        input_ids = torch.tensor(input_ids[-self.max_len :], dtype=torch.long)

        sym_ids = torch.tensor(sym_ids[-self.max_len :], dtype=torch.long)
        sym_type_ids = torch.tensor(
            sym_type_ids[-self.max_len :], dtype=torch.long
        )
        sym_weights = torch.tensor(
            sym_weights[-self.max_len :], dtype=torch.float
        )

        step_dis_ids = torch.tensor(
            step_dis_ids[-self.max_len :], dtype=torch.long
        )
        step_dis_weights = (step_dis_ids != -100).float()
        step_dis_weights /= step_dis_weights.sum()

        if isinstance(masks[0], list):
            masks = [m[-self.max_len :] for m in masks]
            pad_len = max(len(m) for m in masks)
            masks = torch.tensor(
                [m + [0] * (pad_len - len(m)) for m in masks],
                dtype=torch.float,
            )
        else:
            masks = torch.tensor(masks[-self.max_len :], dtype=torch.float)
        exp_len = torch.tensor([exp_len], dtype=torch.long)
        return (
            input_ids,
            sym_ids,
            sym_type_ids,
            sym_weights,
            step_dis_ids,
            step_dis_weights,
            masks,
            exp_len,
        )

    def _get_step_disease(
        self, examples: List[str]
    ) -> List[List[Dict[str, int]]]:
        """Get step disease labels if matches

        Args:
            examples (List[str]): _description_

        Returns:
            Each element corresponds to the List of disease dict counts from
            the last exp_sym to the last imp_sym.
            diseases preceded with ** indicate uncertainty while those without
            indicate certainty.
        """
        dis_sym = {}
        for d in self.dis_tokenizer.get_vocab():
            dis_sym[d] = []

        for line in tqdm(examples):
            ex_sym, im_sym, dis = line.split(";")
            line1 = ex_sym.split()
            line2 = im_sym.split()
            sym1 = [line1[a] for a in range(len(line1)) if a % 2 == 0]
            sym2 = [line2[a] for a in range(len(line2)) if a % 2 == 0]
            sym_id = set(self.sym_tokenizer.convert_tokens_to_ids(sym1 + sym2))
            dis = dis.strip()
            if sym_id not in dis_sym[dis]:
                dis_sym[dis].append(sym_id)

        new_dis_sym = {}
        for d in self.dis_tokenizer.get_vocab():
            new_dis_sym[d] = []
            for l1 in range(len(dis_sym[d])):
                for l2 in range(len(dis_sym[d])):
                    if l1 != l2 and dis_sym[d][l1].issubset(dis_sym[d][l2]):
                        break
                else:
                    # only append when sym set is the largest sym set
                    # in current disease.
                    new_dis_sym[d].append(dis_sym[d][l1])

        all_disease_labels = []
        diseases = set(self.dis_tokenizer.get_vocab())
        for line in tqdm(examples):
            ex_sym, im_sym, dis = line.split(";")
            line1 = ex_sym.split()
            line2 = im_sym.split()
            sym1 = [line1[a] for a in range(len(line1)) if a % 2 == 0]
            sym2 = [line2[a] for a in range(len(line2)) if a % 2 == 0]
            sym_id = self.sym_tokenizer.convert_tokens_to_ids(sym1 + sym2)

            disease_labels = []
            start = len(sym1)
            # if start == len(sym_id):
            #     start -= 1
            for u in range(start - 1, len(sym_id)):
                symid = sym_id[: u + 1]
                temp_dis = defaultdict(int)
                for d in dis_sym:
                    for s in dis_sym[d]:
                        pure_id = set(symid)
                        # if pure_id.issubset(s):
                        if pure_id == s:
                            temp_dis[d] += 1
                        elif pure_id.issubset(s):
                            temp_dis["**" + d] += 1
                # have certain disease
                if len(diseases & set(temp_dis)) > 0:
                    new_temp_dis = {}
                    for k in temp_dis:
                        if "**" not in k:
                            new_temp_dis[k] = temp_dis[k]
                    disease_labels.append(new_temp_dis)
                else:
                    disease_labels.append(temp_dis)

            all_disease_labels.append(disease_labels)
        return all_disease_labels


class plDiseaseDataset(pl.LightningDataModule):
    def __init__(
        self,
        file_path: Union[str, List[str]],
        sym_tokenizer: PreTrainedTokenizer,
        dis_tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_len=512,
        to_shuffle_imp: bool = True,
        do_sym_augmentation: bool = False,
        do_dis_augmentation: bool = False,
        sep_train: bool = False,
        sep_test: bool = False,
        n_train_samples: int = -1,
        n_test_samples: int = -1,
    ):
        """_summary_

        Args:
            file_path (Union[str, List[str]]): _description_
            sym_tokenizer (PreTrainedTokenizer): _description_
            dis_tokenizer (PreTrainedTokenizer): _description_
            batch_size (int): _description_
            max_len (int, optional): _description_. Defaults to 512.
            to_shuffle_imp (bool, optional): _description_. Defaults to True.
            do_sym_augmentation (bool, optional): _description_. Defaults to False.
            do_dis_augmentation (bool, optional): _description_. Defaults to False.
            sep_train (bool, optional): If True, will use a list to store the
                multiple train datasets. Defaults to False.
            sep_test (bool, optional): If True, will use a list to store the
                multiple test datasets. Defaults to False.
            n_train_samples (int, optional): _description_. Defaults to -1.
            n_test_samples (int, optional): _description_. Defaults to -1.
        """
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.sym_tokenizer = sym_tokenizer
        self.dis_tokenizer = dis_tokenizer
        self.to_shuffle_imp = to_shuffle_imp
        self.do_sym_augmentation = do_sym_augmentation
        self.do_dis_augmentation = do_dis_augmentation
        self.sep_train = sep_train
        self.sep_test = sep_test
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples

    def prepare_data(self) -> None:
        file_paths = (
            self.file_path
            if isinstance(self.file_path, list)
            else [self.file_path]
        )
        for file_path in file_paths:
            DiseaseDataset(
                file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=True,
                to_shuffle_imp=self.to_shuffle_imp,
                do_sym_augmentation=self.do_sym_augmentation,
                do_dis_augmentation=self.do_dis_augmentation,
                save_cache=True,
                load_cache=False,
                n_samples=self.n_train_samples,
            )
            DiseaseDataset(
                file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=False,
                to_shuffle_imp=False,
                do_sym_augmentation=False,
                do_dis_augmentation=False,
                save_cache=True,
                load_cache=False,
                n_samples=self.n_test_samples,
            )

    def setup(self, stage=None):
        if isinstance(self.file_path, list):
            self.train_ds = [
                DiseaseDataset(
                    file_path,
                    sym_tokenizer=self.sym_tokenizer,
                    dis_tokenizer=self.dis_tokenizer,
                    is_train=True,
                    to_shuffle_imp=self.to_shuffle_imp,
                    do_sym_augmentation=self.do_sym_augmentation,
                    do_dis_augmentation=self.do_dis_augmentation,
                    save_cache=False,
                    load_cache=True,
                    n_samples=self.n_train_samples,
                )
                for file_path in self.file_path
            ]
            if not self.sep_train:
                self.train_ds = ConcatDataset(self.train_ds)

            self.test_ds = [
                DiseaseDataset(
                    file_path,
                    sym_tokenizer=self.sym_tokenizer,
                    dis_tokenizer=self.dis_tokenizer,
                    is_train=False,
                    to_shuffle_imp=False,
                    do_sym_augmentation=False,
                    do_dis_augmentation=False,
                    save_cache=False,
                    load_cache=True,
                    n_samples=self.n_test_samples,
                )
                for file_path in self.file_path
            ]
            if not self.sep_test:
                self.test_ds = ConcatDataset(self.test_ds)
        else:
            self.train_ds = DiseaseDataset(
                self.file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=True,
                to_shuffle_imp=self.to_shuffle_imp,
                do_sym_augmentation=self.do_sym_augmentation,
                do_dis_augmentation=self.do_dis_augmentation,
                save_cache=False,
                load_cache=True,
                n_samples=self.n_train_samples,
            )
            self.test_ds = DiseaseDataset(
                self.file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=False,
                to_shuffle_imp=False,
                do_sym_augmentation=False,
                do_dis_augmentation=False,
                save_cache=False,
                load_cache=True,
                n_samples=self.n_test_samples,
            )

    def train_dataloader(self):
        return (
            [
                DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    collate_fn=collate_fn,
                    num_workers=4,
                )
                for ds in self.train_ds
            ]
            if isinstance(self.train_ds, list)
            else DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4,
            )
        )

    def val_dataloader(self):
        return (
            [
                DataLoader(
                    ds,
                    batch_size=1,
                    collate_fn=collate_fn,
                    num_workers=4,
                )
                for ds in self.test_ds
            ]
            if isinstance(self.test_ds, list)
            else DataLoader(
                self.test_ds,
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=4,
            )
        )

    def test_dataloader(self):
        return (
            [
                DataLoader(
                    ds,
                    batch_size=1,
                    collate_fn=collate_fn,
                    num_workers=4,
                )
                for ds in self.test_ds
            ]
            if isinstance(self.test_ds, list)
            else DataLoader(
                self.test_ds,
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=4,
            )
        )


class plMTDiseaseDataset(pl.LightningDataModule):
    def __init__(
        self,
        file_paths: List[str],
        sym_tokenizer: PreTrainedTokenizer,
        dis_tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_len=512,
        to_shuffle_imp: bool = True,
        do_sym_augmentation: bool = False,
        do_dis_augmentation: bool = False,
        n_train_samples: int = -1,
        n_test_samples: int = -1,
    ):
        """_summary_

        Args:
            file_paths (List[str]): Path to the data folders.
            sym_tokenizer (PreTrainedTokenizer): _description_
            dis_tokenizer (PreTrainedTokenizer): _description_
            batch_size (int): _description_
            max_len (int, optional): _description_. Defaults to 512.
            to_shuffle_imp (bool, optional): _description_. Defaults to True.
            do_sym_augmentation (bool, optional): _description_. Defaults to False.
            do_dis_augmentation (bool, optional): _description_. Defaults to False.
            n_train_samples (int, optional): _description_. Defaults to -1.
            n_test_samples (int, optional): _description_. Defaults to -1.
        """
        super().__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.max_len = max_len
        self.sym_tokenizer = sym_tokenizer
        self.dis_tokenizer = dis_tokenizer
        self.to_shuffle_imp = to_shuffle_imp
        self.do_sym_augmentation = do_sym_augmentation
        self.do_dis_augmentation = do_dis_augmentation
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples

    def prepare_data(self) -> None:
        for file_path in self.file_paths:
            DiseaseDataset(
                file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=True,
                to_shuffle_imp=self.to_shuffle_imp,
                do_sym_augmentation=self.do_sym_augmentation,
                do_dis_augmentation=self.do_dis_augmentation,
                save_cache=True,
                load_cache=False,
                n_samples=self.n_train_samples,
            )
            DiseaseDataset(
                file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=False,
                to_shuffle_imp=False,
                do_sym_augmentation=False,
                do_dis_augmentation=False,
                save_cache=True,
                load_cache=False,
                n_samples=self.n_test_samples,
            )

    def setup(self, stage=None):
        self.train_ds = [
            DiseaseDataset(
                file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=True,
                to_shuffle_imp=self.to_shuffle_imp,
                do_sym_augmentation=self.do_sym_augmentation,
                do_dis_augmentation=self.do_dis_augmentation,
                save_cache=False,
                load_cache=True,
                n_samples=self.n_train_samples,
            )
            for file_path in self.file_paths
        ]
        self.test_ds = [
            DiseaseDataset(
                file_path,
                sym_tokenizer=self.sym_tokenizer,
                dis_tokenizer=self.dis_tokenizer,
                is_train=False,
                to_shuffle_imp=False,
                do_sym_augmentation=False,
                do_dis_augmentation=False,
                save_cache=False,
                load_cache=True,
                n_samples=self.n_test_samples,
            )
            for file_path in self.file_paths
        ]

    def train_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4,
            )
            for ds in self.train_ds
        ]

    def val_dataloader(self):
        return [
            DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=4)
            for ds in self.test_ds
        ]

    def test_dataloader(self):
        return [
            DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=4)
            for ds in self.test_ds
        ]
