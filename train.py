from utils import getlogger, generate_from_tensor
from data import (
    FixedVocabTokenizer,
    plDiseaseDataset,
)
from pytorch_lightning import seed_everything
from models import COAD
from typing import Dict, Any
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import torch
import argparse
import pytorch_lightning as pl
import torchmetrics
import wandb
import os
import json


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


class plCOAD(pl.LightningModule):
    def __init__(
        self,
        model: COAD,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.sym_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=model.sym_head.output_dim
        )
        self.dis_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=model.dis_head.output_dim
        )

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, train_batch, batch_idx):
        outputs = self.model(**train_batch)
        sym_loss = outputs[0]
        dis_loss = outputs[1]
        sym_logits = outputs[2]
        dis_logits = outputs[3]
        loss = sym_loss + dis_loss

        # shape: (B, T)
        sym_preds = sym_logits[:, :-1].max(dim=2)[1]
        # shape: (B, T)
        sym_ids = train_batch["sym_ids"][:, 1:]
        sym_mask = sym_ids != -100
        self.sym_acc(
            sym_preds[sym_mask].reshape(-1), sym_ids[sym_mask].reshape(-1)
        )

        # shape: (B, T)
        dis_preds = dis_logits.max(dim=2)[1]
        dis_ids = train_batch["step_dis_ids"]
        dis_mask = dis_ids != -100
        self.dis_acc(
            dis_preds[dis_mask].reshape(-1), dis_ids[dis_mask].reshape(-1)
        )
        self.log(
            "train/sym_loss",
            sym_loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train/sym_acc",
            self.sym_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train/dis_loss",
            dis_loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train/dis_acc",
            self.dis_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "trainer/learning_rate",
            self.optimizers().param_groups[0]["lr"],
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        # batch size is always 1 for validation
        # get dis id from the last step.
        if self.trainer.global_step == 0 and self.config["use_wandb"]:
            wandb.define_metric("train/dis_loss", summary="min")
            wandb.define_metric("train/sym_loss", summary="min")
            if self.config["sep_test"]:
                for ds in self.config["dataset"]:
                    wandb.define_metric(f"val/dis_acc/{ds}", summary="max")
                    wandb.define_metric(f"val/sym_rec/{ds}", summary="max")
                    wandb.define_metric(f"val/len/{ds}", summary="min")
                wandb.define_metric("val/avg_dis_acc", summary="max")
                wandb.define_metric("val/avg_sym_rec", summary="max")
                wandb.define_metric("val/avg_len", summary="min")
            else:
                wandb.define_metric("val/dis_acc", summary="max")
                wandb.define_metric("val/sym_rec", summary="max")
                wandb.define_metric("val/len", summary="min")
        dis_id = val_batch["step_dis_ids"][0, -1]
        sym_ids = val_batch["sym_ids"][0]
        sym_type_ids = val_batch["sym_type_ids"][0]
        if self.config["eos_token_id"] is not None:
            sym_ids = sym_ids[:-1]
            sym_type_ids = sym_type_ids[:-1]
        sym_rec, dis_succ, total_len = generate_from_tensor(
            model=self.model,
            sym_ids=sym_ids,
            sym_type_ids=sym_type_ids,
            dis_id=dis_id,
            exp_len=val_batch["exp_len"][0],
            max_turn=self.config["max_turn"],
            min_probability=self.config["min_p"],
            end_probability=self.config["end_p"],
            eos_token_id=self.config["eos_token_id"],
        )
        return sym_rec, dis_succ, total_len

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        # batch size is always 1 for validation
        # get dis id from the last step.
        dis_id = test_batch["step_dis_ids"][0, -1]
        sym_ids = test_batch["sym_ids"][0]
        sym_type_ids = test_batch["sym_type_ids"][0]
        if self.config["eos_token_id"] is not None:
            sym_ids = sym_ids[:-1]
            sym_type_ids = sym_type_ids[:-1]
        sym_rec, dis_succ, total_len = generate_from_tensor(
            model=self.model,
            sym_ids=sym_ids,
            sym_type_ids=sym_type_ids,
            dis_id=dis_id,
            exp_len=test_batch["exp_len"][0],
            max_turn=self.config["max_turn"],
            min_probability=self.config["min_p"],
            end_probability=self.config["end_p"],
            eos_token_id=self.config["eos_token_id"],
        )
        return sym_rec, dis_succ, total_len

    def validation_epoch_end(self, outputs) -> None:
        if self.config["sep_test"]:
            avg_accs = []
            avg_recs = []
            avg_lens = []
            for i, ds in enumerate(self.config["dataset"]):
                test_rec, test_succ, test_len = zip(*outputs[i])
                avg_recs.append(np.mean(test_rec).item())
                avg_accs.append(np.mean(test_succ).item())
                avg_lens.append(np.mean(test_len).item())
                self.log(
                    f"val/dis_acc/{ds}",
                    avg_accs[-1],
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"val/sym_rec/{ds}",
                    avg_recs[-1],
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"val/len/{ds}",
                    avg_lens[-1],
                    prog_bar=False,
                    sync_dist=True,
                )
            self.log(
                "val/avg_dis_acc",
                np.mean(avg_accs),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/avg_sym_rec",
                np.mean(avg_recs),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/avg_len",
                np.mean(avg_lens),
                prog_bar=True,
                sync_dist=True,
            )
        else:
            test_rec, test_succ, test_len = zip(*outputs)
            avg_rec = np.mean(test_rec)
            avg_acc = np.mean(test_succ)
            avg_len = np.mean(test_len)
            self.log("val/dis_acc", avg_acc, prog_bar=True, sync_dist=True)
            self.log("val/sym_rec", avg_rec, prog_bar=True, sync_dist=True)
            self.log("val/len", avg_len, prog_bar=True, sync_dist=True)

    def test_epoch_end(self, outputs) -> None:
        if self.config["sep_test"]:
            avg_accs = []
            avg_recs = []
            avg_lens = []
            for i, ds in enumerate(self.config["dataset"]):
                test_rec, test_succ, test_len = zip(*outputs[i])
                avg_recs.append(np.mean(test_rec).item())
                avg_accs.append(np.mean(test_succ).item())
                avg_lens.append(np.mean(test_len).item())
                self.log(
                    f"test/dis_acc/{ds}",
                    avg_accs[-1],
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"test/sym_rec/{ds}",
                    avg_recs[-1],
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"test/len/{ds}",
                    avg_lens[-1],
                    prog_bar=False,
                    sync_dist=True,
                )
            self.log(
                "test/avg_dis_acc",
                np.mean(avg_accs),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/avg_sym_rec",
                np.mean(avg_recs),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/avg_len",
                np.mean(avg_lens),
                prog_bar=True,
                sync_dist=True,
            )
        else:
            test_rec, test_succ, test_len = zip(*outputs)
            avg_rec = np.mean(test_rec)
            avg_acc = np.mean(test_succ)
            avg_len = np.mean(test_len)
            self.log("test/dis_acc", avg_acc, prog_bar=True, sync_dist=True)
            self.log("test/sym_rec", avg_rec, prog_bar=True, sync_dist=True)
            self.log("test/len", avg_len, prog_bar=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["config"] = self.config

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            eps=self.config["adam_epsilon"],
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]


def eval(config, val=True):
    if config.get("checkpoint") is None:
        raise ValueError("Must provide `checkpoint` for evaluation.")

    checkpoint = config["checkpoint"]
    d = torch.load(checkpoint, map_location="cpu")
    config = d["config"]
    config["use_wandb"] = False
    syms = []
    dis = []
    datasets = config["dataset"]
    if isinstance(datasets, str):
        datasets = [datasets]
    for ds in datasets:
        syms += (
            open(f"{BASE_PATH}/data/{ds}/symptoms.txt", "r")
            .read()
            .splitlines()
        )
        dis += (
            open(f"{BASE_PATH}/data/{ds}/diseases.txt", "r")
            .read()
            .splitlines()
        )
    sym_tokenizer = FixedVocabTokenizer(
        syms,
        eos_token=config["eos_token"],
        unk_token=None,
    )
    dis_tokenizer = FixedVocabTokenizer(dis, eos_token=None, unk_token=None)
    config["eos_token_id"] = sym_tokenizer.eos_token_id
    _model = COAD(
        n_syms=len(sym_tokenizer.get_vocab()),
        n_dis=len(dis_tokenizer.get_vocab()),
        dis_hidden_sizes=config["dis_hidden_sizes"],
        dis_dropout_p=config["dis_dropout_p"],
        sym_hidden_sizes=config["sym_hidden_sizes"],
        sym_dropout_p=config["sym_dropout_p"],
        emb_dropout_p=config["emb_dropout_p"],
    )
    model = plCOAD.load_from_checkpoint(
        checkpoint, model=_model, config=config
    )
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="ddp",
        devices=1 if torch.cuda.is_available() else None,
        enable_progress_bar=True,
    )
    if isinstance(config["dataset"], list):
        file_path = [f"{BASE_PATH}/data/{ds}" for ds in config["dataset"]]
    else:
        file_path = f"{BASE_PATH}/data/{config['dataset']}"
    ds = plDiseaseDataset(
        file_path=file_path,
        sym_tokenizer=sym_tokenizer,
        dis_tokenizer=dis_tokenizer,
        batch_size=config["batch_size"],
        do_sym_augmentation=config["do_sym_augmentation"],
        do_dis_augmentation=config["do_dis_augmentation"],
        sep_test=config["sep_test"],
        n_train_samples=config["n_train_samples"],
        n_test_samples=config["n_test_samples"],
    )
    if val:
        outputs = trainer.validate(model, datamodule=ds)
    else:
        outputs = trainer.test(model, datamodule=ds)
    print(outputs)


def train(config):
    syms = []
    dis = []
    datasets = config["dataset"]
    if isinstance(datasets, str):
        datasets = [datasets]
    for ds in datasets:
        syms += (
            open(f"{BASE_PATH}/data/{ds}/symptoms.txt", "r")
            .read()
            .splitlines()
        )
        dis += (
            open(f"{BASE_PATH}/data/{ds}/diseases.txt", "r")
            .read()
            .splitlines()
        )
    syms = list(dict.fromkeys(syms))
    dis = list(dict.fromkeys(dis))

    sym_tokenizer = FixedVocabTokenizer(
        syms,
        eos_token=config["eos_token"],
        unk_token=None,
    )
    dis_tokenizer = FixedVocabTokenizer(dis, eos_token=None, unk_token=None)
    config["eos_token_id"] = sym_tokenizer.eos_token_id
    wandb_logger = None
    if config["use_wandb"]:
        if not config["project_name"]:
            raise ValueError("Must specify `project_name` if using wandb.")
        wandb_logger = WandbLogger(
            project=config["project_name"],
            name=config["save_prefix"],
            config=config,
        )
    _model = COAD(
        n_syms=len(sym_tokenizer.get_vocab()),
        n_dis=len(dis_tokenizer.get_vocab()),
        dis_hidden_sizes=config["dis_hidden_sizes"],
        dis_dropout_p=config["dis_dropout_p"],
        sym_hidden_sizes=config["sym_hidden_sizes"],
        sym_dropout_p=config["sym_dropout_p"],
        emb_dropout_p=config["emb_dropout_p"],
    )
    model = plCOAD(model=_model, config=config)
    if isinstance(config["dataset"], list):
        file_path = [f"{BASE_PATH}/data/{ds}" for ds in config["dataset"]]
    else:
        file_path = f"{BASE_PATH}/data/{config['dataset']}"

    ds = plDiseaseDataset(
        file_path=file_path,
        sym_tokenizer=sym_tokenizer,
        dis_tokenizer=dis_tokenizer,
        batch_size=config["batch_size"],
        do_sym_augmentation=config["do_sym_augmentation"],
        do_dis_augmentation=config["do_dis_augmentation"],
        n_train_samples=config["n_train_samples"],
        n_test_samples=config["n_test_samples"],
        sep_test=config["sep_test"],
    )
    callbacks = []
    callbacks.append(
        EarlyStopping(
            monitor="val/avg_dis_acc" if config["sep_test"] else "val/dis_acc",
            min_delta=0.00,
            patience=50,
            verbose=True,
            mode="max",
        )
    )
    model_checkpoint = ModelCheckpoint(
        monitor="val/avg_dis_acc" if config["sep_test"] else "val/dis_acc",
        dirpath=config["save_path"],
        filename=(
            f"{config['save_prefix']}-epoch={{epoch:02d}}"
            + (
                "-dis_acc={val/avg_dis_acc:.2f}"
                if config["sep_test"]
                else "-dis_acc={val/dis_acc:.2f}"
            )
            + (
                "-sym_rec={val/avg_sym_rec:.2f}"
                if config["sep_test"]
                else "-sym_rec={val/sym_rec:.2f}"
            )
            + (
                "-avg_len={val/avg_len:.2f}"
                if config["sep_test"]
                else "-avg_len={val/len:.2f}"
            )
        ),
        auto_insert_metric_name=False,
        mode="max",
        save_weights_only=True,
    )
    callbacks.append(model_checkpoint)
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="ddp",
        devices=torch.cuda.device_count()
        if torch.cuda.is_available()
        else None,
        # devices=1 if torch.cuda.is_available() else None,
        # devices=None,
        max_epochs=config["n_epochs"],
        # overfit_batches=1,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(model=model, datamodule=ds)
    if config["use_wandb"]:
        wandb.run.summary["best_path"] = model_checkpoint.best_model_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--batch_size", "-bz", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--dis_dropout_p", type=float, default=0)
    parser.add_argument("--sym_dropout_p", type=float, default=0)
    parser.add_argument("--emb_dropout_p", type=float, default=0.1)
    parser.add_argument(
        "--dis_hidden_sizes",
        nargs="+",
        help="The hidden size of dis net",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--sym_hidden_sizes",
        nargs="+",
        help="The hidden size of sym net",
        type=int,
        default=[],
    )
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--max_turn", type=int, default=20)
    parser.add_argument("--eos_token", type=str, default="[EOS]")
    parser.add_argument("--save_prefix", type=str)
    parser.add_argument("--dataset", "-ds", nargs="+", type=str, required=True)
    # parser.add_argument("--dataset", "-ds", type=str, required=True)
    parser.add_argument("--n_train_samples", type=int, default=-1)
    parser.add_argument("--n_test_samples", type=int, default=-1)
    parser.add_argument("--do_sym_augmentation", "-sym", action="store_true")
    parser.add_argument("--do_dis_augmentation", "-dis", action="store_true")
    parser.add_argument(
        "--save_path",
        type=str,
        help="Directory path for checkpoints.",
        default="saved/",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help=(
            "The minimum probability to inquire a sypmtom.  If lower than this"
            " threshold, will predict disease instead."
        ),
    )
    parser.add_argument(
        "--end_p",
        type=float,
        default=0.9,
        help="The probability threshold for predicting end of sentence token.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument("--project_name", type=str, help="Name for wandb")

    parser.add_argument(
        "--n_trials", type=int, help="Number of trials for ray tuning."
    )
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument(
        "--sep_test",
        action="store_true",
        help=(
            "Evaluate the datasets one by one (if multiple datasets)."
            " Otherwise, combine datasets and evaluate jointly."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        help="Location of the checkpoint for evaluation.",
    )
    args = parser.parse_args()

    if args.sep_test and len(args.dataset) <= 1:
        raise ValueError(
            "If using sep_test, you have to provide more than one dataset."
        )
    return args


def main(config):
    seed_everything(seed=config["seed"], workers=True)
    logger = getlogger(
        logger_level="debug",
        file_level="debug",
        console_level="info",
        log_path=f"{BASE_PATH}/logs/{config['save_prefix']}.txt",
    )

    if config["do_eval"]:
        eval(config, val=True)
    elif config["do_test"]:
        eval(config, val=False)
    elif config["do_train"]:
        train(config)
    else:
        logger.warning(
            "This script either train, eval or tune."
            " But receive no instructions."
        )


if __name__ == "__main__":
    args = vars(get_args())
    print(json.dumps(args, indent=4))
    main(args)
