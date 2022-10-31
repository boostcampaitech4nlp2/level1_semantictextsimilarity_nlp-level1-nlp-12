from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import torch
from dataloader.dataloader import DataLoader
from model.model import Model
from trainer.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", required=True)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_epoch", type=int)

    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}.yaml")
    for arg_name, value in vars(args).items():
        if arg_name == "learning_rate":
            cfg.train.learning_rate = value
        elif arg_name == "weight_decay":
            cfg.optimizer.weight_decay = value
        elif arg_name == "max_epoch":
            cfg.optimizer.max_epoch = value

    exp_name = "_".join(
        map(
            str,
            [
                cfg.name,
                cfg.model.model_name,
                cfg.optimizer.name,
                cfg.lr_scheduler.name,
                cfg.train.learning_rate,
                cfg.train.batch_size,
            ],
        )
    )
    wandb_logger = WandbLogger(name=exp_name, project=cfg.project)

    dataloader = DataLoader(
        cfg.model.model_name,
        cfg.train.batch_size,
        cfg.data.shuffle,
        cfg.path.train_path,
        cfg.path.dev_path,
        cfg.path.test_path,
        cfg.path.predict_path,
    )
    model = Model(cfg)

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.train.gpus,
        max_epochs=cfg.train.max_epoch,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.logging_step,
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, f"{cfg.model.saved_name}.pt")
