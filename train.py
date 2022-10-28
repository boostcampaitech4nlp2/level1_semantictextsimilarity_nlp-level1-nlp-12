from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import torch
import pytorch_lightning as pl
from dataloader.dataloader import DataLoader
from model.model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f"./wandb_base_project/config/{args.config}.yaml")

    exp_name = "_".join(
        map(
            str,
            [
                cfg.name,
                cfg.model.model_name,
                cfg.optimizer,
                cfg.scheduler.name,
                cfg.train.learning_rate,
                cfg.train.batch_size,
            ],
        )
    )
    wandb.init(config=cfg, name=exp_name, project=cfg.project)
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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.train.gpus,
        max_epochs=cfg.train.max_epoch,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.logging_step,
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, f"{cfg.model.saved_name}.pt")
