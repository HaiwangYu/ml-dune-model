"""LightningCLI entry for larmamba2 (grid-tile masked autoencoder).

Owns its trainer_defaults (LearningRateMonitor only) so the config's single
per-epoch ModelCheckpoint doesn't collide with baked-in ones — the lesson from
gridutils/run_polarmae_fit.py. DDP on this pool additionally needs
NCCL_P2P_DISABLE=1 (set by the trainjob).

    python -m larmamba2.train fit --config larmamba2/configs/larmamba2_full20.yml
"""
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.cli import LightningCLI

from polarmae.datasets import APA2DDataModule

from larmamba2.ssl_module import Larmamba2MAE

if __name__ == "__main__":
    LightningCLI(
        Larmamba2MAE,
        APA2DDataModule,
        trainer_defaults={
            "accelerator": "gpu",
            "precision": "bf16-mixed",
            "callbacks": [LearningRateMonitor()],
        },
        parser_kwargs={"parser_mode": "omegaconf"},
        seed_everything_default=0,
        save_config_callback=None,
    )
