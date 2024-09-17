from pytorch_lightning.loggers import TensorBoardLogger
from src.pcd_de_noising import WeatherNet
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.pcd_de_noising import PCDDataset, PointCloudDataModule
import os
import torch

NUMBER_GPUS = 1


def main() -> None:
    DATASET_PATH = "./data"
    logger = TensorBoardLogger("tb_logs", name="WeatherNet")
    model = WeatherNet()

    data_module = PointCloudDataModule(
        os.path.join(DATASET_PATH, "train"),
        os.path.join(DATASET_PATH, "val"),
        os.path.join(DATASET_PATH, "test"),
    )

    trainer = pl.Trainer(
        logger=logger,
        overfit_batches=2,
        # progress_bar_refresh_rate=30,
        max_epochs=30,
        # flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        # gpus=1,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
