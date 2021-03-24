

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from loaders import get_loader
from models import get_model
import os


def train(cf):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    datamodule = get_loader(cf)
    model = get_model(cf)
    tb_logger= TensorBoardLogger(save_dir=cf.exp.dir)
    csv_logger= CSVLogger(save_dir=cf.exp.dir)
    trainer = pl.Trainer(gpus=1, logger=[tb_logger, csv_logger])
    trainer.fit(model=model, datamodule=datamodule)


def test():
    pass
    # model = LitMNIST.load_from_checkpoint(PATH)
    # trainer = Trainer(tpu_cores=8)
    # trainer.test(model)


@hydra.main(config_name="config")
def main(cf: DictConfig):

    print(OmegaConf.to_yaml(cf))
    print("logging to: ", cf.exp.dir)
    # if not os.path.exists(cf.exp.dir):
    #     os.mkdir(cf.exp.dir)

    if cf.exp.mode == 'train' or cf.exp.mode == 'train_test':

        for fold in cf.exp.folds:
            cf.exp.fold = fold
            train(cf)
            if cf.exp.mode == 'train_test':
                test()

    else:
        raise RuntimeError('mode specified in in_args is not implemented...')


if __name__ == '__main__':
   main()

