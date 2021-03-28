

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from src.loaders import get_loader
from src.models import get_model
from src.utils import exp_utils


def train(cf):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    resume_ckpt_path = None
    cf.exp.version = exp_utils.get_next_version(cf.exp.dir)
    if cf.trainer.resume_from_ckpt:
        cf.exp.version -= 1
        resume_ckpt_path = exp_utils.get_ckpt_path_from_previous_version(cf.exp.dir, cf.exp.version)
        print("resuming previous training:", resume_ckpt_path)


    if cf.trainer.global_seed:
        exp_utils.set_seed(cf.trainer.global_seed)
        cf.trainer.benchmark = False
        print("setting benchmark to False for deterministic training.")

    datamodule = get_loader(cf)
    model = get_model(cf.model.name)(cf)
    tb_logger= TensorBoardLogger(save_dir=cf.exp.group_dir,
                                 name=cf.exp.name,
                                 default_hp_metric=False,
                                 )
    cf.exp.version = tb_logger.version
    csv_logger= CSVLogger(save_dir=cf.exp.group_dir,
                          name=cf.exp.name,
                          version=cf.exp.version)
    checkpoint_callback = ModelCheckpoint(dirpath=cf.exp.version_dir,
                                          filename="best",
                                          monitor=cf.trainer.selection_metric,
                                          mode=cf.trainer.selection_mode,
                                          save_top_k=1,
                                          save_last=True,
                                         )
    trainer = pl.Trainer(gpus=1,
                         logger=[tb_logger, csv_logger],
                         max_epochs=cf.trainer.num_epochs,
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint = resume_ckpt_path,
                         benchmark=cf.trainer.benchmark,
                         check_val_every_n_epoch = cf.trainer.val_every_n_epoch
                         )

    print("logging training to: {}, version: {}".format(cf.exp.dir, cf.exp.version))

    trainer.fit(model=model, datamodule=datamodule)


def test(cf):

    ckpt_path = exp_utils.get_path_to_best_ckpt(cf.exp.dir, cf.trainer.selection_mode)
    print("testing model from checkpoint: {}".format(ckpt_path))
    print("logging testing to: {}, run_version: {}".format(cf.exp.dir, cf.test.name))
    # MIX 2 configs..ugly??? Hydra can not specify another input config. and anyway overwrites would not be included.
    # I need to make sure all potential changes are in hparams saved from pl.
    # there could also be a way via pl_checkpoint = torch.load(ckpt_path). described here under "logging hyperparamters":
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html
    model = get_model(cf.model.name).load_from_checkpoint(ckpt_path)
    datamodule = get_loader(cf)
    tb_logger = TensorBoardLogger(save_dir=cf.exp.dir,
                                  name=cf.test.name,
                                  default_hp_metric=False,
                                  version=0)
    csv_logger = CSVLogger(save_dir=cf.exp.dir,
                          name=cf.test.name,
                          version=0)
    trainer = pl.Trainer(gpus=1, logger=[tb_logger, csv_logger])
    trainer.test(model, datamodule=datamodule)

    # fix str bug
    # test resuming by testing a second time in the same dir
    # how to print the tested epoch into csv log?

@hydra.main(config_path="src/configs", config_name="config")
def main(cf: DictConfig):

    print(OmegaConf.to_yaml(cf))

    if cf.exp.mode == 'train':
        train(cf)

    if cf.exp.mode == 'test':
        test(cf)


if __name__ == '__main__':
   main()

