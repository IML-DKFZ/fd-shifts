

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from src.loaders import get_loader
from src.models import get_model
from src.utils import exp_utils
import analysis
import os


def train(cf, subsequent_testing=False):
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
                                          save_top_k=cf.trainer.save_top_k,
                                          save_last=True,
                                         )
    trainer = pl.Trainer(gpus=1,
                         logger=[tb_logger, csv_logger],
                         max_epochs=cf.trainer.num_epochs,
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint = resume_ckpt_path,
                         benchmark=cf.trainer.benchmark,
                         check_val_every_n_epoch = cf.trainer.val_every_n_epoch,
                         fast_dev_run=cf.trainer.fast_dev_run
                         )

    print("logging training to: {}, version: {}".format(cf.exp.dir, cf.exp.version))

    trainer.fit(model=model, datamodule=datamodule)

    if subsequent_testing:

        if not os.path.exists(cf.test.dir):
            os.makedirs(cf.test.dir)

        trainer.test(ckpt_path=None)
        analysis.main(cf.test.dir, cf.test.dir)



def test(cf):

    # double check that this is not overwritten before it is loaded!
    print("laoding test config from ", cf.test.cf_path)
    cf = OmegaConf.load(cf.test.cf_path)

    if cf.test.model_selection == "best":
        ckpt_path = exp_utils.get_path_to_best_ckpt(cf.exp.dir, cf.trainer.selection_mode)
    else:
        most_recent_version = exp_utils.get_most_recent_version(cf.exp.dir)
        ckpt_path = exp_utils.get_ckpt_path_from_previous_version(cf.exp.dir, most_recent_version)

    print("testing model from checkpoint: {} from model selection tpye {}".format(
        ckpt_path, cf.test.model_selection))
    print("logging testing to: {}".format(cf.test.dir))
    # Todo overwriting training configs from checkpoint not possible atm
    model = get_model(cf.model.name).load_from_checkpoint(ckpt_path)
    datamodule = get_loader(cf)

    if not os.path.exists(cf.test.dir):
        os.makedirs(cf.test.dir)

    trainer = pl.Trainer(gpus=1, logger=False)
    trainer.test(model, datamodule=datamodule)

    # fix str bug
    # test resuming by testing a second time in the same dir
    # how to print the tested epoch into csv log?

@hydra.main(config_path="src/configs", config_name="config")
def main(cf: DictConfig):

    print(OmegaConf.to_yaml(cf))

    if cf.exp.mode == 'train':
        train(cf)

    if cf.exp.mode == 'train_test':
        train(cf, subsequent_testing=True)

    if cf.exp.mode == 'test':
        test(cf)


if __name__ == '__main__':
   main()

