

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from src.loaders.abstract_loader import AbstractDataLoader
from src.models import get_model
from src.models.callbacks import get_callbacks
from src.utils import exp_utils
import analysis
import os
import torch
import sys



def train(cf, subsequent_testing=False):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    print("CHECK CUDNN VERSION", torch.backends.cudnn.version())
    train_deterministic_flag = False
    if cf.exp.global_seed is not False:
        # exp_utils.set_seed(cf.exp.global_seed)
        exp_utils.set_seed(cf.exp.global_seed)
        cf.trainer.benchmark = False
        train_deterministic_flag = True
        print("setting seed {}, benchmark to False for deterministic training.".format(cf.exp.global_seed))


    resume_ckpt_path = None
    cf.exp.version = exp_utils.get_next_version(cf.exp.dir)
    if cf.trainer.resume_from_ckpt:
        cf.exp.version -= 1
        resume_ckpt_path = exp_utils.get_ckpt_path_from_previous_version(cf.exp.dir,
                                                                         cf.exp.version,
                                                                         "last")
        print("resuming previous training:", resume_ckpt_path)

    datamodule = AbstractDataLoader(cf)
    model = get_model(cf.model.name)(cf)
    tb_logger= TensorBoardLogger(save_dir=cf.exp.group_dir,
                                 name=cf.exp.name,
                                 default_hp_metric=False,
                                 )
    cf.exp.version = tb_logger.version
    csv_logger= CSVLogger(save_dir=cf.exp.group_dir,
                          name=cf.exp.name,
                          version=cf.exp.version)

    trainer = pl.Trainer(gpus=1,
                         logger=[tb_logger, csv_logger],
                         max_epochs=cf.trainer.num_epochs,
                         callbacks=get_callbacks(cf),
                         resume_from_checkpoint = resume_ckpt_path,
                         benchmark=cf.trainer.benchmark,
                         check_val_every_n_epoch = cf.trainer.val_every_n_epoch,
                         fast_dev_run=cf.trainer.fast_dev_run,
                         num_sanity_val_steps=5,
                         deterministic= train_deterministic_flag,
                         # limit_train_batches=1,
                         limit_val_batches=0 if cf.trainer.val_split is None else 1.0,
                         # replace_sampler_ddp=False,
                         # accelerator="ddp"
                         )

    print("logging training to: {}, version: {}".format(cf.exp.dir, cf.exp.version))
    trainer.fit(model=model, datamodule=datamodule)
    # analysis.main(in_path=cf.exp.version_dir,
    #               out_path=cf.exp.version_dir,
    #               query_studies={"iid_study": cf.data.dataset})

    if subsequent_testing:

        if not os.path.exists(cf.test.dir):
            os.makedirs(cf.test.dir)

        if cf.test.selection_criterion == "latest":
            ckpt_path = None
            print("testing with latest model...")
        elif "best" in cf.test.selection_criterion:
            ckpt_path = cf.test.best_ckpt_path
            print("testing with best model from {} and epoch {}".format(cf.test.best_ckpt_path,
                                                                        torch.load(ckpt_path)["epoch"]))
        trainer.test(ckpt_path=ckpt_path)
        analysis.main(in_path=cf.test.dir,
                      out_path=cf.test.dir,
                      query_studies=cf.eval.query_studies,
                      cf=cf)



def test(cf):

    if "best" in cf.test.selection_criterion and cf.test.only_latest_version is False:
        ckpt_path = exp_utils.get_path_to_best_ckpt(cf.exp.dir,
                                                    cf.test.selection_criterion,
                                                    cf.test.selection_mode)
    else:
        print("CHECK cf.exp.dir", cf.exp.dir)
        cf.exp.version = exp_utils.get_most_recent_version(cf.exp.dir)
        ckpt_path = exp_utils.get_ckpt_path_from_previous_version(cf.exp.dir,
                                                                       cf.exp.version,
                                                                       cf.test.selection_criterion)

    print("testing model from checkpoint: {} from model selection tpye {}".format(
        ckpt_path, cf.test.selection_criterion))
    print("logging testing to: {}".format(cf.test.dir))
    # via kwargs I can overwrite test configs at least manually in theconfig file for now.
    model = get_model(cf.model.name).load_from_checkpoint(ckpt_path, hparams_file=os.path.join(cf.exp.version_dir, "hparams.yaml"))
    datamodule = AbstractDataLoader(cf)

    if not os.path.exists(cf.test.dir):
        os.makedirs(cf.test.dir)

    trainer = pl.Trainer(gpus=1, logger=False, callbacks=get_callbacks(cf))
    trainer.test(model, datamodule=datamodule)
    analysis.main(in_path=cf.test.dir,
                  out_path=cf.test.dir,
                  query_studies=cf.eval.query_studies)

    # fix str bug
    # test resuming by testing a second time in the same dir
    # how to print the tested epoch into csv log?

@hydra.main(config_path="src/configs", config_name="config")
def main(cf: DictConfig):

    sys.stdout = exp_utils.Logger(cf.exp.log_path)
    sys.stderr = exp_utils.Logger(cf.exp.log_path)
    print(OmegaConf.to_yaml(cf))

    if cf.exp.mode == 'train':
        train(cf)

    if cf.exp.mode == 'train_test':
        train(cf, subsequent_testing=True)

    if cf.exp.mode == 'test':
        test(cf)

    if cf.exp.mode == 'analysis':
        analysis.main(in_path=cf.test.dir,
                      out_path=cf.test.dir,
                      query_studies=cf.eval.query_studies,
                      cf=cf)


if __name__ == '__main__':
   main()

