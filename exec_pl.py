

import utils.exp_utils as utils
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from imlone.loaders import get_loader
from imlone.models import get_model


def train(cf):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    # data_loader = utils.import_module('dl', os.path.join(cf.exp.work_dir, cf.exp.dataset_source, 'svhn_loader.py'))

    datamodule = get_loader(cf)
    model = get_model(cf)
    tb_logger= TensorBoardLogger(save_dir=cf.exp.dir)
    csv_logger= CSVLogger(save_dir=cf.exp.dir)
    trainer = pl.Trainer(gpus=1, logger=[tb_logger, csv_logger])
    trainer.fit(model=model, train_dataloader=datamodule.train_dataloader, val_dataloaders=datamodule.val_dataloader)


def test(cf):
    pass

    # results_dict = eval_utils.test_sym2img(args, logger)
    # utils.write_test(args, writer, results_dict)
    # writer.close()


@hydra.main(config_name="config")
def main(cf: DictConfig):

    print(OmegaConf.to_yaml(cf))

    if cf.exp.mode == 'train' or cf.exp.mode == 'train_test':

        # cf = utils.prep_exp(cf, use_stored_settings=cf.use_stored_settings)

        for fold in cf.exp.folds:
            cf.exp.fold = fold
            train(cf)
            if cf.exp.mode == 'train_test':
                test(cf)

    # elif cf.exp.mode == 'test':
    #
    #     args = utils.prep_exp(in_args, is_training=False, use_stored_settings=True)
    #     writer = SummaryWriter(log_dir=args.plot_dir)
    #     if in_args.apply_test_config_mods:
    #         args = utils.apply_test_config_mods(args)
    #     data_loader = utils.import_module('dl', os.path.join(args.exec_dir, args.dataset_source, 'svhn_loader.py'))
    #     if folds is None:
    #         folds = range(args.n_cv_splits)
    #
    #     for fold in folds:
    #         args.fold_dir = os.path.join(args.exp_dir, 'fold_{}'.format(fold))
    #         args.checkpoint_path = os.path.join(args.fold_dir, args.checkpoint_name)
    #         logger = utils.get_logger(args.fold_dir)
    #         args.fold = fold
    #         test()

    else:
        raise RuntimeError('mode specified in in_args is not implemented...')



if __name__ == '__main__':
    main()




