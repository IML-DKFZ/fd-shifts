from pytorch_lightning.callbacks import Callback
import torch
from collections import OrderedDict

class TrainingStages(Callback):

    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        self.milestones = milestones
        self.train_bn = train_bn


    def on_train_epoch_start(self, trainer, pl_module):


        if pl_module.current_epoch == self.milestones[0]: # this is the end before the queried epoch
            print("Starting Training of ConfidNet")
            pl_module.training_stage = 1
            # # todo here goes the pretraining flag!
            # best_ckpt_path = trainer.checkpoint_callbacks[0].best_model_path
            # loaded_state_dict = torch.load(best_ckpt_path)["state_dict"]
            # pl_module.load_state_dict(loaded_state_dict, strict=True)
            # loaded_state_dict = OrderedDict((k.replace("backbone.encoder.",""),v) for k, v in loaded_state_dict.items() if "backbone.encoder" in k)
            # pl_module.network.encoder.load_state_dict(loaded_state_dict, strict=True)
            # print("loaded checkpoint {} into new encoder".format(best_ckpt_path))
            # self.disable_bn(pl_module.network)
            # new_optimizer = torch.optim.Adam(pl_module.network.confid_net.parameters(), # todo need nicer naming!!
            #                    lr=pl_module.learning_rate_confidnet,
            #                    weight_decay=pl_module.weight_decay)
            # trainer.optimizers = [new_optimizer]
            # trainer.optimizer_frequencies = []
            # self.freeze_layers(pl_module, keep_string="confid_net")
            # print("CHECK TRAIN STAGES")



        if pl_module.current_epoch == self.milestones[1]:
            print("Starting Training Fine Tuning ConfidNet")# new optimizer or add param groups? both adam according to paper!
            pl_module.training_stage = 2
            best_ckpt_path = trainer.checkpoint_callbacks[1].best_model_path
            loaded_state_dict = torch.load(best_ckpt_path)["state_dict"]
            loaded_state_dict = OrderedDict((k.replace("network.confid_net.",""),v) for k, v in loaded_state_dict.items() if
                "network.confid_net" in k)
            pl_module.network.confid_net.load_state_dict(loaded_state_dict, strict=True)
            print("loaded checkpoint {} into confidnet".format(trainer.checkpoint_callbacks[1].best_model_path))
            self.disable_dropout(pl_module.network)
            new_optimizer = torch.optim.Adam(pl_module.network.parameters(),
                                              lr=pl_module.learning_rate_confidnet_finetune,
                                              weight_decay=pl_module.weight_decay)
            trainer.optimizers = [new_optimizer]

        # for ix, x in enumerate(pl_module.network.named_parameters()):
        #     if ix == 0:
        #         print(pl_module.current_epoch, x[0], x[1].mean())


    def freeze_layers(self, model, freeze_string=None, keep_string=None):
        for param in model.named_parameters():
            if freeze_string is not None and freeze_string in param[0]:
                param[1].requires_grad = False
            if keep_string is not None and keep_string not in param[0]:
                param[1].requires_grad = False

    def disable_bn(self, model):
        # Freeze also BN running average parameters
        for layer in model.named_modules():
            if "bn" in layer[0] or "cbr_unit.1" in layer[0]:
                layer[1].momentum = 0
                layer[1].eval()

    def disable_dropout(self, model):

        for layer in model.named_modules():
            if "dropout" in layer[0]:
                layer[1].eval()
