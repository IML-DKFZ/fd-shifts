from pytorch_lightning.callbacks import Callback
import torch
from collections import OrderedDict

class TrainingStages(Callback):

    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        self.milestones = milestones
        self.train_bn = train_bn


    def on_train_epoch_start(self, trainer, pl_module):

        # self.check_weight_consistency(pl_module)


        if pl_module.current_epoch == self.milestones[0]: # this is the end before the queried epoch
            print("Starting Training ConfidNet")
            pl_module.training_stage = 1
            if pl_module.pretrained_backbone_path is None: # trained from scratch, reload best epoch
                best_ckpt_path = trainer.checkpoint_callbacks[0].best_model_path
                loaded_ckpt = torch.load(best_ckpt_path)
                loaded_state_dict = loaded_ckpt["state_dict"]
                pl_module.load_state_dict(loaded_state_dict, strict=True) # load epoch for all models
                loaded_state_dict = OrderedDict(
                    (k.replace("backbone.", ""), v) for k, v in loaded_state_dict.items() if
                    "backbone" in k)
                pl_module.network.load_state_dict(loaded_state_dict, strict=False) # copy backbone into new encoder

            else:
                best_ckpt_path = pl_module.pretrained_backbone_path
                loaded_ckpt = torch.load(best_ckpt_path)

                # relict from old pretrained backbone naming. Todo if it was not for this relcit the if else could only be differing in the path!
                loaded_state_dict = loaded_ckpt["state_dict"]
                loaded_state_dict = OrderedDict(
                    (k.replace("encoder.", ""), v) if
                    "encoder" in k else (k, v) for k, v in loaded_state_dict.items())
                # TODO TRY TO LAOD MINE LIKE BNELOW
                # load their backbone for sanity checking
                # loaded_state_dict = loaded_ckpt["model_state_dict"]

                pl_module.backbone.encoder.load_state_dict(loaded_state_dict, strict=False)
                pl_module.backbone.classifier.load_state_dict(loaded_state_dict, strict=False)
                pl_module.network.encoder.load_state_dict(loaded_state_dict, strict=False)
                pl_module.network.classifier.load_state_dict(loaded_state_dict, strict=False)



            print("loaded checkpoint {} from epoch {} into new encoder".format(best_ckpt_path, loaded_ckpt["epoch"]))
            self.freeze_layers(pl_module.backbone)
            self.freeze_layers(pl_module.network.encoder)
            self.freeze_layers(pl_module.network.classifier)

            new_optimizer = torch.optim.Adam(pl_module.network.confid_net.parameters(), # todo need nicer naming!!
                               lr=pl_module.learning_rate_confidnet,
                               weight_decay=pl_module.weight_decay)
            trainer.optimizers = [new_optimizer]
            trainer.optimizer_frequencies = []

            # self.check_weight_consistency(pl_module)

        if pl_module.current_epoch >= self.milestones[0]:
            self.disable_bn(pl_module)

        if pl_module.current_epoch == self.milestones[1]:
            print("Starting Training Fine Tuning ConfidNet")# new optimizer or add param groups? both adam according to paper!
            pl_module.training_stage = 2
            best_ckpt_path = pl_module.pretrained_confidnet_path if pl_module.pretrained_confidnet_path is not None else trainer.checkpoint_callbacks[1].best_model_path
            loaded_ckpt = torch.load(best_ckpt_path)
            loaded_state_dict = loaded_ckpt["state_dict"]
            loaded_state_dict = OrderedDict((k.replace("network.confid_net.",""),v) for k, v in loaded_state_dict.items() if
                "network.confid_net" in k)
            pl_module.network.confid_net.load_state_dict(loaded_state_dict, strict=True)

            print("loaded checkpoint {} from epoch {} into new encoder".format(best_ckpt_path, loaded_ckpt["epoch"]))
            self.unfreeze_layers(pl_module.network.encoder)
            new_optimizer = torch.optim.Adam(pl_module.network.parameters(),  # TODO check if classifier still frozen
                                              lr=pl_module.learning_rate_confidnet_finetune,
                                              # weight_decay=pl_module.weight_decay
                                             )
            trainer.optimizers = [new_optimizer]
            trainer.optimizer_frequencies = []

            # self.check_weight_consistency(pl_module)

        if pl_module.current_epoch >= self.milestones[1]:
            self.disable_dropout(pl_module)


    def freeze_layers(self, model, freeze_string=None, keep_string=None):
        for param in model.named_parameters():
            if freeze_string is None and keep_string is None:
                param[1].requires_grad = False
            if freeze_string is not None and freeze_string in param[0]:
                param[1].requires_grad = False
            if keep_string is not None and keep_string not in param[0]:
                param[1].requires_grad = False

    def unfreeze_layers(self, model, unfreeze_string=None):
        for param in model.named_parameters():
           if unfreeze_string is None or unfreeze_string in param[0]:
                param[1].requires_grad = True

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


    def check_weight_consistency(self, pl_module):

        for ix, x in enumerate(pl_module.backbone.named_parameters()):
            if ix == 0:
                print("BACKBONE", x[0], x[1].mean().item())

        for ix, x in enumerate(pl_module.network.encoder.named_parameters()):
            if ix == 0:
                print("CONFID ENCODER", x[0], x[1].mean().item())

        for ix, x in enumerate(pl_module.network.confid_net.named_parameters()):
            if ix == 0:
                print("CONFIDNET", x[0], x[1].mean().item())

        for ix, x in enumerate(pl_module.network.classifier.named_parameters()):
            if ix == 0:
                print("CLassifier", x[0], x[1].mean().item())