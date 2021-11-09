from pytorch_lightning.callbacks import Callback
import torch
from collections import OrderedDict
from copy import deepcopy

class TrainingStages(Callback):

    def __init__(self, milestones, disable_dropout_at_finetuning):
        self.milestones = milestones
        self.disable_dropout_at_finetuning = disable_dropout_at_finetuning

    def on_train_start(self, trainer, pl_module):
        if pl_module.pretrained_backbone_path is not None:
            self.milestones[1] = self.milestones[1] - self.milestones[0]
            self.milestones[0] = 0

    def on_train_epoch_start(self, trainer, pl_module):

        # self.check_weight_consistency(pl_module)

        if pl_module.current_epoch == self.milestones[0]: # this is the end before the queried epoch
            print("Starting Training ConfidNet")
            pl_module.training_stage = 1
            if pl_module.pretrained_backbone_path is None: # trained from scratch, reload best epoch
                best_ckpt_path = trainer.checkpoint_callbacks[0].last_model_path # No backbone model selection!!
                print("Check last backbone path", best_ckpt_path)
            else:
                best_ckpt_path = pl_module.pretrained_backbone_path

            loaded_ckpt = torch.load(best_ckpt_path)
            loaded_state_dict = loaded_ckpt["state_dict"]

            backbone_encoder_state_dict = OrderedDict(
                (k.replace("backbone.encoder.", ""), v) for k, v in loaded_state_dict.items() if
                "backbone.encoder." in k)
            if len(backbone_encoder_state_dict) == 0:
                backbone_encoder_state_dict = loaded_state_dict
            backbone_classifier_state_dict = OrderedDict(
                (k.replace("backbone.classifier.", ""), v) for k, v in loaded_state_dict.items() if
                "backbone.classifier." in k)

            pl_module.backbone.encoder.load_state_dict(backbone_encoder_state_dict, strict=True)
            pl_module.backbone.classifier.load_state_dict(backbone_classifier_state_dict, strict=True)
            pl_module.network.encoder.load_state_dict(backbone_encoder_state_dict, strict=True)
            pl_module.network.classifier.load_state_dict(backbone_classifier_state_dict, strict=True)

            print("loaded checkpoint {} from epoch {} into backbone and network.".format(best_ckpt_path, loaded_ckpt["epoch"]))

            pl_module.network.encoder = deepcopy(pl_module.backbone.encoder)
            pl_module.network.classifier = deepcopy(pl_module.backbone.classifier)

            print("freezing backbone and enabling confidnet")
            self.freeze_layers(pl_module.backbone.encoder)
            self.freeze_layers(pl_module.backbone.classifier)
            self.freeze_layers(pl_module.network.encoder)
            self.freeze_layers(pl_module.network.classifier)

            new_optimizer = torch.optim.Adam(pl_module.network.confid_net.parameters(),
                               lr=pl_module.learning_rate_confidnet,
                                )

            trainer.optimizers = [new_optimizer]
            trainer.lr_schedulers = []
            new_schedulers = []
            if pl_module.confidnet_lr_scheduler:
                print("initializing new scheduler for confidnet...")
                new_schedulers.append({"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=new_optimizer,
                                                                                         T_max=self.milestones[1]-self.milestones[0],
                                                                                         verbose=True),
                                      "interval": "epoch",
                                      "frequency": 1,
                                      "name": "confidnet_adam"})


                trainer.lr_schedulers = trainer.configure_schedulers(new_schedulers)

            lr_monitor = [x for x in trainer.callbacks if "lr_monitor" in x.__str__()]
            if len(lr_monitor) > 0:
                lr_monitor[0].__init__()
                lr_monitor[0].on_train_start(trainer)



            # self.check_weight_consistency(pl_module)

        if pl_module.current_epoch >= self.milestones[0]:
            self.disable_bn(pl_module.backbone.encoder)
            self.disable_bn(pl_module.network.encoder)
            for param_group in trainer.optimizers[0].param_groups:
                print("CHECK ConfidNet RATE", param_group["lr"])


        if pl_module.current_epoch == self.milestones[1]:
            print("Starting Training Fine Tuning ConfidNet")# new optimizer or add param groups? both adam according to paper!
            pl_module.training_stage = 2
            if pl_module.pretrained_confidnet_path is not None:
                best_ckpt_path = pl_module.pretrained_confidnet_path
            elif hasattr(pl_module, "test_selection_criterion") and "latest" not in pl_module.test_selection_criterion:
                best_ckpt_path = trainer.checkpoint_callbacks[1].best_model_path
                print("Test selection criterion", pl_module.test_selection_criterion)
                print("Check BEST confidnet path", best_ckpt_path)
            else:
                best_ckpt_path = None
                print("going with latest confidnet")
            if best_ckpt_path is not None:
                loaded_ckpt = torch.load(best_ckpt_path)
                loaded_state_dict = loaded_ckpt["state_dict"]
                loaded_state_dict = OrderedDict((k.replace("network.confid_net.",""),v) for k, v in loaded_state_dict.items() if
                    "network.confid_net" in k)
                pl_module.network.confid_net.load_state_dict(loaded_state_dict, strict=True)
                print("loaded checkpoint {} from epoch {} into new encoder".format(best_ckpt_path, loaded_ckpt["epoch"]))

            self.unfreeze_layers(pl_module.network.encoder)
            new_optimizer = torch.optim.Adam(pl_module.network.parameters(),
                                              lr=pl_module.learning_rate_confidnet_finetune,
                                             )
            trainer.optimizers = [new_optimizer]
            trainer.optimizer_frequencies = []

            # self.check_weight_consistency(pl_module)

        if self.disable_dropout_at_finetuning:
            if pl_module.current_epoch >= self.milestones[1]:
                self.disable_dropout(pl_module.backbone.encoder)
                self.disable_dropout(pl_module.network.encoder)

        # print("FINAL CHECK BACKBONE ENC")
        # for layer in pl_module.backbone.encoder.named_modules():
        #        print(layer[1], layer[1].training)
        # print("FINLA CHECK NETWORK ENC")
        # for layer in pl_module.network.encoder.named_modules():
        #        print(layer[1], layer[1].training)

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
            # print("BN CHECK", layer)
            if "bn" in layer[0] or "cbr_unit.1" in layer[0] or isinstance(layer[1], torch.nn.BatchNorm2d):
                layer[1].momentum = 0
                layer[1].eval()
                # print("disabling bn", layer[1])

    def disable_dropout(self, model):

        for layer in model.named_modules():
            if "dropout" in layer[0] or isinstance(layer[1], torch.nn.Dropout):
                layer[1].eval()
                # print("disabling dropout", layer[1], layer[0])


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
                print("CONFID CLassifier", x[0], x[1].mean().item())
