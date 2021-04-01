from pytorch_lightning.callbacks import Callback
import torch

class TrainingStages(Callback):

    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        self.milestones = milestones
        self.train_bn = train_bn


    def on_train_epoch_start(self, trainer, pl_module):

        for ix, x in enumerate(pl_module.network.named_parameters()):
            if ix == 0:
                print(pl_module.current_epoch, x[0], x[1].mean())

        if pl_module.current_epoch == self.milestones[0]: # this is the end before the queried epoch
            print("Starting Training of ConfidNet")
            # todo load best checkpoint of encoder into new encoder
            pl_module.network.encoder.load_state_dict(trainer.checkpoint_callbacks[0].best_model_path, strict=True)
            print("loading checkpoint {} into new encoder".format(trainer.checkpoint_callback[0].best_model_path))
            self.disable_bn(pl_module)
            new_optimizers = torch.optim.Adam([v for k,v in pl_module.named_parameters() if "uncertainty" in k], # todo need nicer naming!!
                               lr=pl_module.learning_rate_confidnet,
                               weight_decay=pl_module.weight_decay)
            trainer.optimizers = [new_optimizers]


        if pl_module.current_epoch == self.milestones[0]: # new optimizer or add param groups? both adam according to paper!
            pl_module.network.confid_net.load_state_dict(trainer.checkpoint_callbacks[1].best_model_path, strict=False)
            self.disable_dropout(pl_module.network)
            new_optimizers = torch.optim.Adam(pl_module.network.parameters(),
                                              lr=pl_module.learning_rate_confidnet_finetune,
                                              weight_decay=pl_module.weight_decay)
            trainer.optimizers = [new_optimizers]

            # print("FREEZING", pl_module.current_epoch)
            # self.freeze_layers(pl_module, freeze_string="conv1")
            # self.disable_dropout(pl_module)

        # for ix, x in enumerate(pl_module.encoder.named_parameters()):
        #     if ix == 0:
        #         print(pl_module.current_epoch, x[0], x[1].mean())


    def freeze_layers(self, model, freeze_string):
        for param in model.named_parameters():
            if freeze_string in param[0]:
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
