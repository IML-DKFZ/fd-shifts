from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class DropoutEnablerMixin(nn.Module):
    """Ensures we can enable and disable dropout"""

    @abstractmethod
    def enable_dropout(self) -> None:
        """Enable Droupout for the Module"""

    @abstractmethod
    def disable_dropout(self) -> None:
        """Disable Droupout for the Module"""


class Network(nn.Module, metaclass=ABCMeta):
    """Network that has an encoder and a classifier"""

    @property
    def encoder(self) -> DropoutEnablerMixin:
        """
        Returns:
            Encoder network
        """
        return self._encoder

    @encoder.setter
    def encoder(self, model: DropoutEnablerMixin) -> None:
        """
        Encoder network
        """
        self._encoder = model

    @property
    def classifier(self) -> nn.Module:
        """
        Returns:
            Classifier network
        """
        return self._classifier

    @classifier.setter
    def classifier(self, model: nn.Module) -> None:
        """Classifier network"""
        self._classifier = model

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def head(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
