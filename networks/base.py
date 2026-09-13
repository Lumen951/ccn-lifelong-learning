from abc import ABC, abstractmethod


class BaseLSTM(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def get_target_without_sideeffects(self, inputs):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def decay_gradient(self, decay_rate):
        pass

    @abstractmethod
    def backward(self, layer):
        pass

    @abstractmethod
    def update_parameters(self, layer, error):
        pass

    @abstractmethod
    def print_features_stats(self):
        pass
