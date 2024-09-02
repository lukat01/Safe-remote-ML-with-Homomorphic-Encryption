from abc import ABC, abstractmethod

import tenseal as ts
import torch

coefficients = [0.5, 0.197, 0, -0.004]
default_float = torch.float


class AbstractLogisticRegression(ABC):

    def __init__(self, context=None, x_train=None, y_train=None,
                 num_features=None, iterations=None, weight=None, bias=None,
                 learning_rate=1, regularization_strength=0.1):
        self.ctx = context
        self.x_train = x_train
        self.y_train = y_train
        self.num_features = num_features
        self.iterations = iterations
        if weight is None and bias is None:
            lr = torch.nn.Linear(self.num_features, 1)
            self.weight = torch.tensor(lr.weight.data.tolist()[0], dtype=default_float)
            self.bias = torch.tensor(lr.bias.data.tolist(), dtype=default_float)
        else:
            self.weight = weight
            self.bias = bias
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
        if x_train is not None:
            self._count = len(x_train)
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength

    @abstractmethod
    def train(self):
        pass

    def forward(self, x):
        out = x.dot(self.weight) + self.bias
        out = self.sigmoid(out)
        return out

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = enc_out - enc_y
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("No data for training is provided")
        self.bias -= self._delta_b * (self.learning_rate / self._count)
        reg_term = self.weight * self.regularization_strength
        factor = self.learning_rate / self._count
        self.weight -= (self._delta_w + reg_term) * factor
        self._delta_w = 0
        self._delta_b = 0

    @staticmethod
    def sigmoid(x):
        if isinstance(x, ts.CKKSVector):
            out = x.polyval(coefficients)
        else:
            coefficients_tensor = torch.tensor(coefficients, dtype=default_float)
            powers = torch.arange(len(coefficients_tensor), dtype=default_float).unsqueeze(0)
            x_powers = x ** powers
            out = (coefficients_tensor.unsqueeze(0) * x_powers).sum(dim=-1)
        return out

    def predict(self, x_data):
        return [self.forward(x) for x in x_data]


class PlainLogisticRegression(AbstractLogisticRegression):

    def __init__(self, x_train=None, y_train=None, num_features=None, iterations=None,
                 weight=None, bias=None, learning_rate=1, regularization_strength=0.1):
        super().__init__(None, x_train, y_train, num_features, iterations, weight, bias,
                         learning_rate, regularization_strength)

    def train(self):
        for _ in range(self.iterations):
            for x, y in zip(self.x_train, self.y_train):
                out = self.forward(x)
                self.backward(x, out, y)
            self.update_parameters()


class EncryptedLogisticRegression(AbstractLogisticRegression):

    def __init__(self, context, x_train=None, y_train=None, num_features=None, iterations=None,
                 weight=None, bias=None, learning_rate=1, regularization_strength=0.1):
        super().__init__(context, x_train, y_train, num_features, iterations, weight, bias,
                         learning_rate, regularization_strength)
        self.current_iteration = 0

    def set(self, x, y):
        self._count = len(x)
        self.x_train = x
        self.y_train = y
        self.weight = ts.ckks_vector(self.ctx, self.weight)
        self.bias = ts.ckks_vector(self.ctx, self.bias)

    def reset(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def train(self):
        for enc_x, enc_y in zip(self.x_train, self.y_train):
            enc_out = self.forward(enc_x)
            self.backward(enc_x, enc_out, enc_y)
        self.update_parameters()
        self.current_iteration += 1
        return True if self.current_iteration >= self.iterations else False, self.weight, self.bias
