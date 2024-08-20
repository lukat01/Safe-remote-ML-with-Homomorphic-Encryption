from abc import ABC, abstractmethod

import tenseal as ts
import torch

# coefficients = [0.5, 0.197, 0, -0.004]
coefficients = [0.5, 0.197]

fixed_weight = [0.167891725897789, 0.3086147904396057, 0.24430279433727264,
                0.16990801692008972, 0.16139128804206848, 0.14825797080993652,
                0.3167053461074829, 0.05549086257815361, 0.12054791301488876]
fixed_bias = [-0.04890957474708557]


class AbstractLogisticRegression(ABC):

    def __init__(self, context, x_train, y_train, num_features, iterations):
        self.ctx = context
        self.x_train = x_train
        self.y_train = y_train
        self.num_features = num_features
        self.iterations = iterations
        if num_features is not None:
            lr = torch.nn.Linear(self.num_features, 1)
            self.weight = torch.tensor(lr.weight.data.tolist()[0], dtype=torch.float)
            self.bias = torch.tensor(lr.bias.data.tolist(), dtype=torch.float)
            # self.weight = torch.tensor(fixed_weight, dtype=torch.float)
            # self.bias = torch.tensor(fixed_bias, dtype=torch.float)
        else:
            self.weight = None
            self.bias = None
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    @abstractmethod
    def train(self):
        pass

    def reset(self, weight, bias):
        pass

    def set(self, x, y):
        pass

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        self.bias -= self._delta_b * (1 / self._count)
        self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = enc_out - enc_y
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1

    @staticmethod
    def sigmoid(x):
        return x.polyval(coefficients)

    @abstractmethod
    def forward(self, enc_x):
        pass

    def predict(self, x_data):
        return [self.forward(x) for x in x_data]


class PlainLogisticRegression(AbstractLogisticRegression):

    def __init__(self, context, x_train=None, y_train=None, num_features=None, iterations=None,
                 weight=None, bias=None):
        super().__init__(context, x_train, y_train, num_features, iterations)
        if weight is not None and bias is not None:
            self.weight = weight
            self.bias = bias
            self.enc_weight = ts.ckks_tensor(self.ctx, self.weight)
            self.enc_bias = ts.ckks_tensor(self.ctx, self.bias)
        else:
            self.enc_weight = None
            self.enc_bias = None

    def train(self):
        for _ in range(self.iterations):
            for x, y in zip(self.x_train, self.y_train):
                out = self.forward(x)
                self.backward(x, out, y)
            self.update_parameters()
        self.enc_weight = ts.ckks_tensor(self.ctx, self.weight)
        self.enc_bias = ts.ckks_tensor(self.ctx, self.bias)

    def forward(self, x):
        if isinstance(x, ts.CKKSTensor):
            out = x.dot(self.enc_weight)
            out = out.reshape([1])
            out = out + self.enc_bias
        else:
            out = x.dot(self.weight)
            out = out + self.bias
        out = self.sigmoid(out)
        return out

    @staticmethod
    def sigmoid(x):
        if isinstance(x, ts.CKKSTensor):
            out = AbstractLogisticRegression.sigmoid(x)
        else:
            coefficients_tensor = torch.tensor(coefficients, dtype=torch.float)
            powers = torch.arange(len(coefficients_tensor), dtype=torch.float).unsqueeze(0)
            x_powers = x ** powers
            out = (coefficients_tensor.unsqueeze(0) * x_powers).sum(dim=-1)
        return out


class EncryptedLogisticRegression(AbstractLogisticRegression):

    def __init__(self, context, x_train=None, y_train=None, num_features=None, iterations=None,
                 weight=None, bias=None, double=False):
        super().__init__(context, x_train, y_train, num_features, iterations)
        if weight is not None and bias is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.weight = ts.ckks_tensor(self.ctx, self.weight)
            self.bias = ts.ckks_tensor(self.ctx, self.bias)
        self.current_iteration = 0
        self.double = double

    def set(self, x, y):
        self.x_train = x
        self.y_train = y
        if self.double:
            self.weight = ts.ckks_tensor(self.ctx, self.weight)
            self.bias = ts.ckks_tensor(self.ctx, self.bias)

    def reset(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, enc_x: ts.CKKSTensor):
        enc_out = enc_x.dot(self.weight)
        enc_out = enc_out.reshape([1])
        enc_out += self.bias
        enc_out = self.sigmoid(enc_out)
        return enc_out

    def train(self):
        for enc_x, enc_y in zip(self.x_train, self.y_train):
            enc_out = self.forward(enc_x)
            self.backward(enc_x, enc_out, enc_y)
        self.update_parameters()
        self.current_iteration += 1
        return True if self.current_iteration >= self.iterations else False, self.weight, self.bias
