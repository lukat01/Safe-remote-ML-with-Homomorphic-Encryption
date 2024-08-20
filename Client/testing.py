import tenseal as ts
import numpy as np

from homomorphic_lr import create_context

start_weights = [0.167891725897789, 0.3086147904396057, 0.24430279433727264,
                 0.16990801692008972, 0.16139128804206848, 0.14825797080993652,
                 0.3167053461074829, 0.05549086257815361, 0.12054791301488876]
start_bias = [-0.04890957474708557]

x = [-1.0151, 0.5093, 0.9560, -0.1000, -0.8467, -1.6417, -0.6911, 1.5906, -0.2347]
y = [0]
coefficients = [0.5, 0.197, 0, -0.004]
_delta_w = 0
_delta_b = 0
_count = 0

# forward
weights = np.array(start_weights, dtype=np.float64)
bias = np.array(start_bias, dtype=np.float64)
x = np.array(x, dtype=np.float64)
forward_plain = np.add(x.dot(weights), bias)
forward_plain = np.polyval(coefficients[::-1], forward_plain)
print(f"Forward result: {forward_plain[0]}")
# backward
out_minus_y = np.subtract(forward_plain, y)
print(out_minus_y)
_delta_w += forward_plain * out_minus_y
print(_delta_w)
_delta_b += out_minus_y
print(_delta_b)
_count += 1
# update
bias -= _delta_b * (1 / _count)
weights -= _delta_w * (1 / _count) + weights * 0.05
print(bias)
print(weights)

print()
context = create_context()

weights_encrypted = ts.ckks_vector(context, start_weights)
x_encrypted = ts.ckks_vector(context, x)
y_encrypted = ts.ckks_vector(context, y)
bias_encrypted = ts.ckks_vector(context, start_bias)
forward_encrypted = x_encrypted.dot(weights_encrypted) + bias_encrypted
forward_encrypted = forward_encrypted.polyval(coefficients)

forward_decrypted = forward_encrypted.decrypt()

print(f"Forward result: {forward_decrypted[0]}, "
      f"difference = {abs(forward_plain - forward_decrypted[0])}")
_delta_w_enc = ts.ckks_vector(context, [0])
_delta_b_enc = ts.ckks_vector(context, [0])
_count_enc = 0
# backward
out_minus_y_enc = forward_decrypted - y_encrypted
print(out_minus_y_enc.decrypt(), abs(out_minus_y_enc.decrypt()[0] - out_minus_y))
_delta_w_enc += forward_encrypted * out_minus_y_enc
print(_delta_w_enc.decrypt(), abs(_delta_w_enc.decrypt()[0] - _delta_w))
_delta_b_enc += out_minus_y_enc
print(_delta_b_enc.decrypt(), abs(_delta_b_enc.decrypt()[0] - _delta_b))
_count_enc += 1
# update
bias_encrypted -= _delta_b_enc * (1 / _count_enc)
weights_encrypted -= _delta_w_enc * (1 / _count_enc) + weights_encrypted * 0.05
print(bias_encrypted.decrypt()[0], abs(bias_encrypted.decrypt()[0] - bias[0]))
print(weights_encrypted.decrypt(), abs(weights_encrypted.decrypt() - weights))

print(bias.tolist())
print(bias_encrypted.decrypt())
print(weights.tolist())
print(weights_encrypted.decrypt())
