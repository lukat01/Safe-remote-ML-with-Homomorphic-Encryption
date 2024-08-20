import tenseal as ts
import numpy as np

weights = [0.167891725897789, 0.3086147904396057, 0.24430279433727264,
           0.16990801692008972, 0.16139128804206848, 0.14825797080993652,
           0.3167053461074829, 0.05549086257815361, 0.12054791301488876]
vector = [-1.0151, 0.5093, 0.9560, -0.1000, -0.8467, -1.6417, -0.6911, 1.5906, -0.2347]
bias = [-0.04890957474708557]
# coefficients = [0.5, 0.197, 0, -0.004]
coefficients = [0.5, 0.197]

weights = np.array(weights, dtype=np.float32)
vector = np.array(vector, dtype=np.float32)
result_plain = vector.dot(weights) + bias[0]
# result_plain = np.polyval(coefficients[::-1], result_plain)

print(f"Plain result: {result_plain}")

for (poly_mod, coefficients_mod_bit_sizes, precision) in [
    (8192, [60, 40, 40, 60], 40)
]:

    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod,
        coeff_mod_bit_sizes=coefficients_mod_bit_sizes
    )
    context.global_scale = 2 ** precision
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.auto_relin = True
    context.auto_rescale = True
    context.auto_mod_switch = True

    weights_encrypted = ts.ckks_vector(context, weights)
    vector_encrypted = ts.ckks_vector(context, vector)
    bias_encrypted = ts.ckks_vector(context, bias)
    result_encrypted = vector_encrypted.dot(weights_encrypted) + bias_encrypted
    # result_encrypted = result_encrypted.polyval(coefficients)
    result_decrypted = result_encrypted.decrypt()

    print(f"Decrypted vector result: {result_decrypted[0]}, "
          f"difference = {abs(result_plain - result_decrypted[0])}")
