import os
import pickle

import tenseal as ts
import torch
from enum import Enum
import json
from time import time
import requests

url = "http://127.0.0.1:5000"


class Operation(Enum):
    TRAIN = "training"
    EVAL = "evaluation"


def create_context():
    poly_mod_degree = 2 ** 13  # 2 ** 13
    coefficient_mod_bit_sizes = [60, 40, 40, 60]  # [40, 21, 21, 21, 21, 21, 21, 40]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coefficient_mod_bit_sizes)
    ctx.global_scale = 2 ** 40  # 21
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.auto_relin = True
    ctx.auto_rescale = True
    ctx.auto_mod_switch = True
    return ctx


def registration(client):
    print("Creating and serializing context")
    context = create_context()
    serialized = context.serialize()
    client_serialized = context.serialize(save_secret_key=True)
    os.makedirs(f"Client/clients/{client}")
    os.makedirs(f"Client/clients/{client}/models")
    with open(f'Client/clients/{client}/{client}.bin', 'wb') as file:
        pickle.dump(client_serialized.decode("iso-8859-1"), file)
    print("Context created")

    print("Sending registration request")
    response = requests.post(url=f"{url}/register/{client}", data=serialized)
    print(f"Registration: {response.status_code}\n")
    if response.status_code != 201:
        response_dict = json.loads(response.content)
        raise RuntimeError(f"Error while creating new context: {response_dict['error']}")
    return context


def get_context(client):
    print("Loading context")
    with open(f'Client/clients/{client}/{client}.bin', 'rb') as file:
        data = pickle.load(file)
        context = ts.context_from(data.encode("iso-8859-1"))
        print("Context loaded\n")
        return context


def send_data(client, model, operation: Operation, ctx, x_data, y_data=None,  # both x and y are list of list
              num_features=None, iterations=None, encrypted=True, double=False):

    operation_str = operation.value
    if encrypted:
        print(f"Encrypting data for {operation_str}")
        start_time = time()
        x_data = [ts.ckks_tensor(ctx, x).serialize().decode("iso-8859-1") for x in x_data]
        if operation == Operation.TRAIN:
            y_data = [ts.ckks_tensor(ctx, y).serialize().decode("iso-8859-1") for y in y_data]
        end_time = time()
        print(f"Encryption of the {operation_str} set took {end_time - start_time:.4f} seconds")
    else:
        x_data = x_data.tolist()
        y_data = y_data.tolist() if y_data is not None else None

    print(f"Sending data for {operation_str}")
    json_request = {
        f"x_{operation.name.lower()}": x_data,
        f"y_{operation.name.lower()}": y_data,
        "iterations": iterations,
        "num_features": num_features,
        "double": double
    }
    start_time = time()
    op_response = requests.post(
        url=f"{url}/{operation.name.lower()}_{'encrypted' if encrypted else 'plain'}/{client}/{model}",
        json=json_request
    )

    if op_response.status_code // 100 != 2:
        response_dict = json.loads(op_response.content)
        raise RuntimeError(f"Error while sending data: {response_dict['error']}")

    finalization = False
    while (op_response.status_code == 202 or finalization) and operation == Operation.TRAIN and encrypted:
        response_dict = json.loads(op_response.content.decode("iso-8859-1"))
        bias = ts.ckks_tensor_from(ctx, response_dict["bias"].encode("iso-8859-1"))
        bias = bias.decrypt()
        bias = ts.ckks_tensor(ctx, bias)
        weight = ts.ckks_tensor_from(ctx, response_dict["weight"].encode("iso-8859-1"))
        weight = weight.decrypt()
        weight = ts.ckks_tensor(ctx, weight)
        json_request = {
            "bias": bias.serialize().decode("iso-8859-1"),
            "weight": weight.serialize().decode("iso-8859-1"),
            "finalization": finalization
        }
        op_response = requests.post(
            url=f"{url}/{operation.name.lower()}_encrypted_continue/{client}/{model}",
            json=json_request
        )
        if op_response.status_code == 200 and not finalization:
            finalization = True
        elif op_response.status_code == 200 and finalization:
            finalization = False

    end_time = time()
    print(f"{operation_str.capitalize()} of the model took {end_time - start_time:.4f} seconds\n")

    if operation == Operation.TRAIN:
        response_dict = json.loads(op_response.content)
        return response_dict["model"]
    elif operation == Operation.EVAL and encrypted:
        response_dict = json.loads(op_response.content.decode("iso-8859-1"))
        predictions_encrypted = [ts.ckks_tensor_from(ctx, p.encode("iso-8859-1"))
                                 for p in response_dict["predictions"]]
        predictions_decrypted = [p.decrypt().tolist() for p in predictions_encrypted]
        return torch.tensor(predictions_decrypted, dtype=torch.float)
    else:
        response_dict = json.loads(op_response.content)
        predictions = response_dict["predictions"]
        encrypted_prediction = response_dict["encrypted_prediction"]
        if encrypted_prediction:
            predictions = [ts.ckks_tensor_from(ctx, p.encode("iso-8859-1")) for p in predictions]
            predictions = [p.decrypt() for p in predictions]
        else:
            predictions = [[p] for p in predictions]
        return torch.tensor(predictions, dtype=torch.float)


def training(client, model, ctx, x_data, y_data, num_features, iterations, encrypted=True, double=False):
    return send_data(client, model, Operation.TRAIN, ctx, x_data, y_data, num_features, iterations, encrypted, double)


def testing(client, model, ctx, x_data, encrypted=True):
    return send_data(client, model, Operation.EVAL, ctx, x_data, encrypted=encrypted)


def predict_single_entry(client, model, ctx, x_data, encrypted=True):
    predicted = send_data(client, model, Operation.EVAL, ctx, x_data, encrypted=encrypted).item()
    return 1 if predicted > 0.5 else 0


def calculate_accuracy(real_values, predicted_values):
    correct = torch.abs(real_values - predicted_values) < 0.5
    return correct.float().mean()


def get_num_features(client, model):
    op_response = requests.post(
        url=f"{url}/get_num_features/{client}/{model}",
    )
    response_dict = json.loads(op_response.content)
    if op_response.status_code != 200:
        raise RuntimeError(f"Error while creating new context: {response_dict['error']}")
    return response_dict["num_features"]


def delete_model(client, model):
    op_response = requests.delete(
        url=f"{url}/delete_model/{client}/{model}",
    )
    response_dict = json.loads(op_response.content)
    if op_response.status_code != 200:
        raise RuntimeError(f"Error while creating new context: {response_dict['error']}")
    return response_dict["message"]
