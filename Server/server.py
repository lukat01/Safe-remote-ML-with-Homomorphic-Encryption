import os
import glob
import pickle

import torch.serialization
from flask import jsonify, Flask
from flask import request
from time import time

from lr_models import *

# clients: dict[str, ts.Context] = dict()
# models: dict[str, dict[str, AbstractLogisticRegression]] = dict()
active: dict[str, dict[str, AbstractLogisticRegression]] = dict()
models: dict[str, dict[str, set[str]]] = dict()

application = Flask(__name__)
STORAGE_URL = "Server\\clients"
PLAIN = "plain"
ENCRYPTED = "encrypted"
pattern = os.path.join(STORAGE_URL, '*.bin')


def initialize():
    print("Server initialization started")
    for filepath in glob.glob(pattern):
        # with open(filepath, 'rb') as ctx_file:
        #     ctx_data = pickle.load(ctx_file)
        # init_ctx = ts.context_from(ctx_data)
        basename = os.path.basename(filepath)
        c_id, _ = os.path.splitext(basename)
        # clients[c_id] = init_ctx
        models[c_id] = dict()
        models[c_id][PLAIN] = set()
        models[c_id][ENCRYPTED] = set()
        folder_pattern = os.path.join(STORAGE_URL, c_id, PLAIN, '*.bin')
        for model_file in glob.glob(folder_pattern):
            model_name, _ = os.path.splitext(os.path.basename(model_file))
            # with open(model_file, 'rb') as model:
            #    loaded_weight, loaded_bias = pickle.load(model)
            #    models[c_id][model_name] = PlainLogisticRegression(init_ctx, weight=loaded_weight,
            #    bias=loaded_bias)
            models[c_id][PLAIN].add(model_name)
        folder_pattern = os.path.join(STORAGE_URL, c_id, ENCRYPTED, '*.bin')
        for model_file in glob.glob(folder_pattern):
            model_name, _ = os.path.splitext(os.path.basename(model_file))
            models[c_id][ENCRYPTED].add(model_name)
            # with open(model_file, 'rb') as model:
            #    loaded_weight, loaded_bias = pickle.load(model)
            #    loaded_weight = ts.ckks_tensor_from(init_ctx, loaded_weight)
            #    loaded_bias = ts.ckks_tensor_from(init_ctx, loaded_bias)
            #    models[c_id][model_name] = EncryptedLogisticRegression(init_ctx,
            #    weight=loaded_weight, bias=loaded_bias)
        print("Server initialization completed")


torch.serialization.add_safe_globals([initialize])


def model_train_process_response(completed, weight, bias):
    if not completed and (weight is None or bias is None):
        json_response = {
            "error": "internal server error"
        }
        return jsonify(json_response), 500

    json_response = {
        "bias": bias.serialize().decode("iso-8859-1"),
        "weight": weight.serialize().decode("iso-8859-1"),
        "message": "accepted"
    }

    return jsonify(json_response), 200 if completed else 202


@application.route("/", methods=["GET"])
def index():
    return "OK", 200


@application.route("/register/<client_id>", methods=["POST"])
def add_client(client_id):
    if client_id is None:
        return jsonify({"error": "Client ID is missing"}), 400
    if client_id in models:  # clients:
        return jsonify({"error": "Client ID already exists"}), 409
    try:
        ts.context_from(request.data)
    except Exception as e:
        return jsonify({"error": f"Failed to create context: {str(e)}"}), 400
    # clients[client_id] = ctx
    models[client_id] = dict()  # dict()
    models[client_id][PLAIN] = set()
    models[client_id][ENCRYPTED] = set()
    os.makedirs(f"{STORAGE_URL}/{client_id}/{PLAIN}")
    os.makedirs(f"{STORAGE_URL}/{client_id}/{ENCRYPTED}")
    with open(f'{STORAGE_URL}/{client_id}.bin', 'wb') as file:
        pickle.dump(request.data, file)
    return jsonify({"message": "Client registered successfully"}), 201


@application.route("/train_encrypted/<client_id>/<model_id>", methods=["POST"])
def train_encrypted(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:  # clients:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id in models[client_id][PLAIN] or model_id in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model already exists"}), 409

    print("Reading received data")
    start_time = time()
    enc_x_train = request.json["x_train"]
    print("X data read")
    enc_y_train = request.json["y_train"]
    print("Y data read")
    num_features = request.json["num_features"]
    iterations = request.json["iterations"]
    double = request.json["double"]
    end_time = time()
    print(f"Data loaded in {end_time - start_time} seconds")

    if enc_x_train is None or enc_y_train is None or num_features is None or iterations is None:
        return jsonify(
            {"error": "Send all required json data: iterations, num_features, enc_x_train, enc_y_train"}), 400

    # context = clients[client_id]
    with open(f"{STORAGE_URL}/{client_id}.bin", 'rb') as ctx_file:
        ctx_data = pickle.load(ctx_file)
        context = ts.context_from(ctx_data)
    print("Raw data conversion")
    enc_x_train = [ts.ckks_tensor_from(context, e.encode("iso-8859-1")) for e in enc_x_train]
    enc_y_train = [ts.ckks_tensor_from(context, e.encode("iso-8859-1")) for e in enc_y_train]
    print("Raw data converted to CKKS tensors")
    if double:
        model = active[client_id][model_id]
        model.set(enc_x_train, enc_y_train)
    else:
        model = EncryptedLogisticRegression(context, enc_x_train, enc_y_train, num_features, iterations)
        # models[client_id][model_id] = model
        active[client_id][model_id] = model
        models[client_id][ENCRYPTED].add(model_id)

    completed, weight, bias = model.train()

    return model_train_process_response(completed, weight, bias)


@application.route("/train_encrypted_continue/<client_id>/<model_id>", methods=["POST"])
def train_encrypted_continue(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400

    weight = request.json["weight"]
    bias = request.json["bias"]
    finalization = request.json["finalization"]

    if weight is None or bias is None or finalization is None:
        return jsonify({"error": "Send all required json data: weight, bias, completed"}), 400

    # context = clients[client_id]
    with open(f"{STORAGE_URL}/{client_id}.bin", 'rb') as ctx_file:
        ctx_data = pickle.load(ctx_file)
        context = ts.context_from(ctx_data)

    model = active[client_id][model_id]

    weight = ts.ckks_tensor_from(context, weight.encode("iso-8859-1"))
    bias = ts.ckks_tensor_from(context, bias.encode("iso-8859-1"))
    model.reset(weight, bias)

    if not finalization:
        completed, weight, bias = model.train()
        return model_train_process_response(completed, weight, bias)

    with open(f'{STORAGE_URL}/{client_id}/{ENCRYPTED}/{model_id}.bin', 'wb') as file:
        pickle.dump((model.weight.serialize(), model.bias.serialize()), file)

    return jsonify({"message": "Training fully completed", "model": model_id}), 200


@application.route("/train_plain/<client_id>/<model_id>", methods=["POST"])
def train_plain(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:  # clients:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id in models[client_id][PLAIN] or model_id in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model already exists"}), 409

    x_train = torch.tensor(request.json["x_train"], dtype=torch.float64)
    y_train = torch.tensor(request.json["y_train"], dtype=torch.float64)
    num_features = request.json["num_features"]
    iterations = request.json["iterations"]
    double = request.json["double"]

    if x_train is None or y_train is None or num_features is None or iterations is None:
        return jsonify(
            {"error": "Send all required json data: iterations, num_features, enc_x_train, enc_y_train"}), 400

    with open(f"{STORAGE_URL}/{client_id}.bin", 'rb') as ctx_file:
        ctx_data = pickle.load(ctx_file)
        context = ts.context_from(ctx_data)

    model = PlainLogisticRegression(context, x_train, y_train, num_features, iterations)
    # models[client_id][model_id] = model
    models[client_id][PLAIN].add(model_id)
    if double:
        model_enc = EncryptedLogisticRegression(context, num_features=num_features,
                                                iterations=iterations, bias=model.bias, weight=model.weight,
                                                double=True)
        enc_index = model_id.find(f"_{PLAIN}")
        model_id_enc = model_id[:enc_index]
        active[client_id][model_id_enc] = model_enc
        models[client_id][ENCRYPTED].add(model_id_enc)

    model.train()

    with open(f'{STORAGE_URL}/{client_id}/{PLAIN}/{model_id}.bin', 'wb') as file:
        pickle.dump((model.weight, model.bias), file)
    return jsonify({"message": "Training fully completed", "model": model_id}), 200


@application.route("/eval_encrypted/<client_id>/<model_id>", methods=["POST"])
def eval_encrypted(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:  # clients:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id not in models[client_id][PLAIN] and model_id not in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model doesn't exists"}), 404

    enc_x_eval = request.json["x_eval"]
    if enc_x_eval is None:
        return jsonify(
            {"error": "Send all required json data: x_eval"}), 400

    with open(f"{STORAGE_URL}/{client_id}.bin", 'rb') as ctx_file:
        ctx_data = pickle.load(ctx_file)
        context = ts.context_from(ctx_data)
    enc_x_eval = [ts.ckks_tensor_from(context, e.encode("iso-8859-1")) for e in enc_x_eval]
    model = models[client_id][model_id]
    predictions = model.predict(enc_x_eval)
    json_response = {
        "message": "Evaluation completed",
        "predictions": [p.serialize().decode("iso-8859-1") for p in predictions]
    }
    return jsonify(json_response), 200


@application.route("/eval_plain/<client_id>/<model_id>", methods=["POST"])
def eval_plain(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:  # clients:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id not in models[client_id][PLAIN] and model_id not in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model doesn't exists"}), 404
    x_eval = request.json["x_eval"]
    if x_eval is None:
        return jsonify(
            {"error": "Send all required json data: x_eval"}), 400
    x_eval = torch.tensor(x_eval, dtype=torch.float64)
    # TODO: this
    folder = ENCRYPTED if model_id in models[client_id][ENCRYPTED] else PLAIN
    model = models[client_id][model_id]
    if isinstance(model, EncryptedLogisticRegression):
        with open(f"{STORAGE_URL}/{client_id}.bin", 'rb') as ctx_file:
            ctx_data = pickle.load(ctx_file)
            context = ts.context_from(ctx_data)
        x_eval = [ts.ckks_tensor(context, x) for x in x_eval]
    predictions = model.predict(x_eval)
    if isinstance(model, EncryptedLogisticRegression):
        predictions = [p.serialize().decode("iso-8859-1") for p in predictions]
    else:
        predictions = [p.tolist()[0] for p in predictions]
    json_response = {
        "message": "Evaluation completed",
        "predictions": predictions,
        "encrypted_prediction": isinstance(model, EncryptedLogisticRegression)
    }
    return jsonify(json_response), 200


@application.route("/get_num_features/<client_id>/<model_id>", methods=["GET"])
def get_num_features(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:  # clients:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id not in models[client_id][PLAIN] and model_id not in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model doesn't exists"}), 404
    num_features = 0  # TODO: figure this out
    return jsonify({"num_features": num_features}), 200


if __name__ == "__main__":
    initialize()
    application.run(debug=True)
