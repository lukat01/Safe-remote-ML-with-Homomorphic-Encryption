import gc
import os
import glob
import pickle
import threading

import torch.serialization
from flask import jsonify, Flask
from flask import request
from time import time
# from memory_profiler import profile

from lr_models import *

active: dict[str, dict[str, EncryptedLogisticRegression]] = dict()
models: dict[str, dict[str, set[str]]] = dict()
lock = threading.Lock()

application = Flask(__name__)
STORAGE_URL = "Server\\clients"
PLAIN = "plain"
ENCRYPTED = "encrypted"
pattern = os.path.join(STORAGE_URL, '*.bin')


def initialize():
    print("Server initialization started")

    if not os.path.exists(STORAGE_URL):
        os.makedirs(STORAGE_URL)
        print(f"Created directory: {STORAGE_URL}")

    for client_folder in glob.glob(os.path.join(STORAGE_URL, '*')):
        if os.path.isdir(client_folder):
            c_id = os.path.basename(client_folder)
            models[c_id] = dict()
            models[c_id][PLAIN] = set()
            models[c_id][ENCRYPTED] = set()
            active[c_id] = dict()

            plain_folder_pattern = os.path.join(client_folder, PLAIN, '*.bin')
            for model_file in glob.glob(plain_folder_pattern):
                model_name, _ = os.path.splitext(os.path.basename(model_file))
                models[c_id][PLAIN].add(model_name)

            encrypted_folder_pattern = os.path.join(client_folder, ENCRYPTED, '*.bin')
            for model_file in glob.glob(encrypted_folder_pattern):
                model_name, _ = os.path.splitext(os.path.basename(model_file))
                models[c_id][ENCRYPTED].add(model_name)

            client_bin_path = os.path.join(client_folder, 'client.bin')
            if not os.path.isfile(client_bin_path):
                raise RuntimeWarning(f"{client_bin_path} file not found!")

    print("Server initialization completed")


def model_train_process_response(completed, weight, bias, model=None, single=False):
    if not completed and (weight is None or bias is None):
        json_response = {
            "error": "internal server error"
        }
        return jsonify(json_response), 500

    json_response = {
        "bias": bias.serialize().decode("iso-8859-1"),
        "weight": weight.serialize().decode("iso-8859-1"),
        "message": "accepted",
        "single": single and completed
    }

    if completed:
        json_response["model"] = model

    return jsonify(json_response), 200 if completed and not single else 202


@application.route("/", methods=["GET"])
def index():
    return "OK", 200


@application.route("/register/<client_id>", methods=["POST"])
def add_client(client_id):
    if client_id is None:
        return jsonify({"error": "Client ID is missing"}), 400
    with lock:
        if client_id in models:
            return jsonify({"error": "Client ID already exists"}), 409
    try:
        ts.context_from(request.data)
    except Exception as e:
        return jsonify({"error": f"Failed to create context: {str(e)}"}), 400
    models[client_id] = dict()
    models[client_id][PLAIN] = set()
    models[client_id][ENCRYPTED] = set()
    active[client_id] = dict()
    os.makedirs(f"{STORAGE_URL}/{client_id}/{PLAIN}")
    os.makedirs(f"{STORAGE_URL}/{client_id}/{ENCRYPTED}")
    with open(f'{STORAGE_URL}/{client_id}/{client_id}.bin', 'wb') as file:
        pickle.dump(request.data, file)
    return jsonify({"message": "Client registered successfully"}), 201


@application.route("/train_encrypted/<client_id>/<model_id>", methods=["POST"])
# @profile()
def train_encrypted(client_id, model_id):
    double = request.json["double"]
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:
        return jsonify({"error": "Client doesn't exist, please register"}), 404

    with lock:
        if model_id in models[client_id][PLAIN] or model_id in models[client_id][ENCRYPTED]:
            if double and active[client_id][model_id].current_iteration == 0:
                pass
            else:
                return jsonify({"error": "Model already exists"}), 409
        if not double:
            models[client_id][ENCRYPTED].add(model_id)
    try:
        print("Reading received data")
        start_time = time()
        enc_x_train = request.json["x_train"]
        print("X data read")
        enc_y_train = request.json["y_train"]
        print("Y data read")
        num_features = request.json["num_features"]
        iterations = request.json["iterations"]
        end_time = time()
        print(f"Data loaded in {end_time - start_time} seconds")

        if enc_x_train is None or enc_y_train is None or num_features is None or iterations is None:
            return jsonify(
                {"error": "Send all required json data: iterations, num_features, enc_x_train, enc_y_train"}), 400

        with open(f"{STORAGE_URL}/{client_id}/{client_id}.bin", 'rb') as ctx_file:
            ctx_data = pickle.load(ctx_file)
            context = ts.context_from(ctx_data)
            del ctx_data
            gc.collect()

        print("Raw data conversion")
        enc_x_train = [ts.ckks_vector_from(context, e.encode("iso-8859-1")) for e in enc_x_train]
        enc_y_train = [ts.ckks_vector_from(context, e.encode("iso-8859-1")) for e in enc_y_train]
        print("Raw data converted to CKKS vectors")

        if double:
            model = active[client_id][model_id]
            model.set(enc_x_train, enc_y_train)
        else:
            model = EncryptedLogisticRegression(context, enc_x_train, enc_y_train, num_features, iterations)
            active[client_id][model_id] = model

        completed, weight, bias = model.train()

        del request.json["x_train"]
        del request.json["y_train"]
        gc.collect()

        return model_train_process_response(completed, weight, bias, model_id, True)
    except Exception as e:
        if client_id in models and ENCRYPTED in models[client_id] and model_id in models[client_id][ENCRYPTED]:
            models[client_id][ENCRYPTED].remove(model_id)
        if client_id in active and model_id in active[client_id]:
            active[client_id].pop(model_id)

        return jsonify({"error": str(e)}), 500


@application.route("/train_encrypted_continue/<client_id>/<model_id>", methods=["POST"])
# @profile
def train_encrypted_continue(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400

    try:
        weight = request.json["weight"]
        bias = request.json["bias"]
        finalization = request.json["finalization"]

        if weight is None or bias is None or finalization is None:
            return jsonify({"error": "Send all required json data: weight, bias, completed"}), 400

        with open(f"{STORAGE_URL}/{client_id}/{client_id}.bin", 'rb') as ctx_file:
            ctx_data = pickle.load(ctx_file)
            context = ts.context_from(ctx_data)
            del ctx_data
            gc.collect()

        model = active[client_id][model_id]

        weight = ts.ckks_vector_from(context, weight.encode("iso-8859-1"))
        bias = ts.ckks_vector_from(context, bias.encode("iso-8859-1"))
        model.reset(weight, bias)

        if not finalization:
            completed, weight, bias = model.train()
            return model_train_process_response(completed, weight, bias)

        with open(f'{STORAGE_URL}/{client_id}/{ENCRYPTED}/{model_id}.bin', 'wb') as file:
            pickle.dump((model.weight.serialize(), model.bias.serialize()), file)

        active[client_id].pop(model_id, None)

        return jsonify({"message": "Training fully completed", "model": model_id}), 200
    except Exception as e:
        if client_id in models and ENCRYPTED in models[client_id] and model_id in models[client_id][ENCRYPTED]:
            models[client_id][ENCRYPTED].remove(model_id)
        if client_id in active and model_id in active[client_id]:
            active[client_id].pop(model_id)
        model_path = f"{STORAGE_URL}/{client_id}/{ENCRYPTED}/{model_id}.bin"
        if os.path.exists(model_path):
            os.remove(model_path)
        return jsonify({"error": str(e)}), 500


@application.route("/train_plain/<client_id>/<model_id>", methods=["POST"])
def train_plain(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:
        return jsonify({"error": "Client doesn't exist, please register"}), 404

    double = request.json["double"]

    try:
        with lock:
            if model_id in models[client_id][PLAIN] or model_id in models[client_id][ENCRYPTED]:
                return jsonify({"error": "Model already exists"}), 409
            models[client_id][PLAIN].add(model_id)
            if double:
                enc_index = model_id.find(f"_{PLAIN}")
                model_id_enc = model_id[:enc_index]
                models[client_id][ENCRYPTED].add(model_id_enc)

        x_train = torch.tensor(request.json["x_train"], dtype=default_float)
        y_train = torch.tensor(request.json["y_train"], dtype=default_float)
        num_features = request.json["num_features"]
        iterations = request.json["iterations"]

        if x_train is None or y_train is None or num_features is None or iterations is None:
            return jsonify(
                {"error": "Send all required json data: iterations, num_features, enc_x_train, enc_y_train"}), 400

        with open(f"{STORAGE_URL}/{client_id}/{client_id}.bin", 'rb') as ctx_file:
            ctx_data = pickle.load(ctx_file)
            context = ts.context_from(ctx_data)
            del ctx_data
            gc.collect()

        model = PlainLogisticRegression(context, x_train, y_train, num_features, iterations)

        if double:
            w, b = model.weight.clone(), model.bias.clone()
            model_enc = EncryptedLogisticRegression(context, num_features=num_features,
                                                    iterations=iterations, bias=b, weight=w)
            active[client_id][model_id_enc] = model_enc

        model.train()

        with open(f'{STORAGE_URL}/{client_id}/{PLAIN}/{model_id}.bin', 'wb') as file:
            pickle.dump((model.weight, model.bias), file)

        return jsonify({"message": "Training fully completed", "model": model_id}), 200
    except Exception as e:
        if client_id in models and PLAIN in models[client_id] and model_id in models[client_id][PLAIN]:
            models[client_id][PLAIN].remove(model_id)
        model_path = f"{STORAGE_URL}/{client_id}/{PLAIN}/{model_id}.bin"
        if os.path.exists(model_path):
            os.remove(model_path)
        if double:
            if (client_id in models and ENCRYPTED in models[client_id] and
                    model_id_enc in models[client_id][ENCRYPTED]):
                models[client_id][ENCRYPTED].remove(model_id_enc)
            if client_id in active and model_id_enc in active[client_id]:
                active[client_id].pop(model_id_enc)

        return jsonify({"error": str(e)}), 500


@application.route("/eval_encrypted/<client_id>/<model_id>", methods=["POST"])
# @profile
def eval_encrypted(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id not in models[client_id][PLAIN] and model_id not in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model doesn't exists"}), 404

    enc_x_eval = request.json["x_eval"]
    if enc_x_eval is None:
        return jsonify(
            {"error": "Send all required json data: x_eval"}), 400

    with open(f"{STORAGE_URL}/{client_id}/{client_id}.bin", 'rb') as ctx_file:
        ctx_data = pickle.load(ctx_file)
        context = ts.context_from(ctx_data)
        del ctx_data
        gc.collect()

    enc_x_eval = [ts.ckks_vector_from(context, e.encode("iso-8859-1")) for e in enc_x_eval]
    folder = ENCRYPTED if model_id in models[client_id][ENCRYPTED] else PLAIN
    with open(f"{STORAGE_URL}/{client_id}/{folder}/{model_id}.bin", 'rb') as model_file:
        loaded_weight, loaded_bias = pickle.load(model_file)
        if folder == ENCRYPTED:
            loaded_weight = ts.ckks_vector_from(context, loaded_weight)
            loaded_bias = ts.ckks_vector_from(context, loaded_bias)
            model = EncryptedLogisticRegression(context, weight=loaded_weight, bias=loaded_bias)
        else:
            model = PlainLogisticRegression(context, weight=loaded_weight, bias=loaded_bias)

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
    if client_id not in models:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id not in models[client_id][PLAIN] and model_id not in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model doesn't exists"}), 404
    x_eval = request.json["x_eval"]
    if x_eval is None:
        return jsonify(
            {"error": "Send all required json data: x_eval"}), 400
    x_eval = torch.tensor(x_eval, dtype=default_float)

    folder = ENCRYPTED if model_id in models[client_id][ENCRYPTED] else PLAIN
    with open(f"{STORAGE_URL}/{client_id}/{client_id}.bin", 'rb') as ctx_file:
        ctx_data = pickle.load(ctx_file)
        context = ts.context_from(ctx_data)
        del ctx_data
        gc.collect()

    if folder == ENCRYPTED:
        x_eval = [ts.ckks_vector(context, x) for x in x_eval]

    with open(f"{STORAGE_URL}/{client_id}/{folder}/{model_id}.bin", 'rb') as model_file:
        loaded_weight, loaded_bias = pickle.load(model_file)
        if folder == ENCRYPTED:
            loaded_weight = ts.ckks_vector_from(context, loaded_weight)
            loaded_bias = ts.ckks_vector_from(context, loaded_bias)
            model = EncryptedLogisticRegression(context, weight=loaded_weight, bias=loaded_bias)
        else:
            model = PlainLogisticRegression(context, weight=loaded_weight, bias=loaded_bias)

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


@application.route("/delete_model/<client_id>/<model_id>", methods=["DELETE"])
def delete_model(client_id, model_id):
    if client_id is None or model_id is None:
        return jsonify({"error": "Send all required data: client ID, model ID"}), 400
    if client_id not in models:
        return jsonify({"error": "Client doesn't exist, please register"}), 404
    if model_id not in models[client_id][PLAIN] and model_id not in models[client_id][ENCRYPTED]:
        return jsonify({"error": "Model doesn't exist"}), 404

    if model_id in models[client_id][ENCRYPTED]:
        model_path = f"{STORAGE_URL}/{client_id}/{ENCRYPTED}/{model_id}.bin"
        if os.path.exists(model_path):
            os.remove(model_path)
        models[client_id][ENCRYPTED].remove(model_id)
        active[client_id].pop(model_id, None)

    if model_id in models[client_id][PLAIN]:
        model_path = f"{STORAGE_URL}/{client_id}/{PLAIN}/{model_id}.bin"
        if os.path.exists(model_path):
            os.remove(model_path)
        models[client_id][PLAIN].remove(model_id)
        active[client_id].pop(model_id, None)

    return jsonify({"message": "Model deleted successfully"}), 200


if __name__ == "__main__":
    torch.serialization.add_safe_globals([initialize])
    initialize()
    application.run(debug=True)
