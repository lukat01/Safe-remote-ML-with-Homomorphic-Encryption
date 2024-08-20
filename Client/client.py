import argparse
from homomorphic_lr import *
from preprocessing import preprocessing_data, get_model_data


def print_diffs(tensors, labels):
    if len(tensors) < 2:
        print("Not enough tensors to compare.")
        return

    tensor_lists = [tensor.tolist() for tensor in tensors]
    tensor_lengths = [len(tensor_list) for tensor_list in tensor_lists]
    if not all(length == tensor_lengths[0] for length in tensor_lengths):
        print("All tensors must have the same length.")
        return

    for index in range(len(tensor_lists[0])):
        predictions = [tensor_list[index] for tensor_list in tensor_lists]
        diffs = [
            f"diff({labels[0]}, {labels[i]}): {abs(predictions[0][0] - predictions[i][0])}"
            for i in range(1, len(predictions))
        ]
        print(', '.join(f"{label}: {predictions[i][0]}" for i, label in enumerate(labels)) + ', ' + ', '.join(diffs))


def main(args):
    client_id = args.client_id
    model_id = args.model_id
    row_num = args.row_num
    file = args.file
    label = args.label
    columns_remove = args.columns_remove
    num_iterations = args.num_iterations
    operations = args.operations
    vector = args.vector
    encrypt_vector = args.encrypt_vector

    if "info" in operations:
        _, _, stored_columns, stored_label = get_model_data(model_id, client_id)
        print(f"Columns: {stored_columns}")
        print(f"Number of features: {len(stored_columns)}")
        print(f"Label: {stored_label}")
        return

    if "del" in operations:
        message = delete_model(client_id, model_id)
        print(message)
        return

    if "all" in operations:
        operations = ["reg", "pp", "pe", "ee"]

    context = registration(client_id) if "reg" in operations else get_context(client_id)

    if vector:
        print(encrypt_vector)
        # test vector: 1 39 0.0 0 0 195.0 106.0 80.0 77.0 => 0
        pp_vector = preprocessing_data(model_id=model_id, client_id=client_id, vector=vector)
        result = predict_single_entry(client_id, model_id,
                                      context, pp_vector, encrypt_vector)
        print(f"{vector} => {result}")
        return

    double = any(op in operations for op in ("pp", "pe")) and any(op in operations for op in ("ee", "ep"))

    if any(op in operations for op in ["pp", "pe", "ee", "ep"]):
        print("Preprocessing data")
        data_to_send = preprocessing_data(
            file=f"./Client/data/{file}",
            label=label,
            columns_remove=columns_remove,
            row_num=row_num,
            model_id=model_id,
            client_id=client_id,
            double=double
        )
        print("Data preprocessing completed\n")
    elif any(op in operations for op in ["p", "e"]) and file:
        print("Preprocessing data for evaluation")
        data_to_send = preprocessing_data(
            file=f"./Client/data/{file}",
            label=label,
            columns_remove=columns_remove,
            for_prediction=True,
            model_id=model_id,
            client_id=client_id
        )
        print("Data preprocessing completed\n")
    else:
        return

    def run_training_and_testing(train_model, enc_train=False, enc_test=False, calc_accuracy=True):
        if train_model:
            training(
                client_id,
                model_id + ("_plain" if double and not enc_train else ""),
                context,
                data_to_send["x_train"],
                data_to_send["y_train"],
                data_to_send["num_features"],
                num_iterations,
                enc_train,
                double
            )

        predictions = testing(
            client_id,
            model_id + ("_plain" if double and not enc_train else ""),
            context,
            data_to_send["x_test"],
            enc_test
        )
        if calc_accuracy:
            accuracy = calculate_accuracy(data_to_send["y_test"], predictions)
            print(f"Accuracy for {'encrypted' if enc_train else 'plain'} training and "
                  f"{'encrypted' if enc_test else 'plain'} evaluation: {accuracy * 100}%\n")
        return predictions

    labels = []
    tensors = []
    if "pp" in operations:
        # Plain training and plain evaluation
        pp = run_training_and_testing(train_model=True, enc_train=False, enc_test=False)
        labels.append("pp")
        tensors.append(pp)

    if "pe" in operations:
        # Plain training and encrypted evaluation
        pe = run_training_and_testing(train_model=False if 'pp' in operations else True,
                                      enc_train=False, enc_test=True)
        labels.append("pe")
        tensors.append(pe)

    if "ee" in operations:
        # Encrypted training and encrypted evaluation
        ee = run_training_and_testing(train_model=True, enc_train=True, enc_test=True)
        labels.append("ee")
        tensors.append(ee)

    if "ep" in operations:
        # Encrypted training and plain evaluation
        ep = run_training_and_testing(train_model=False if 'ee' in operations else True,
                                      enc_train=True, enc_test=False)
        labels.append("ep")
        tensors.append(ep)

    if data_to_send["x_test"].shape[0] <= 50 and len(tensors) >= 2:
        print_diffs(tensors, labels)

    if "p" in operations:
        # Plain evaluation
        result = run_training_and_testing(train_model=False, enc_train=True,
                                          enc_test=False, calc_accuracy=False)
        for x, y in zip(data_to_send["og_x_test"], result):
            print(f"{x.tolist()} => {1 if y > 0.5 else 0}")

    if "e" in operations:
        # Encrypted evaluation
        result = run_training_and_testing(train_model=False, enc_train=True,
                                          enc_test=True, calc_accuracy=False)
        for x, y in zip(data_to_send["og_x_test"], result):
            print(f"{x.tolist()} => {1 if y > 0.5 else 0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run homomorphic logistic regression with various configurations.")

    parser.add_argument("--client_id", type=str, default="client", help="Client ID")
    parser.add_argument("--model_id", type=str, default="model", help="Model ID")
    parser.add_argument("--row_num", type=int, default=None,
                        help="Number of rows to process (use all rows if not specified)")
    parser.add_argument("--file", type=str, default=None,
                        help="Data file path (use default label and columns_remove if not specified)")
    parser.add_argument("--label", type=str, help="Label column name")
    parser.add_argument("--columns_remove", type=str, nargs="*", default=[],
                        help="Columns to remove from data")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of iterations for training")
    parser.add_argument("--operations", type=str, nargs="+",
                        choices=["reg", "pp", "pe", "ee", "ep", "all", "e", "p", "info", "del"], default=["all"],
                        help="Operation to run")
    parser.add_argument("--vector", type=float, nargs="+", default=None,
                        help="Vector for single prediction")
    parser.add_argument("--encrypt_vector", action='store_true',
                        help="Flag that indicates that the vector should be encrypted")

    arguments = parser.parse_args()

    contains_e_or_p = any(item in {"e", "p"} for item in arguments.operations)
    if contains_e_or_p:
        if not all(item in {'e', 'p'} for item in arguments.operations):
            parser.error("if e or p are in the operations they are only ones allowed")

    if arguments.file:
        if not arguments.label and not contains_e_or_p:
            parser.error("--file is provided, but --label is missing")
    elif not arguments.vector:
        arguments.file = "data.csv"
        arguments.label = "TenYearCHD"
        arguments.columns_remove = ["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"]

    if arguments.num_iterations <= 0:
        parser.error("Number of iterations must be positive integer")

    main(arguments)
