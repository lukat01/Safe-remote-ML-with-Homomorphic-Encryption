import os
import random
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from homomorphic_lr import default_float

random.seed(73)
torch.random.manual_seed(73)


def preprocessing_data(file=None, label=None, columns_remove=None, ratio=0.3, row_num=None, stratify=False,
                       model_id="", client_id="", for_prediction=False, vector=None, double=False):
    if vector:
        stored_mean, stored_std, stored_label, stored_columns = get_model_data(model_id, client_id)
        df = pd.DataFrame([vector], columns=stored_columns)
        df = (df - stored_mean) / stored_std
        df = torch.tensor(df.values, dtype=default_float)
        return df

    data = pd.read_csv(file)
    data = data.dropna()
    data = data.drop(columns=columns_remove)

    if for_prediction:
        stored_mean, stored_std, _, _ = get_model_data(model_id, client_id)
        og_data = data.values
        data = (data - stored_mean) / stored_std
        x = torch.tensor(data.values, dtype=default_float)
        return {"x_test": x, "og_x_test": og_data}

    grouped = data.groupby(label)

    if not for_prediction:
        if row_num is not None:
            min_samples_per_class = min(row_num // len(grouped), grouped.size().min())
            data = grouped.apply(lambda a: a.sample(min_samples_per_class, random_state=73)).reset_index(drop=True)
        else:
            min_samples = grouped.size().min()
            data = grouped.apply(lambda a: a.sample(min_samples, random_state=73)).reset_index(drop=True)

    y = torch.tensor(data[label].values, dtype=default_float).unsqueeze(1)
    data = data.drop(columns=[label])
    mean = data.mean()
    std = data.std()
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values, dtype=default_float)

    data_for_file = pd.DataFrame({
        "mean": mean,
        "std": std,
        "label": label,
        "columns": data.columns.tolist()
    })
    data_for_file.to_csv(f'Client/clients/{client_id}/models/{model_id}.csv')
    if double:
        data_for_file.to_csv(f'Client/clients/{client_id}/models/{model_id}_plain.csv')

    result = dict()
    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio,
                                                            stratify=y.numpy(), random_state=73)
        result["x_train"] = x_train
        result["x_test"] = x_test
        result["y_train"] = y_train
        result["y_test"] = y_test
    else:
        indexes = [i for i in range(len(x))]
        random.shuffle(indexes)
        delimiter = int(len(x) * ratio)
        test_indexes, train_indexes = indexes[:delimiter], indexes[delimiter:]
        result["x_train"] = x[train_indexes]
        result["x_test"] = x[test_indexes]
        result["y_train"] = y[train_indexes]
        result["y_test"] = y[test_indexes]

    result["num_features"] = x.shape[1]
    return result


def get_model_data(model_id, client_id):
    file_path = f'Client/clients/{client_id}/models/{model_id}.csv'
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    data_from_file = pd.read_csv(file_path)

    mean = data_from_file["mean"]
    std = data_from_file["std"]
    label = data_from_file["label"].iloc[0]
    columns = data_from_file["columns"].tolist()

    mean = pd.Series(mean.values, index=columns)
    std = pd.Series(std.values, index=columns)
    return mean, std, label, columns
