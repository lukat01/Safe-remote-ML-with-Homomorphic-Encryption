import csv
import random


def generate_csv(filename, num_rows, num_cols):
    if num_cols < 1:
        raise ValueError("Number of columns must be at least 1")

    header = [f'v{i + 1}' for i in range(num_cols)] + ['l']
    rows = []

    for _ in range(num_rows):
        row = [random.randint(1, 100) for _ in range(num_cols)]
        row.append(random.randint(0, 1))
        rows.append(row)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


generate_csv('data/1000.csv', 100, 999)
