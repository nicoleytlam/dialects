import csv
import random
import os

def csv_to_datasets(csv_file, output_dir="data", train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """
    Convert a CSV file into text format and split it into train, dev, and test datasets.

    :param csv_file: Input CSV file name (with three columns: Sentence, Question, Answer).
    :param output_dir: Directory where output files will be saved.
    :param train_ratio: Proportion of data for training set.
    :param dev_ratio: Proportion of data for development set.
    :param test_ratio: Proportion of data for test set.
    """
    # Ensure ratios sum to 1
    assert train_ratio + dev_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Read the CSV and transform the data
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip the header row if present
        data = []
        for row in reader:
            row = row[:3]
            sentence, question, answer = row
            data.append(f"{sentence} {question}\t{answer}")

    # Shuffle data
    random.shuffle(data)

    # Check dataset size
    if len(data) < 10:
        raise ValueError("Dataset too small to split meaningfully.")

    # Split the data
    total = len(data)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_data = data[:train_end]
    dev_data = data[train_end:dev_end]
    test_data = data[dev_end:]

    # Write to files
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    train_file = os.path.join(output_dir, f"{base_name}.train")
    dev_file = os.path.join(output_dir, f"{base_name}.dev")
    test_file = os.path.join(output_dir, f"{base_name}.test")

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(train_data))

    with open(dev_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(dev_data))

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(test_data))

    print(f"Train, dev, and test files created in: '{output_dir}'")

# Example usage
# csv_to_datasets('dialect.csv', output_dir='output')
