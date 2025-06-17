import csv
import sys
import os
from statistics import mean, stdev


def parse_results(dataset_name, filenames):
    base_path = os.path.join("legalbench", "results", dataset_name)
    output_file = os.path.join(base_path, f"overall_{dataset_name}.csv")

    results = []

    for filename in filenames:
        path = os.path.join(base_path, filename)
        values = []

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        value = float(row[-1])  # Assumes result_or_error is last column
                        values.append(value)
                    except ValueError:
                        continue  # Skip header or invalid rows

            if values:
                avg = mean(values)
                std = stdev(values) if len(values) > 1 else 0.0
                results.append((filename, avg, std))
            else:
                results.append((filename, "", ""))  # No valid values

        except FileNotFoundError:
            print(f"File not found: {path}")
            results.append((filename, "", ""))  # Error case

    # Write to CSV
    with open(output_file, "w", newline='', encoding="utf-8") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["filename", "avg", "std_dev"])
        for row in results:
            writer.writerow(row)

    print(f"CSV results written to: {output_file}")


# === Usage example ===
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python accuracy.py <dataset_name> <file1.csv> [<file2.csv> ...]")
        sys.exit(1)

    dataset = sys.argv[1]
    files = sys.argv[2:]
    parse_results(dataset, files)
