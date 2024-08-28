import sys
sys.path.append('/root/projects/registry/mimic')

from mimic_two_dataset_helper import MimicTwoDatasetHelper


path = "/root/projects/local_mnt/mimic2-jpg/mimic-cxr-jpg-2.1.0.physionet.org"
mimic_two_dataset_helper = MimicTwoDatasetHelper(dataset_path=path, labels_generator="chexpert", group_multi_images=True)

train_dataset = mimic_two_dataset_helper.get_torch_train_dataset()
test_dataset = mimic_two_dataset_helper.get_torch_test_dataset()
validation_dataset = mimic_two_dataset_helper.get_torch_validation_dataset()

def separate_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Split the text into lines
    lines = text.split("\n")

    # Initialize variables
    part1 = []
    part2 = []
    found_separator = False

    # Loop through the lines and separate the text
    for line in lines:

        if not line or line.isspace() or line == "\n":
            continue

        if "findings" in line or "impression" in line:
            found_separator = True
        if found_separator:
            part2.append(line)
        else:
            part1.append(line)

    # Join the parts back into strings
    part1 = "\n".join(part1).strip()
    part2 = "\n".join(part2).strip()

    return part1, part2

import os
num_threads = os.cpu_count()

import json
from concurrent.futures import ThreadPoolExecutor


# Function to write a single line to the jsonl file
def write_line(data, file_path):
    json_line = json.dumps(data)
    with open(file_path, 'a') as f:
        f.write(json_line + '\n')

# Function to generate the jsonl lines and write them in parallel
def generate_jsonl(dataset, type="train"):
    print(f"Generating {type} dataset jsonl file...")
    file_path = f"/root/projects/InternVL-Epsi/output/jsonl2/{type}_dataset.jsonl"
    print(f"Saving file to {file_path}")

    def process_entry(entry):
        report_path = entry['report'][0]
        with open(report_path, 'r') as file:
            report = file.read()

        part1, part2 = separate_text(report)

        topics = entry["labels_names"]
        flattened_list = list(set([item for sublist in topics for item in sublist]))
        topics = ", ".join(flattened_list)

        query = f"Patient topics: {topics}.\nClinic notes {part1}.\nGenerate impression and findings for the x-rays: "

        row = {
            "query": query,
            "response": part2,
            "images": entry['image_file']
        }

        # Submit the row to be written to the jsonl file
        executor.submit(write_line, row, file_path)

    # Use ThreadPoolExecutor to write the data lines in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for entry in dataset:
            process_entry(entry)

    print("Finished writing data to jsonl file.")


generate_jsonl(validation_dataset, type="validation")
generate_jsonl(test_dataset, type="test")
generate_jsonl(train_dataset, type="train")