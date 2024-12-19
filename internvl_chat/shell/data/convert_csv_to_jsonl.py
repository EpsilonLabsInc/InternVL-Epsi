import json
import re

import pandas as pd
from tqdm import tqdm


# gradient_cr = pd.read_csv("/mnt/data/ruian/sav/cleaned_CRs/GRADIENT_CR_batch_1.csv")
mimic2_df = pd.read_csv("/mnt/data/ruian/sav/cleaned_CRs/MIMIC2.csv")
# eufy = pd.read_csv("/mnt/data/ruian/sav/cleaned_CRs/eufy.csv")


def construct_query(row, old_entry):
    clinical_data = row["clinical_data"].values[0] if not row["clinical_data"].values[0] == 'Not available.' else ""
    exam = row["exam"].values[0] if not row["exam"].values[0] == 'Not available.' else ""
    comparison = row["comparison"].values[0] if not row["comparison"].values[0] == 'None found.' else ""

    findings = row["findings"].values[0] if not row["findings"].values[0] == 'Not available.' else ""
    impression = row["impression"].values[0]

    raw_report = row["report_text"].values[0]
    gender = ""
    if "male" in raw_report.lower() or "man" in raw_report.lower():
        gender = "male"
    elif "female" in raw_report.lower() or "woman" in raw_report.lower():
        gender = "female"
    elif "_m with" in raw_report.lower():
        gender = "male"
    elif "_f with" in raw_report.lower():
        gender = "female"

    labels = row["labels"].values[0]

    old_prompt = old_entry["conversations"][0]["value"]

    prompt = "".join(re.findall(r"<image>", old_prompt))

    if gender:
        prompt += f"Gender: {gender}\n"
    if exam:
        prompt += f"Xray exam: {exam}\n"
    if clinical_data:
        prompt += f"Clinical_data: {clinical_data}\n"
    if labels:
        prompt += f"Focus is {labels}"

    report = f"Findings: {findings}\nImpression: {impression}"

    return prompt, report


def convert_json_to_jsonl(input_json, output_json):

    print("input_json: ", input_json)
    print("output_json: ", output_json)

    with open(output_json, "w") as file_out:
        with open(input_json, "r") as file:
            # for line in file:
            for line in tqdm(file, desc="Processing lines"):
                entry = json.loads(line)

                image_path_0 = entry["image"][0]
                patient_id = int(image_path_0.split("/")[-3].strip("p"))
                study_id = int(image_path_0.split("/")[-2].strip("s"))

                row = mimic2_df[
                    (mimic2_df["patient_id"] == patient_id)
                    & (mimic2_df["study_uid"] == study_id)
                ]

                if row["findings"].values[0] == 'Not available.':
                    continue

                query, report = construct_query(row, entry)

                entry["conversations"][0]["value"] = query
                entry["conversations"][1]["value"] = report

                json.dump(entry, file_out)
                file_out.write("\n")


if __name__ == "__main__":
    
    suffix = "sav_valid_findings"
    
    input_json = "/root/projects/InternVL-Epsi/output/jsonl/mimic2/gpt/train_dataset_gpt_labels_non_empty.jsonl"
    output_json = f"/root/projects/InternVL-Epsi/output/jsonl/mimic2/gpt/train_dataset_gpt_labels_non_empty_{suffix}.jsonl"
    convert_json_to_jsonl(input_json, output_json)

    # input_json = "/root/projects/InternVL-Epsi/output/jsonl/mimic2/gpt/test_dataset_gpt_labels.jsonl"
    # output_json = "/root/projects/InternVL-Epsi/output/jsonl/mimic2/gpt/test_dataset_gpt_labels_sav.jsonl"
    # convert_json_to_jsonl(input_json, output_json)

    # input_json = "/root/projects/InternVL-Epsi/output/jsonl/mimic2/gpt/validation_dataset_gpt_labels.jsonl"
    # output_json = "/root/projects/InternVL-Epsi/output/jsonl/mimic2/gpt/validation_dataset_gpt_labels_sav.jsonl"
    # convert_json_to_jsonl(input_json, output_json)

    input_json = "/mnt/data/ruian/mimic2/gpt/test_dataset_gpt_labels_per_label_10_valid_findings.jsonl"
    output_json = f"/mnt/data/ruian/mimic2/gpt/test_dataset_gpt_labels_per_label_10_{suffix}.jsonl"
    convert_json_to_jsonl(input_json, output_json)
