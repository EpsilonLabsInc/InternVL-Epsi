import json
import os
import pickle

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm
from time import time


def load_cache(cache_file):
    """Load the cache file if it exists."""

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    return {}


def save_cache(cache, cache_file):
    """Save the cache to the file."""
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)


def separate_text_rules(text):
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

        if "reason for examination" in line:
            part1.append(line)
            found_separator = True
        else:
            if found_separator:
                part2.append(line)
            else:
                part1.append(line)

    # Join the parts back into strings
    part1 = "\n".join(part1).strip()
    part2 = "\n".join(part2).strip()

    return {"gpt_clinic_note": part1, "gpt_impressions_and_findings": part2}


def separate_text_gpt(
    report, client, model="gpt-4o-mini", max_tokens=1024, temperature=0
):  # Low temperature to prioritize accuracy
    """Sends the report content to the Azure OpenAI API to get labels."""
    template_dict = {
        "clinic note": "text for clinic notes",
        "impressions_and_findings": "text for impressions and findings",
    }

    system_message = f"""You are an assistant that divides radiology reports. Divide the reports into (1) clinic notes (2) impressions and findings. Do not change any wording. Output in json format with a template {template_dict}."""
    # Construct messages for API call
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": report},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    # token_count = response.usage.total_tokens

    # Text of the group selected by the user
    parts = response.choices[0].message.content
    parts = json.loads(parts)

    return parts


def separate_report(report, client):
    try:
        parts = separate_text_gpt(report, client)
    except Exception as e:
        print(f"Error: {e}")
        parts = separate_text_rules(report)

    return parts


def read_report(file_path):
    """Reads the content of a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def process_report(row, client, cache, cache_file_name):
    report = read_report(row["report"])
    # report = row["report_text"]

    cached_result = cache.get(report)

    if cached_result is not None:
        parts = cached_result
    else:
        parts = separate_report(report, client)

        cache[report] = parts

        save_cache(cache, cache_file_name)

    return pd.Series(
        {
            "prompt": parts.get("clinic note", ""),
            "impressions_and_findings": parts.get("impressions_and_findings", ""),
        }
    )


def process_report_parallel(df_chunk, cache_file_name, client, pbar=None):
    print(f">>>>>>> loading cache from {cache_file_name}")
    cache = load_cache(cache_file_name)
    print(f"<<<<<<< loaded cache with {len(cache)} entries from {cache_file_name}")

    result = df_chunk.apply(
        lambda row: process_report(row, client, cache, cache_file_name), axis=1
    )

    if pbar is not None:
        pbar.update(len(df_chunk))

    return result


# Function to parallelize the DataFrame processing
def parallelize_dataframe(df, func, client, cache_file_name, n_cores=4):
    # Split the DataFrame into chunks
    df_split = np.array_split(df, n_cores)

    # Create a shared tqdm progress bar
    with tqdm(total=len(df), desc="Processing rows") as pbar:
        results = []
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            # Submit tasks and pass the progress bar to each
            futures = [
                executor.submit(
                    func, chunk, f"{cache_file_name}_part{i}.pkl", client, pbar
                )
                for i, chunk in enumerate(df_split)
            ]

            for future in as_completed(futures):
                results.append(future.result())

    # Combine the results from all the futures
    return pd.concat(results)


def get_source_df(source="mimic2"):

    print(f"Loading data from {source}")

    if source == "mimic2":
        file_path = "/mnt/data/ruian/mimic2_removed_previous_corrected_labels_0830.pkl"
        df_all = pd.read_pickle(file_path)
    elif source == "gradient_chest_xr":
        file_path = (
            "/mnt/data/ruian/gradient/chext_XR/REPORTS/CR/us-data-cr-23july2024.csv"
        )
        df_all = pd.read_csv(
            file_path,
            delimiter=",",  # Specify comma as the delimiter
            quotechar='"',  # Handle quotes around strings properly
            escapechar="\\",  # Escape special characters
            na_values="NA",  # Handle 'NA' as missing values
            skipinitialspace=True,  # Skip any space after the delimiter
            engine="python",  # Use the Python engine for more flexibility
        )
    return df_all


if __name__ == "__main__":
    # cache_file = "/mnt/data/ruian/gpt_cache/mimic_report_separation_cache.pkl"
    cache_file_name = "/mnt/data/ruian/gpt_cache/mimic_report_separation_cache"
    # cache_file_name = "/mnt/data/ruian/gpt_cache/gradient_xr_separation_cache"

    # client = AzureOpenAI(
    #     azure_endpoint="https://epsilon-eastus.openai.azure.com/openai/deployments/epsilon-mini-4o/chat/completions?api-version=2024-02-15-preview",
    #     api_key="9b568fdffb144272811cb5fad8b584a0",
    #     api_version="2024-02-15-preview",
    # )

    client = AzureOpenAI(
        azure_endpoint="https://epsilon-eastus.openai.azure.com/openai/deployments/epsilon-mini-4o/chat/completions?api-version=2024-02-15-preview",
        api_key="9b568fdffb144272811cb5fad8b584a0",
        # api_version="2024-04-01-preview",
        api_version="2023-03-15-preview"
    )

    source = "mimic2"
    # source = "gradient_chest_xr"

    df_all = get_source_df(source)

    # df_test = df_all.head(256 * 4)
    df_test = df_all

    # df[["prompt", "impressions_and_findings"]] = df.apply(process_report, axis=1)
    # print(df.head())

    # Process the DataFrame in parallel
    t0 = time()
    df_results = parallelize_dataframe(
        df_test, process_report_parallel, client, cache_file_name, n_cores=8
    )
    t1 = time()
    print(f"Processing time: {t1 - t0:.2f}s")

    # Now `df_results` has the new columns with the extracted information

    df_test = pd.concat([df_test, df_results], axis=1)
    print(df_test.head())

    df_test.to_pickle(
        f"/mnt/data/ruian/{source}_removed_previous_corrected_labels_gpt_separation_0901_{len(df_test)}.pkl"
    )
