import os

import pickle
import shutil

from collections import defaultdict
from tqdm import tqdm


def aggregate_results_one_folder(world_size, description, output_dir):
    aggregated_results = []

    for rank in range(world_size):
        # output_path = f"{output_dir}/{description}_{rank}.pkl"
        output_path = f"{output_dir}/{rank}.pkl"
        with open(output_path, "rb") as f:
            aggregated_results.extend(pickle.load(f))

    # Save the final aggregated results
    final_output_path = f"{output_dir}/{description}-final_output.pkl"
    with open(final_output_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Aggregated results saved to {final_output_path}")


def aggregate_results(world_size, description, output_dir, base_dir):
    aggregated_results = []

    aggregate_results_per_label = defaultdict(list)

    for rank in tqdm(range(world_size), desc=f"Processing {description}"):
        output_path = os.path.join(output_dir, f"{rank}.pkl")

        if not os.path.exists(output_path):
            print(f"Warning: {output_path} does not exist. Skipping...")

            continue

        with open(output_path, "rb") as f:
            results = pickle.load(f)

            aggregated_results.extend(results)

            print(len(results))

            for entry in results:
                labels = entry.get("labels", [])

                for label in labels:
                    aggregate_results_per_label[label].append(entry)

    # Save the final aggregated results

    final_output_path = os.path.join(base_dir, f"{description}.pkl")
    final_output_label_path = os.path.join(base_dir, f"{description}_label.pkl")

    print(len(aggregated_results))

    with open(final_output_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(len(aggregate_results_per_label))

    with open(final_output_label_path, "wb") as f:
        pickle.dump(aggregate_results_per_label, f)

    print(f"Aggregated results saved to {final_output_path}")


if __name__ == "__main__":
    world_size = 8

    # # for on folder
    # input_dir = output_dir = "/mnt/data/ruian/internvl2/pkls/no_label/"
    # input_dir = output_dir = "/mnt/data/ruian/internvl2/pkls/with_label/"
    # input_dir = output_dir = "/mnt/data/ruian/internvl2/pkls/test/checkpoint-295691/"

    # description = output_dir.split("/")[-2]
    # aggregate_results_one_folder(world_size, description, output_dir)

    # for nested folders
    base_dir = "/mnt/data/ruian/internvl2/pkls/test"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/9b_12tiles_label"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/26b_6tiles_label"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/26b_6tiles_no_label"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/9b_6titles_label_hardcases"
    # base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/9b_6titles_label_hardcases_fewer"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/9b_mpo"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/all_on_136_ckpt1_1/"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/all_on_136_0202"

    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/26b_all_on_filter_0202"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/26b_batch1_on_filter_0203"
    base_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/26b_lunglesion_on_filter_0203"

    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if os.path.isdir(full_path):  # Ensure it's a directory
            description = subdir
            print(f"Processing directory: {description}")
            aggregate_results(world_size, description, full_path, base_dir)

            # shutil.rmtree(full_path)
            # print(f"Removed directory: {full_path}")
