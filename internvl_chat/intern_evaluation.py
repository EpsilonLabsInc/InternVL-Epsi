import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torchvision.transforms as T
from internvl.model.internvl_chat import InternVLChatModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer

import pydicom
from google.cloud import storage
from io import BytesIO

# Display the Python path
sys.path.append("/root/projects/InternVL-Epsi/internvl_chat")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dcm_2_rgb(dcm_data, image_path):
    if hasattr(dcm_data, 'pixel_array'):
        pixel_array = dcm_data.pixel_array
    else:
        print("111", image_path)
    # pixel_array = dcm_data.pixel_array

    # Normalize the pixel values to the range 0-255
    # The pixel values in a DICOM file may not be in the 0-255 range, so normalization is needed
    pixel_array_normalized = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
    pixel_array_normalized = pixel_array_normalized.astype(np.uint8)

    # Convert grayscale DICOM data to an RGB image by stacking the array 3 times (R, G, B channels)
    rgb_array = np.stack([pixel_array_normalized]*3, axis=-1)

    # Convert the NumPy array to a PIL Image
    rgb_image = Image.fromarray(rgb_array)

    rows = dcm_data.Rows
    cols = dcm_data.Columns
    # 1.6M pixels seems to cause issue of OOM during training
    if rows * cols > 16000000:
        # Compress the image by resizing by a factor of 2
        new_size = (cols // 2, rows // 2)
        rgb_image = rgb_image.resize(new_size, Image.Resampling.LANCZOS)

    return rgb_image

def load_image(image_file, input_size=448, max_num=12):
    if 'dcm' in image_file:
        dcm_data = get_dcm_from_bucket(image_file)
        image = dcm_2_rgb(dcm_data, image_file)
    else:
        image = Image.open(image_file).convert("RGB")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


test_jsonl = "./test_dataset_converted.jsonl"
test_jsonl = "/mnt/data/ruian/gradient/22JUL2024/dev_test_dataset_corrected_text_5000_test_0916.jsonl"
test_jsonl = "/mnt/data/ruian/mimic2/gpt/test_dataset_gpt_labels.jsonl"
test_jsonl = "/mnt/data/ruian/mimic2/gpt/test_dataset_gpt_labels_per_label.jsonl"


generation_config = dict(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.1,
    top_k=100,
    num_beams=2,
    repetition_penalty=1.5,
)

def get_dcm_from_bucket(gcp_bucket_path):
    base = "gs://epsilon-data-us-central1/"
    gcp_bucket_path = base + gcp_bucket_path

    path_parts = gcp_bucket_path.split("/")
    bucket_name = path_parts[2]
    blob_path = "/".join(path_parts[3:])

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    dicom_data = blob.download_as_bytes()

    dicom_file = pydicom.dcmread(BytesIO(dicom_data))

    return dicom_file


def generate_output(dataset_jsonl, model, tokenizer, output_path):
    with open(dataset_jsonl, "r") as file:
        results = []
        times = []

        # Wrap the loop with tqdm and set total to 100
        for idx, line in enumerate(tqdm(file, total=100, desc="Processing")):
            # Parse the line as a JSON object
            start_time = time.time()
            entry = json.loads(line)

            # Process the JSON object (e.g., print it)
            image_paths = entry["image"]
            # image_paths = [
            #     each.replace(
            #         "projects/local_mnt/mimic2-jpg", "projects/data/mimic2-jpg"
            #     )
            #     for each in image_paths
            # ]
            entry["image"] = image_paths

            pixel_values_list = [
                load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                for image_path in image_paths
            ]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            num_patches_list = [
                pixel_values.size(0) for pixel_values in pixel_values_list
            ]

            query, truth_report = entry["conversations"]
            query = query["value"]
            truth_report = truth_report["value"]

            try:
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    query,
                    generation_config,
                    num_patches_list=num_patches_list,
                )
            except Exception as e:
                print(f"Error: {e}")
                continue

            result = {"truth": truth_report, "generated": response}

            results.append(result)
            end_time = time.time()
            times.append(end_time - start_time)

            # Break the loop after processing 100 entries
            if len(results) >= 100:
                with open(output_path, "wb") as f:
                    pickle.dump(results, f)
                break

        avg_time = np.mean(times)
        top_90_time = np.percentile(times, 90)
        top_95_time = np.percentile(times, 95)
        top_99_time = np.percentile(times, 99)

        print(f"Average time per loop: {avg_time:.4f} seconds")
        print(f"90th percentile time: {top_90_time:.4f} seconds")
        print(f"95th percentile time: {top_95_time:.4f} seconds")
        print(f"99th percentile time: {top_99_time:.4f} seconds")


if __name__ == "__main__":


    if len(sys.argv) < 2:
        print("Usage: python3 -m intern_evaluation.py <description>")
        sys.exit(1)

    description = sys.argv[1]

    output_dir = f"./output/internvl/{description}"

    if os.path.exists(output_dir):
        user_input = input(f"The directory '{output_dir}' already exists. Do you want to continue? (y/n): ").strip().lower()

        if user_input != 'y':
            print("Exiting the script.")
            sys.exit(1)  # Exit with a status of 1 indicating cancellation by the user
    else:
        # Proceed with creating the directory if needed or continue processing
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
        print(f"Directory '{output_dir}' created.")

    folder = "/mnt/data/ruian/internvl2/"

    # checkpoint_dir = "/mnt/data/ruian/internvl2/has_weak_label_1e-7/"
    # checkpoint_dir = "/mnt/data/ruian/internvl2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240822_064120_1e-6_no_weaklabel/"
    # checkpoint_dir = "/mnt/data/ruian/internvl2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240822_191911_1e-5_no_weaklabel/"
    # checkpoint_dir = "/mnt/data/ruian/internvl2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240823_070315_1e-4_no_weaklabel/"
    # checkpoint_dir = "/mnt/data/ruian/internvl2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240823_165349_1e-4_no_weaklabel"
    # checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240824_205017_5e-5_no_weaklabel"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240826_170902_1e-4_no_weaklabel"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240827_062836_1e-5_has_weaklabel"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240828_161003_1e-5_has_weaklabel"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240829_171802_1e-4_has_weaklabel"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240831_033438_1e-4_has_corrected_label"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240901_040154_1e-5_has_corrected_label"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240917_195139_1e-5_gradient_chest_XR_no_label"
    # checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20240902_064729_5e-5_has_corrected_label"
    checkpoint_dir = folder + "internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20241008_210554_1e-5_mimic_gpt"

    checkpoints = sorted(
        [
            os.path.join(checkpoint_dir, ckpt)
            for ckpt in os.listdir(checkpoint_dir)
            if ckpt.startswith("checkpoint-")
        ]
    )

    print(22222222)
    print(checkpoints)

    # checkpoints = ["OpenGVLab/InternVL2-8B"]

    for checkpoint in checkpoints[-1:]:

        path_prefix = "/".join(checkpoint.split("/")[-1:])

        output_path = f"{output_dir}/{path_prefix}.pkl"

        if os.path.exists(output_path):
            print(f"File {output_path} already exists, skipping.")
            continue

        print(f"Loading model from {checkpoint}>>>")

        model = InternVLChatModel.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, use_fast=False
        )

        print(f"Generating evaluation output loaded from {checkpoint}.\n")


        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        generate_output(test_jsonl, model, tokenizer, output_path)
        print(f"Done for checkpoint {checkpoint}<<<")

        del model
        torch.cuda.empty_cache()
