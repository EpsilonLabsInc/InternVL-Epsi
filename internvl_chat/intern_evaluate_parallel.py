import gc
import json
import os
import pickle
import sys
import time
from io import BytesIO

import numpy as np
import pydicom
import torch
import torch.distributed as dist
import torchvision.transforms as T
# from google.cloud import storage
from internvl.model.internvl_chat import InternVLChatModel
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer


def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


sys.path.append("/home/ruian/projects/InternVL-Epsi/internvl_chat")


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
    if hasattr(dcm_data, "pixel_array"):
        pixel_array = dcm_data.pixel_array
    else:
        print("111", image_path)
    # pixel_array = dcm_data.pixel_array

    # Normalize the pixel values to the range 0-255
    # The pixel values in a DICOM file may not be in the 0-255 range, so normalization is needed
    pixel_array_normalized = (
        (pixel_array - np.min(pixel_array))
        / (np.max(pixel_array) - np.min(pixel_array))
        * 255
    )
    pixel_array_normalized = pixel_array_normalized.astype(np.uint8)

    # Convert grayscale DICOM data to an RGB image by stacking the array 3 times (R, G, B channels)
    rgb_array = np.stack([pixel_array_normalized] * 3, axis=-1)

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


# generation_config = dict(
#     max_new_tokens=1024,
#     do_sample=True,
#     temperature=0.5,
#     top_k=100,
#     num_beams=2,
#     repetition_penalty=1.5,
# )

generation_config = dict(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.5,
    top_k=100,
    num_beams=2,
    repetition_penalty=1.5
)

print("generation_config: ", generation_config)

def get_dcm_from_bucket(gcp_bucket_path, date="22JUL2024"):
    base = f"gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/{date}/"
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

def get_dcm_from_local(local_path):

    # prefix = "/home/ruian/projects/data/gradient/gradient-cxr/22JUL2024/"
    prefix = ""

    dicom_file = pydicom.dcmread(prefix + local_path)

    return dicom_file

def load_image(image_file, input_size=448, max_num=12):
    if "dcm" in image_file:
        # dcm_data = get_dcm_from_bucket(image_file)
        dcm_data = get_dcm_from_local(image_file)
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



def generate_output(lines, model, tokenizer, output_path, rank):
    results = []
    times = []

    for line in tqdm(lines, desc=f"Processing_{rank}"):
        # print('----------------------------')

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
        try:
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

            response = model.module.chat(
                tokenizer,
                pixel_values,
                query,
                generation_config,
                num_patches_list=num_patches_list,
            )
            # print(f"at rank {rank}, Good!!!!!!!!")
        except Exception as e:
            print(f"Error: {e}")
            print(entry)
            # print("Error query:", query)
            # print("Error truth_report:", truth_report)
            continue

        # result = {"idx": entry["idx"], "truth": truth_report, "generated": response}
        # results.append(result)

        # print(">>>")
        # print(truth_report)
        # print("<<<")
        # print(response)

        entry["truth"] = truth_report
        entry["generated"] = response

        results.append(entry)

        end_time = time.time()
        times.append(end_time - start_time)

    # Save results for this rank
    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def aggregate_results(world_size, description, output_dir):
    aggregated_results = []

    for rank in range(world_size):
        output_path = f"{output_dir}/{description}_{rank}.pkl"
        with open(output_path, "rb") as f:
            aggregated_results.extend(pickle.load(f))

    # Save the final aggregated results
    final_output_path = f"{output_dir}/{description}-final_output.pkl"
    with open(final_output_path, "wb") as f:
        pickle.dump(aggregated_results, f)

    print(f"Aggregated results saved to {final_output_path}")



def main():
    # Initialize distributed processing
    print("Initializing distributed processing...")
    init_distributed()
    print("Distributed initialization complete.")


    rank = dist.get_rank()  # Get the rank of the current process
    world_size = dist.get_world_size()  # Total number of processes

    if rank == 0:  # Only rank 0 prints logs
        print(f"Running inference with {world_size} GPUs...")


    # test_jsonl = "/mnt/data/ruian/cr_all3/combined_output_test_1129.jsonl" # with labels
    # test_jsonl = "/mnt/data/ruian/cr_all3/combined_output_test_no_label_1122.jsonl" # no labels
    # test_jsonl = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136.jsonl"
    test_jsonl = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136_nolabel_nebius.jsonl"
    # test_jsonl = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136_system_msg.jsonl"
    # test_jsonl = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/11192024_test_selected_136_system_msg_random_synonym.jsonl"
    # test_jsonl = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/combined_output_test_no_label_01222025_nebius.jsonl"
    # test_jsonl = "/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/combined_output_test_no_label_01222025_nebius_filtered.jsonl"


    checkpoint_dir = "/mnt/data/ruian/internvl2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20241120_063721_1e-5_all_3" # with labels
    # checkpoint_dir = "/mnt/data/ruian/internvl2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20241122_215434_1e-5_all_3"
    checkpoint_dir = "/mnt/data/ruian/internvl2/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_20241126_075221_1e-5_all_3/old"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_8b_finetune_lora_20241221_055656_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_tiles_12"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_26b_finetune_lora_20241229_000315_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm"

    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_8b_finetune_lora_20250105_061340_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_tiles_6_labels_hardcases_500"
    # checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_26b_finetune_lora_20241231_182820_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm"
    # checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_8b_finetune_lora_20250107_220852_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_tiles_6_hardcases_top_500"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_8b_finetune_lora_20250108_233246_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_tiles_6_system_msg"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_8b_MPO_finetune_lora_20250110_043333_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_tiles_6"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_8b_finetune_lora_20250114_194025_1e-5_2.5_gradient_full_rm-no-findings_rm-bad-dcm_tiles_6_system_msg_random_synonym"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_26b_finetune_lora_20250124_030251_1e-5_all_data"
    # checkpoint_dir = "/mnt/gradient_batch123/training/internvl2.5_26b_finetune_lora_20250124_030251_1e-5_all_data"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_26b_finetune_lora_20250128_075408_1e-5_all_data/"
    # checkpoint_dir = "/mnt/gradient_batch123/training/internvl2.5_26b_finetune_lora_20250131_001554_1e-5_all_data/"
    checkpoint_dir = "/mnt/training/internvl2.5_26b_finetune_lora_20241229_184000_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_no_label"
    checkpoint_dir = "/home/ruian/projects/InternVL-Epsi/internvl_chat/training/internvl2.5_26b_finetune_lora_20250205_032537_1e-5_lunglesion"

    if len(sys.argv) < 2:
        print("Usage: python3 -m intern_evaluation.py <description>")
        sys.exit(1)

    description = sys.argv[1]
    # output_dir = f"/mnt/data/ruian/internvl2/pkls/{description}"
    output_dir = f"/home/ruian/projects/InternVL-Epsi/internvl_chat/test_data/pkls/{description}"

    if not os.path.exists(output_dir):
        # Proceed with creating the directory if needed or continue processing
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
        print(f"Directory '{output_dir}' created.")

    checkpoints = sorted(
        [
            os.path.join(checkpoint_dir, ckpt)
            for ckpt in os.listdir(checkpoint_dir)
            if ckpt.startswith("checkpoint-")
        ],
        key=lambda x: int(x.split("-")[-1])
    )

    print(f"Found {len(checkpoints)} checkpoints to evaluate. They are:")
    print(checkpoints)

    for checkpoint in checkpoints:

        suffix = checkpoint.split("/")[-1]
        print(f"Loading model from {checkpoint}, with a suffix of {suffix} at rank {rank}")

        output_path = f"{output_dir}/{suffix}/{rank}.pkl"

        if os.path.exists(output_path):
            print(f"Warning: {output_path} already exists. Skipping...")
            continue

        model = InternVLChatModel.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(f"cuda:{rank}")

        # base_model = InternVLChatModel.from_pretrained(
        #     "./pretrained/InternVL2_5-26B-MPO",  # Path to the original base model (non-LoRA)
        #     torch_dtype=torch.bfloat16,
        #     device_map=None)

        # from peft import PeftModel

        # print("loading lora parts")
        # model = PeftModel.from_pretrained(base_model,
        #                                   checkpoint,
        #                                   is_local_files_only=True)

        model.eval()

        # print(f"at rank {rank}, model loaded from {checkpoint}")
        # print(f">>>><mode is {model}")
        model = DDP(model, device_ids=[rank], output_device=rank)
        # print(f"at rank {rank}, model wrapped in DDP")
        # print(f"<<<<<mode is {model}")

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, use_fast=False
        )

        # Partition dataset among GPUs
        with open(test_jsonl, "r") as file:
            all_lines = file.readlines()

        # Each GPU gets its portion of data
        local_lines = all_lines[rank::world_size]
        # print(f"at {rank}, local_lines: {local_lines[:10]}")

        os.makedirs(f"{output_dir}/{suffix}", exist_ok=True)

        print(f"saving world-{rank} to {output_path}")

        generate_output(local_lines, model, tokenizer, output_path, rank)

        # if rank == 0:  # Aggregate results on rank 0
        #     aggregate_results(world_size, description, output_dir=output_dir)

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()