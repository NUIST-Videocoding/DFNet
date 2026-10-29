import json
import os
import shutil

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

from dataset_path_utils import normalize_relative_file_name, resolve_image_path


def simple_filename(filename_ext):
    filename_base = os.path.basename(filename_ext)
    return os.path.splitext(filename_base)[0]


def model_loader(settings):
    cfg = get_cfg()
    cfg.merge_from_file(settings["yaml_path"])
    cfg.MODEL.WEIGHTS = settings["pkl_path"]
    model = DefaultPredictor(cfg).model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, cfg


def pick_coco_exp(name, targetlist):
    """Register a temporary COCO subset with nested image paths."""
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    image_root = os.environ["DETECTRON2_DATASETS"]
    anno_path = os.environ.get("DETECTRON2_ANNOS")
    if not anno_path:
        raise RuntimeError("DETECTRON2_ANNOS is not set")

    with open(anno_path, "r", encoding="utf-8") as anno_file:
        coco_json = json.load(anno_file)

    target_names = {normalize_relative_file_name(x) for x in targetlist}
    target_basenames = {os.path.basename(x) for x in target_names}
    target_stems = {os.path.splitext(os.path.basename(x))[0] for x in target_names}

    selected_images = []
    for image in coco_json.get("images", []):
        file_name = normalize_relative_file_name(image.get("file_name", ""))
        basename = os.path.basename(file_name)
        stem = os.path.splitext(basename)[0]
        if not (
            file_name in target_names
            or basename in target_basenames
            or stem in target_stems
        ):
            continue

        source_path = resolve_image_path(image_root, file_name)
        if source_path is None:
            print(
                f"Warning: image does not exist; skipped: "
                f"root={image_root}, file_name={file_name}"
            )
            continue

        destination_path = os.path.join(dataset_dir, *file_name.split("/"))
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)
        selected_image = dict(image)
        selected_image["file_name"] = file_name
        selected_images.append(selected_image)

    if not selected_images:
        raise RuntimeError(
            f"No target images found. image_root={image_root}, anno_path={anno_path}"
        )

    selected_ids = {image["id"] for image in selected_images}
    selected_annotations = [
        annotation
        for annotation in coco_json.get("annotations", [])
        if annotation.get("image_id") in selected_ids
    ]
    subset_json = {
        "info": coco_json.get("info", {}),
        "licenses": coco_json.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco_json.get("categories", []),
    }
    subset_annotation_path = os.path.join(dataset_dir, "my_anno.json")
    with open(subset_annotation_path, "w", encoding="utf-8") as my_file:
        json.dump(subset_json, my_file, ensure_ascii=False)

    register_coco_instances(name, {}, subset_annotation_path, dataset_dir)
    print(
        f"Registered COCO subset {name}: "
        f"{len(selected_images)} images, {len(selected_annotations)} annotations"
    )


def print_settings(settings, index):
    model_name = settings["model_name"]
    vtm_param = settings["VTM"]
    print()
    print("Evaluation of proposed methods for:", model_name.upper())
    print(f"Settings ID: {index}")
    print(f"VTM parameters      : {vtm_param}")


def save_feature_map_msff_192(filename, features, debug=False):
    feat = [features.squeeze()]
    print("features.shape", features.shape)
    width_list = [16]
    height_list = [12]
    tile_big = np.empty((0, feat[0].shape[2] * width_list[0]))
    for blk, width, height in zip(feat, width_list, height_list):
        big_blk = np.empty((0, blk.shape[2] * width))
        for row in range(height):
            big_blk_col = np.empty((blk.shape[1], 0))
            for col in range(width):
                tile = blk[col + row * width].cpu().numpy()
                if debug:
                    cv2.putText(
                        tile,
                        f"{col + row * width}",
                        (32, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                big_blk_col = np.hstack((big_blk_col, tile))
            big_blk = np.vstack((big_blk, big_blk_col))
        tile_big = np.vstack((tile_big, big_blk))
    tile_big = tile_big.astype(np.uint16)
    cv2.imwrite(filename, tile_big)
