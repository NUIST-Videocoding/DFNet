import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import csv
import glob
import json

import utils
from eval_single_ours import DetectEval
import cv2
from dataset_path_utils import bpp_image_files, coco_image_file_names, compress_paths

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPRESS_ROOT = os.path.join(PROJECT_ROOT, "compress")
DATASET_ROOT = os.environ.get(
    "DFNET_DATASET_ROOT", "/Dataset/DAWN"
)

DATASETS = {
    "df": (
        os.path.join(DATASET_ROOT, "Fog"),
        os.path.join(DATASET_ROOT, "fog_coco.json"),
    ),
    "dr": (
        os.path.join(DATASET_ROOT, "Rain"),
        os.path.join(DATASET_ROOT, "rain_coco.json"),
    ),
    "ds": (
        os.path.join(DATASET_ROOT, "Snow"),
        os.path.join(DATASET_ROOT, "sonw_coco.json"),
    ),
}


def anchor_bpp_ori_auto_all_images(qp, image_dir, qp_bin_dir, bpp_dir, image_names=None):
    """Compute BPP for all supported image files recursively."""
    with open(bpp_dir, "a", encoding="utf-8") as of:
        total_pxls = 0
        total_bits = 0
        img_files = bpp_image_files(image_dir, image_names)
        total_img_count = len(img_files)

        if total_img_count == 0:
            print("Warning: no supported image files found recursively; cannot compute BPP")
            return 0.0

        print(f"Found {total_img_count} images recursively; starting BPP calculation...")

        for idx, img_fname in enumerate(img_files):
            img_file = os.path.basename(img_fname)
            file_prefix = os.path.splitext(img_file)[0]
            print(f"Processing image {idx + 1}/{total_img_count}: {file_prefix}")

            bin_full_fname = os.path.join(qp_bin_dir, f"{file_prefix}.bin")
            print(f"processing {idx}: {bin_full_fname}...")
            if not os.path.exists(bin_full_fname):
                error_msg = f"Binary file does not exist; aborting: {bin_full_fname}"
                print(error_msg)
                raise FileNotFoundError(error_msg)

            total_bits += os.path.getsize(bin_full_fname) * 8
            img = cv2.imread(str(img_fname))
            if img is None:
                raise FileNotFoundError(
                    f"Image file not found or cannot read: {img_fname}"
                )

            h, w = img.shape[:2]
            total_pxls += h * w

        bpp = total_bits / total_pxls if total_pxls else 0.0
        of.write(
            "{}_{}_picture,qp: {},total_bits: {},total_pxls: {},bpp: {}\n".format(
                0, total_img_count, qp, total_bits, total_pxls, bpp
            )
        )
        print(f"Finished {total_img_count} images; final BPP: {bpp:.6f}")
        return bpp


def iter_experiments(id_dataset_qp_dict):
    for name_id, dataset_qp_dict in id_dataset_qp_dict.items():
        for dataset_name, qps in dataset_qp_dict.items():
            for qp in qps:
                yield name_id, dataset_name, qp


if __name__ == "__main__":
    # 按 id -> dataset -> qp 列表配置实验；不同 dataset 可以使用不同 qp。
    id_dataset_qp_dict = {
         "ours": {
           
            "df": [30],
            "dr": [30],
            "ds": [30],
           
        },
        
    }
  

    for id, dataset_name, qp in iter_experiments(id_dataset_qp_dict):
                data_path, anno_path = DATASETS[dataset_name]
        # 动态设置当前循环的数据集路径（覆盖之前的环境变量）
                os.environ["DETECTRON2_DATASETS"] = data_path
                os.environ["DETECTRON2_ANNOS"] = anno_path
                print(f"\n===== 开始测试：ID={id}, 数据集={dataset_name}, QP={qp} =====")
                print(f"  - 数据路径：{data_path}")
                print(f"  - 注释路径：{anno_path}")

                parser = argparse.ArgumentParser()
                # parser.add_argument("-i", "--index", default=45, type=int)
                parser.add_argument("-i", "--index", default=f"{id}_{dataset_name}_{qp}")
              
                # parser.add_argument("-m", "--mode", default="uncompress")

                parser.add_argument("-m", "--mode", default="evaluation")
               


                args = parser.parse_args()
                set_idx = args.index
                paths = compress_paths(COMPRESS_ROOT, set_idx)
                mode = args.mode


                settings_path = os.path.join(
                    PROJECT_ROOT,
                    "settings",
                    f"{set_idx}.json",
                )
                with open(settings_path, "r", encoding="utf-8") as setting_json:
                    print(setting_json)
                    settings = json.load(setting_json)

                if settings["model_name"] == "x101":
                    methods_eval = DetectEval(settings, set_idx)
                    picklist = coco_image_file_names(anno_path)
                    print(f"Loaded {len(picklist)} image paths from COCO JSON")
                    if not picklist:
                        raise RuntimeError(f"COCO JSON contains no images: {anno_path}")
                    methods_eval.prepare_part(picklist, data_name=f"{id}_{dataset_name}_{qp}")

                if mode == "feature_quant":
                    out_path = os.path.join(PROJECT_ROOT, "feature_quant", id) + os.sep
                    os.makedirs(out_path, exist_ok=True)
                    filenames = methods_eval.feature_quant(out_path)

                elif mode == "evaluation":
                  
                    fea_path = paths["feature_dir"]

                    print(fea_path)
  
                    os.makedirs(fea_path, exist_ok=True)

                    filenames = methods_eval.vtm_coding(fea_path)


                    filenames = glob.glob(os.path.join(paths["experiment_dir"], f"{set_idx}_rec", "*.png"))
                 

                    output_path = os.path.join(COMPRESS_ROOT, "predict") + os.sep
                    os.makedirs(output_path, exist_ok=True)
                    if len(filenames)== len(picklist):
                        print("here!!!")
                        bpp = anchor_bpp_ori_auto_all_images(
                            qp,
                            data_path,
                            os.path.join(paths["experiment_dir"], f"{set_idx}_bit"),
                            f"{output_path}{set_idx}_bpp.txt",
                            image_names=picklist,
                        )
                        methods_eval.evaluation(filenames, output_path)

                elif mode == "uncompress":
                    # uncompressed
                    out_path = paths["experiment_dir"] + os.sep
                    os.makedirs(out_path, exist_ok=True)
                    filenames = methods_eval.feature_quant(out_path)

                    fea_path = paths["feature_dir"]
                    os.makedirs(fea_path, exist_ok=True)
                    filenames = methods_eval.feature_coding_novtm(fea_path)

                    filenames = glob.glob(os.path.join(fea_path, "*.png"))
                    # #

                    output_path = os.path.join(COMPRESS_ROOT, "predict") + os.sep
                    os.makedirs(output_path, exist_ok=True)
                    #
                    print("exeing")
                    methods_eval.evaluation(filenames, output_path)
                

