import json
import os


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def normalize_relative_file_name(file_name):
    """Normalize a COCO file name without discarding its directories."""
    return str(file_name).replace("\\", "/").lstrip("./")


def coco_image_file_names(anno_path, limit=None):
    """Read normalized, sorted image file names from a COCO annotation file."""
    with open(anno_path, "r", encoding="utf-8") as annotation_file:
        coco_data = json.load(annotation_file)

    file_names = [
        normalize_relative_file_name(image["file_name"])
        for image in coco_data.get("images", [])
        if image.get("file_name")
    ]
    file_names.sort()
    if limit is not None:
        file_names = file_names[:limit]
    return file_names


def resolve_image_path(image_root, file_name):
    """Resolve nested COCO paths and flat dataset-root file names."""
    relative_name = normalize_relative_file_name(file_name)
    candidates = [
        os.path.join(image_root, *relative_name.split("/")),
        os.path.join(image_root, os.path.basename(relative_name)),
    ]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return None


def iter_image_files(image_dir):
    """Yield all supported image files below ``image_dir`` in sorted order."""
    image_files = []
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(dirpath, filename))
    return sorted(image_files)


def bpp_image_files(image_dir, file_names=None):
    """Return only the images covered by the current COCO selection."""
    if file_names is None:
        return iter_image_files(image_dir)

    image_files = []
    for file_name in file_names:
        image_path = resolve_image_path(image_dir, file_name)
        if image_path is None:
            raise FileNotFoundError(
                f"Image file does not exist: root={image_dir}, file_name={file_name}"
            )
        image_files.append(image_path)
    return image_files


def compress_paths(compress_root, set_idx):
    """Return shared experiment, feature, and input-cache directories."""
    try:
        experiment_id, dataset_name, _ = set_idx.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(
            f"set_idx must end with _<dataset>_<qp>: {set_idx!r}"
        ) from exc

    dataset_prefix = f"{experiment_id}_{dataset_name}"
    experiment_dir = os.path.join(compress_root, experiment_id, dataset_prefix)
    return {
        "experiment_dir": experiment_dir,
        "feature_dir": os.path.join(experiment_dir, f"{set_idx}_ori"),
        "info_dir": os.path.join(experiment_dir, f"{set_idx}_info"),
    }


def coco_metric_name(set_idx):
    """Return the COCO metric branch for an experiment identifier."""
    return "segm" if "seg" in set_idx else "bbox"
