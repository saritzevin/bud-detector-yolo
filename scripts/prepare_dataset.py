from pathlib import Path
import random
import shutil
import yaml
from scripts.config import Config


def polygon_to_bbox(line: str):
    vals = line.strip().split()
    if len(vals) < 9:  # need at least 4 points (polygon)
        return None

    cls = vals[0]
    coords = list(map(float, vals[1:]))

    xs = coords[0::2]
    ys = coords[1::2]

    # clamp to [0,1] and compute bbox corners
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))

    bw = x_max - x_min
    bh = y_max - y_min

    if bw <= 0 or bh <= 0:  # invalid box
        return None

    # convert to YOLO format (xc, yc, w, h)
    xc = x_min + bw / 2
    yc = y_min + bh / 2

    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def find_image(label_path: Path, raw_images: Path, img_exts):
    # match label file to image by name + extension
    for ext in img_exts:
        p = raw_images / f"{label_path.stem}{ext}"
        if p.exists():
            return p
    return None


def main(cfg):
    # raw dataset paths (before split)
    raw_images = Path(cfg["path"], 'images')
    raw_labels = Path(cfg["path"], 'labels')
    out = Path(cfg["output_dir"])

    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    seed = cfg["seed"]

    random.seed(seed)  # reproducible split

    label_files = sorted(raw_labels.glob("*.txt"))
    pairs = []
    img_exts = cfg["img_exts"]

    # build (image, label) pairs
    for label in label_files:
        img = find_image(label, raw_images, img_exts)
        if img is not None:
            pairs.append((img, label))

    if len(pairs) == 0:
        raise RuntimeError("No image-label pairs found")

    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    # split dataset
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split, split_pairs in splits.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)  # copy image

            bbox_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    converted = polygon_to_bbox(line)  # convert seg → bbox
                    if converted:
                        bbox_lines.append(converted)

            # save YOLO labels
            with open(lbl_dir / label_path.name, "w") as f:
                f.write("\n".join(bbox_lines))

    # YOLO dataset config file
    yaml_data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "bud"},
    }

    with open(out / "bud.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print("Done dataset preparation.")


if __name__ == '__main__':
    cfg = Config("data/config.yaml")
    main(cfg)