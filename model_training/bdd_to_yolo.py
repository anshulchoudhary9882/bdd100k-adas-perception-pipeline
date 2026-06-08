import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


CLASS_MAP = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "bike": 6,
    "motor": 7,
    "traffic light": 8,
    "traffic sign": 9,
}


class BDDToYOLO:

    def __init__(
        self,
        json_file,
        image_dir,
        output_dir
    ):
        self.json_file = Path(json_file)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)

        self.images_out = self.output_dir / "images"
        self.labels_out = self.output_dir / "labels"
        self.metadata_out = self.output_dir / "metadata"

        self.images_out.mkdir(
            parents=True,
            exist_ok=True
        )

        self.labels_out.mkdir(
            parents=True,
            exist_ok=True
        )

        self.metadata_out.mkdir(
            parents=True,
            exist_ok=True
        )

    def convert(
        self,
        split
    ):

        print(f"\nProcessing {split}")

        with open(
            self.json_file,
            "r"
        ) as f:

            data = json.load(f)

        image_dst = self.images_out / split
        label_dst = self.labels_out / split

        image_dst.mkdir(
            exist_ok=True
        )

        label_dst.mkdir(
            exist_ok=True
        )

        metadata_rows = []

        for idx, frame in enumerate(
            tqdm(data)
        ):

            image_name = frame["name"]

            image_path = (
                self.image_dir /
                image_name
            )

            if not image_path.exists():
                continue

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            h, w = image.shape[:2]

            label_lines = []

            for obj in frame.get(
                "labels",
                []
            ):

                category = obj.get(
                    "category"
                )

                if category not in CLASS_MAP:
                    continue

                if "box2d" not in obj:
                    continue

                box = obj["box2d"]

                x1 = box["x1"]
                y1 = box["y1"]
                x2 = box["x2"]
                y2 = box["y2"]

                xc = (
                    (x1 + x2) / 2
                ) / w

                yc = (
                    (y1 + y2) / 2
                ) / h

                bw = (
                    x2 - x1
                ) / w

                bh = (
                    y2 - y1
                ) / h

                cls_id = CLASS_MAP[
                    category
                ]

                label_lines.append(
                    f"{cls_id} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{bw:.6f} "
                    f"{bh:.6f}"
                )

            txt_name = (
                Path(image_name).stem
                + ".txt"
            )

            with open(
                label_dst / txt_name,
                "w"
            ) as f:

                f.write(
                    "\n".join(label_lines)
                )

            shutil.copy2(
                image_path,
                image_dst / image_name
            )

            attrs = frame.get(
                "attributes",
                {}
            )

            metadata_rows.append({
                "image_id": idx,
                "image_name": image_name,
                "weather": attrs.get(
                    "weather",
                    "unknown"
                ),
                "scene": attrs.get(
                    "scene",
                    "unknown"
                ),
                "timeofday": attrs.get(
                    "timeofday",
                    "unknown"
                )
            })

        metadata_df = pd.DataFrame(
            metadata_rows
        )

        metadata_df.to_csv(
            self.metadata_out /
            f"{split}_metadata.csv",
            index=False
        )

        print(
            f"Saved metadata for {split}"
        )


if __name__ == "__main__":

    train_converter = BDDToYOLO(
        json_file=
        r"E:\interview\BoschProject\data\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json",

        image_dir=
        "E:\interview\BoschProject\data\bdd100k_images_100k\bdd100k\images\100k\train",

        output_dir=
        "bdd_yolo"
    )

    train_converter.convert(
        "train"
    )

    val_converter = BDDToYOLO(
        json_file=
        r"E:\interview\BoschProject\data\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json",

        image_dir=
        r"E:\interview\BoschProject\data\bdd100k_images_100k\bdd100k\images\100k\val",

        output_dir=
        "bdd_yolo"
    )

    val_converter.convert(
        "val"
    )

    print(
        "\nConversion Complete"
    )