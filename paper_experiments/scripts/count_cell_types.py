import os
import numpy as np
import pandas as pd
from tqdm import tqdm

CONIC_LABELS = {
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective",
}

CONSEP_LABELS = {
    1: "Other",
    2: "Inflammatory",
    3: "Healthy epithelial",
    4: "Malignant epithelial",
    5: "Stroma",
    6: "Muscle",
}

NUCLS_LABELS = {
    1: "Tumor",
    2: "Stroma",
    3: "Lymphocyte",
    4: "Plasma cell",
    5: "Macrophage",
    6: "Other",
}

MONUSAC_LABELS = {
    1: "Epithelial",
    2: "Lymphocyte",
    3: "Macrophage",
    4: "Neutrophil",
}

GLYSAC_LABELS = {
    1: "Other",
    2: "Lymphocyte",
    3: "Epithelial",
    4: "Ambiguous",
}

PUMA_LABELS = {
    1: "Apoptosis",
    2: "Tumor",
    3: "Endothelial",
    4: "Stroma",
    5: "Lymphocyte",
    6: "Histocyte",
    7: "Epithelial",
    8: "Melanophage",
    9: "Other",
}

LABELS = {
    "conic": CONIC_LABELS,
    "consep": CONSEP_LABELS,
    "nucls": NUCLS_LABELS,
    "monusac": MONUSAC_LABELS,
    "glysac": GLYSAC_LABELS,
    "puma": PUMA_LABELS,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count cells in annotations.")
    parser.add_argument(
        "--annotations",
        nargs="+",
        type=str,
        required=True,
        help="Path to annotations directory",
    )
    args = parser.parse_args()

    final_df = {
        "dataset": [],
        "class": [],
        "class_name": [],
        "set": [],
        "count": [],
    }

    for annotation_path in sorted(args.annotations):
        annotations = list(np.load(annotation_path, allow_pickle=True))
        label_count = {}
        for annotation in tqdm(annotations, desc=annotation_path):
            instances = annotation[:, :, 0]
            classifications = annotation[:, :, 1]
            ui = np.unique(instances)
            ui = ui[ui > 0]
            all_instances = []
            for i in range(1, int(classifications.max()) + 1):
                u = np.unique(instances[classifications == i])
                all_instances.extend(u)
                count = len(u)
                if i not in label_count:
                    label_count[i] = 0
                label_count[i] += count
            all_instances = np.unique(all_instances)
            no_annotations = ui[~np.isin(ui, all_instances)]
            N = len(no_annotations)
            if N > 0:
                if 99 not in label_count:
                    label_count[99] = 0
                label_count[99] += N

        if "conic" in annotation_path:
            label_conversion = CONIC_LABELS
        elif "consep" in annotation_path:
            label_conversion = CONSEP_LABELS
        elif "nucls" in annotation_path:
            label_conversion = NUCLS_LABELS
        elif "monusac" in annotation_path:
            label_conversion = MONUSAC_LABELS
        elif "glysac" in annotation_path:
            label_conversion = GLYSAC_LABELS
        elif "puma" in annotation_path:
            label_conversion = PUMA_LABELS
        else:
            label_conversion = {i: f"Class {i}" for i in label_count.keys()}

        for k in sorted(label_count.keys()):
            final_df["dataset"].append(annotation_path.split(os.sep)[-3])
            final_df["class"].append(k)
            final_df["class_name"].append(
                "Unlabeled" if k == 99 else label_conversion[k]
            )
            final_df["set"].append(
                "train" if "train" in annotation_path else "test"
            )
            final_df["count"].append(label_count[k])

    df = pd.DataFrame(final_df)
    # convert to wide format using set as the index
    df = df.pivot(
        index=["dataset", "class", "class_name"], columns="set", values="count"
    )
    df = df.reset_index()
    # rearrange columns
    df = df[["dataset", "class", "class_name", "train", "test"]]
    df["train"] = df["train"].fillna(0).astype(int)
    df["test"] = df["test"].fillna(0).astype(int)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/counts.csv", index=False)
    df.to_latex("data/counts.tex", index=False, multirow=True, escape=False)
