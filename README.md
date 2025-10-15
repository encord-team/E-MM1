[![E-MM1: The Worlds Largest Multimodal Dataset](./supplementary/banner.png)](https://data.encord.com/e-mm1/explorer)

With the E-MM1 dataset, we contribute >100M groups of data (`E-MM1:100M`) from five different modalities; Images, Videos, Audio, Point Clouds, Captions; That's around 1B connections. We further share 1M human ratings of connections (`E-MM1:1M`) and an evaluation dataset, `EShot`.

## Table of Contents

- [Working with `E-MM1`](#working-with-e-mm1)
- [Working with the `E-MM1:100M` split](#working-with-the-e-mm1100m-split)
  - [`E-MM1:100M` file layout](#e-mm1100m-file-layout)
  - [How `E-MM1:100M` groups were formed](#how-e-mm1100m-groups-were-formed)
  - [Column conventions](#e-mm1100m-column-schema)
  - [`E-MM1:100M` Example](#e-mm1100m-example)
- [Working with the `E-MM1:1M` split](#working-with-the-e-mm11m-split)
  - [`E-MM1:1M` file Layout](#e-mm11m-file-layout)
  - [`E-MM1:1M` column schema](#e-mm11m-column-schema)
  - [Example: Extracting all Point-Cloud ↔ Audio groups from `E-MM1:1M`](#example-extracting-all-point-cloud--audio-groups-from-e-mm11m)
- [`EShot`: A Zero-Shot Benchmark for Audio ↔ Point Cloud](#eshot-a-zero-shot-benchmark-for-audio--point-cloud)
  - [Directory Structure](#directory-structure)
  - [File Descriptions](#file-descriptions)
  - [Evaluation Protocol](#evaluation-protocol)
  - [Example](#example)
- [Contributing](#contributing)
  - [What's Coming?](#whats-coming)

---

## Working with `E-MM1`

We provide two dataset splits:

- **`E-MM1:100M` (automated)** — very large, built via nearest-neighbour retrieval.
- **`E-MM1:1M` (annotated)** — high quality, human-verified annotations.

Both splits share the same basic structure:

- An **`infos/`** folder with one CSV **per modality** (e.g., `image.csv`, `audio.csv`, …).  
  Each file in the dataset is uniquely identified by an **`encord_{modality}_id`** column and includes a path to where the data is stored if you follow the [Download instructions][download].
- A **master grouping** that references those IDs to define which items belong together.

## Working with the `E-MM1:100M` Split

**What is in `E-MM1:100M`?** This split contains the large-scale dataset built with nearest-neighbour retrieval.
For each of ~6.7M captions, we retrieved the top-16 nearest neighbours across all modalities, resulting in roughly 1B multimodal connections or 100M groups.

### `E-MM1:100M` file layout

```
encord_phase_1_dataset/
├─ infos/
│ ├─ video.csv
│ ├─ audio.csv
│ ├─ image.csv
│ ├─ points.csv
│ └─ text.csv
├─ nn_1/
│ └─ data_groups.csv
├─ nn_2/
│ └─ data_groups.csv
...
└─ nn_16/
└─ data_groups.csv
```

> 💡 Note: the CSV files are rather big. Therefore we use `git lfs` to store the large files. Therefor, you have to add a few additional commands when you clone.
> As an example, if you just want the nearest neighbour for each caption, run the following commands:
>
> ```
> git lfs pull --include="e-mm1_100m/infos/*.csv"
> git lfs pull --include="e-mm1_100m/nn_1/*.csv"
> ```

### How `E-MM1:100M` groups were formed

We started from ~6.7M captions and retrieved the top-16 nearest neighbours **per modality** for each caption.  
Each `nn_{k}/data_groups.csv` contains, for every caption, the IDs of the _k-th_ nearest neighbour for each modality.

### `E-MM1:100M` column schema

| **Column**             | **Type** | **Description**                                                          |
| :--------------------- | :------- | :----------------------------------------------------------------------- |
| `encord_{modality}_id` | Integer  | Unique ID for a specific file in that modality (e.g., `encord_image_id`) |
| `save_folder`          | String   | Relative folder under your chosen root where the asset is stored.        |
| `file_name`            | String   | Filename of the asset                                                    |
| `encord_text_id`       | Integer  | ID of the caption row in `infos/text.csv`                                |
| `caption`              | String   | The caption text in `infos/text.csv`                                     |

### `E-MM1:100M` Example

This example constructs a DataFrame of first nearest-neighbour groups, substituting Encord IDs with file paths (Using the file structure as defined and setup in [Download][download])

First, download the raw underlying data by following the instructions on the [Download page][download].
Second, use `git lfs` to fetch the csv files needed.

```
git lfs pull --include="e-mm1_100m/infos/*.csv"
git lfs pull --include="e-mm1_100m/nn_1/*.csv"
```

Then, follow the example below to obtain the relevant data.

```python
import os
from pathlib import Path
import polars as pl

ROOT_DATA_PATH = os.getenv("ROOT_DATA_PATH")

CHOSEN_MODALIIES = ["image", "audio", "video", "points"]

SEP = str(Path("/"))

nn1_groups = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_1_dataset" / "nn_1" / "data_groups.csv"
)
image_info = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_1_dataset" / "infos" / "image.csv"
)
audio_info = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_1_dataset" / "infos" / "audio.csv"
)
video_info = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_1_dataset" / "infos" / "video.csv"
)
points_info = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_1_dataset" / "infos" / "points.csv"
)
text_info = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_1_dataset" / "infos" / "text.csv"
)

modality_to_info = {
    "image": image_info,
    "audio": audio_info,
    "video": video_info,
    "points": points_info,
    "text": text_info,
}

for modality in CHOSEN_MODALIIES:
    info_df = modality_to_info[modality]
    info_df = info_df.with_columns(
        (
            pl.lit(str(ROOT_DATA_PATH))
            + SEP
            + pl.col("save_folder")
            + SEP
            + pl.lit(modality)
            + SEP
            + pl.col("file_name")
        ).alias(f"{modality}_file_path")
    )
    join_col = f"encord_{modality}_id"

    nn1_groups = nn1_groups.join(
        info_df.select([join_col, f"{modality}_file_path"]), on=join_col, how="left"
    )
nn1_groups = nn1_groups.join(
    text_info.select(
        [
            "encord_text_id",
            "caption",
        ]
    ),
    on="encord_text_id",
    how="left",
)

```

`TODO FELIX` Please add a print statement and an example table of what's then in this final data frame

## Working with the `E-MM1:1M` split

**What is in `E-MM1:1M`?**

`TODO FELIX` Please write this section in your own words (like for e-mm1:100m) as rn it sounds like LLM

- **`triplets.csv`** is in **long format**. Each row is a _triplet_ that links:

  1. a **caption** (`encord_text_id`),
  2. a **modality_1 item**, and
  3. a **modality_2 item**,  
     along with an **annotation** describing how well the caption matches the modality_2 candidate.

- **How triplets were created:**  
  We started from datasets of _(caption, modality_1)_ pairs. Annotators were shown the **caption** and a **candidate modality_2 item** and asked to label the pairing as:

  - **Good Match**
  - **Partial Match**
  - **No Match**

- **`annotation_mapping.csv`** maps the `annotation` codes used in `triplets.csv` to human-readable labels (`1` → `Good Match`,`2` → `Partial Match`,`3` → `Bad Match`)

### `E-MM1:1M` file layout

```
encord_phase_2_dataset/
├─ infos/
│ ├─ video.csv
│ ├─ audio.csv
│ ├─ image.csv
│ ├─ points.csv
│ └─ text.csv
├─ triplets.csv
├─ annotation_mapping.csv

```

### `E-MM1:1M` column schema

`triplets.csv` columns:

| **Column**               | **Type** | **Description**                                                                          |
| :----------------------- | :------- | :--------------------------------------------------------------------------------------- |
| `encord_text_id`         | Integer  | ID of the caption (joins to `infos/text.csv`)                                            |
| `modality_1`             | String   | modality of item 1 (e.g., `image`, `audio`, `video`, `points`)                           |
| `modality_2`             | String   | modality of item 2 (e.g., `image`, `audio`, `video`, `points`)                           |
| `annotated_modality`     | String   | the modality shown to annotators with the caption                                        |
| `encord_{modality_1}_id` | Integer  | Encord ID for item 1 (e.g., `encord_image_id` when `modality_1 == "image"`)              |
| `encord_{modality_2}_id` | Integer  | Encord ID for item 2 (e.g., `encord_audio_id` when `modality_2 == "audio"`)              |
| `annotation`             | Integer  | categorical code for the label (join to `annotation_mapping.csv` for the readable value) |

> The `infos/*.csv` files share the same conventions as for `E-MM1:100M`: each contains
> `encord_{modality}_id`, `save_folder`, and `file_name`. Follow [the download instructions][download] to build file paths as:  
> `ROOT_DATA_PATH / save_folder / {modality} / file_name`.

### Example: Extracting all Point-Cloud ↔ Audio groups from `E-MM1:1M`

`TODO FELIX` Please add a description here of what the example does.

> 💡 Change the `MODALITIES` variable to specify which modality pairs you want to extract, e.g., `['points','audio','video']` will extract all points-audio and points-video pairs that exist in the dataset. Note that only modality pairs present in the dataset will be extracted. For example, as there are no audio-video pairs in the dataset, that combination will be skipped.\*\*

Before running the example, make sure you have pulled the csv files from `lsf`:

```shell
git lfs pull --include="datasets/e-mm1_1m/**/*.csv"
```

Now, follow the example:

```python
import polars as pl
import os
from pathlib import Path
from itertools import permutations

ROOT_DATA_PATH = os.getenv("ROOT_DATA_PATH")
MODALITIES = ["points", "audio"]
SEP = str(Path("/"))

triplets_df = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_2_dataset" / "triplets.csv"
)

modality_to_path = {
    "image": Path(ROOT_DATA_PATH) / "encord_phase_2_dataset" / "infos" / "image.csv",
    "audio": Path(ROOT_DATA_PATH) / "encord_phase_2_dataset" / "infos" / "audio.csv",
    "video": Path(ROOT_DATA_PATH) / "encord_phase_2_dataset" / "infos" / "video.csv",
    "points": Path(ROOT_DATA_PATH) / "encord_phase_2_dataset" / "infos" / "points.csv",
}

modality_to_info = {}

for modality in MODALITIES:
    modality_to_info[modality] = pl.read_csv(modality_to_path[modality]).with_columns(
        (
            pl.lit(str(ROOT_DATA_PATH))
            + SEP
            + pl.col("save_folder")
            + SEP
            + pl.lit(modality)
            + SEP
            + pl.col("file_name")
        ).alias(f"{modality}_file_path")
    )

modality_pairs = list(permutations(MODALITIES, 2))

processed_triplets = []
for mod1, mod2 in modality_pairs:
    pair_condition = (pl.col("modality_1") == mod1) & (pl.col("modality_2") == mod2)

    mod_1_mod_2_triplets = triplets_df.filter(pair_condition)

    if mod_1_mod_2_triplets.height == 0:
        continue

    mod_1_info = (
        modality_to_info[mod1]
        .select([f"encord_{mod1}_id", f"{mod1}_file_path"])
        .rename({f"{mod1}_file_path": "modality_1_file_path"})
    )
    mod_2_info = (
        modality_to_info[mod2]
        .select([f"encord_{mod2}_id", f"{mod2}_file_path"])
        .rename({f"{mod2}_file_path": "modality_2_file_path"})
    )
    mod_1_mod_2_triplets = mod_1_mod_2_triplets.join(
        mod_1_info,
        left_on=f"encord_modality_1_id",
        right_on=f"encord_{mod1}_id",
        how="left",
    )
    mod_1_mod_2_triplets = mod_1_mod_2_triplets.join(
        mod_2_info,
        left_on=f"encord_modality_2_id",
        right_on=f"encord_{mod2}_id",
        how="left",
    )

    processed_triplets.append(mod_1_mod_2_triplets)

output_triplets = pl.concat(processed_triplets)

# optional : get captions
text_info = pl.read_csv(
    Path(ROOT_DATA_PATH) / "encord_phase_2_dataset" / "infos" / "text.csv"
)
text_info = text_info.select(["encord_text_id", "caption"])
output_triplets = output_triplets.join(text_info, on="encord_text_id", how="left")
```

`TODO FELIX` Please add a print statement and an example table of what's then in this final data frame

## `EShot`: A Zero-Shot Benchmark for Audio ↔ Point Cloud

A benchmark dataset for evaluating zero-shot cross-modal classification between audio and 3D point clouds.

- **~3,500 samples** across audio and point cloud modalities
- **112 categories** for classification
- **Bidirectional evaluation**: audio→points and points→audio

### Directory Structure

```
eshot/
├─ audio/ (save audio here)
├─ point-clouds/ (save point-clouds here)
├─ eshot_audio_info.csv
├─ eshot_points_info.csv
├─ category_to_point_ids.json
├─ category_to_audio_ids.json

```

### File Descriptions

#### `audio/`

Directory containing all audio files. Files are referenced by their `eshot_audio_id` from the CSV files.

#### `point-clouds/`

Directory containing all point cloud files. Files are referenced by their `eshot_points_id` from the CSV files.

#### `eshot_audio_info.csv`

Complete metadata for each audio sample.

| **Columns**      | **Type** | **Description**                             |
| :--------------- | :------- | :------------------------------------------ |
| `eshot_audio_id` | Integer  | Unique identifier for the audio sample      |
| `youtube_id`     | String   | Source YouTube video ID                     |
| `start_time`     | TODO     | Start timestamp of the audio clip (seconds) |
| `end_time`       | TODO     | End timestamp of the audio clip (seconds)   |

#### `eshot_points_info.csv`

Complete metadata for each point cloud sample.

**Schema**:

| **Columns**      | **Type** | **Description**                              |
| :--------------- | :------- | :------------------------------------------- |
| `eshot_point_id` |          | Unique identifier for the point cloud sample |
| `object_id`      |          | Source 3D object identifier                  |

#### `category_to_audio_ids.json`

Maps categories to audio samples.

**Schema**: `dict[str, list[int]]`

```
{
  "category_name": [eshot_audio_id_1, eshot_audio_id_2, ...],
  ...
}
```

Each of the 112 categories maps to a list of eshot_audio_id values. This determines the class of each audio file

category_to_point_ids.json
Maps categories to point cloud samples.
**Schema**: `dict[str, list[int]]`

```{
  "category_name": [eshot_point_id_1, eshot_point_id_2, ...],
  ...
}
```

Each of the 112 categories maps to a list of eshot_point_id values. This determines the class of each point cloud.

### Evaluation Protocol

**Zero-shot classification** using embedding models:

1. Embed all samples in both modalities using your model
2. For each category, create a **class vector** from the opposing modality:
   - Compute mean of all embeddings in that category
   - Normalize to unit length
3. Classify test samples by nearest class vector

#### Example

```python
import os
import polars as pl
import numpy as np
import json
from pathlib import Path

ROOT_DATA_PATH = os.getenv("ROOT_DATA_PATH")


def make_class_embedding(x: np.ndarray):
    """
    Accepts as input NxD array where N is number of items in class and D is dimension of embeddings
    """
    x = x.mean(axis=0)
    x = x / np.linalg.norm(x)
    return x


audio_info = pl.read_csv(Path(ROOT_DATA_PATH) / "eshot" / "eshot_audio_info.csv")
points_info = pl.read_csv(Path(ROOT_DATA_PATH) / "eshot" / "eshot_points_info.csv")

with open("data/eshot/category_to_point_ids.json", "r") as f:
    category_to_point_ids = json.load(f)

with open("data/eshot/category_to_audio_ids.json", "r") as f:
    category_to_audio_ids = json.load(f)


audio_file_paths = [
    Path(ROOT_DATA_PATH, "eshot", "audio", file_name)
    for file_name in audio_info["file_name"]
]

audio_id_to_index = {
    audio_id: idx for idx, audio_id in enumerate(audio_info["eshot_audio_id"])
}

points_file_paths = [
    Path(ROOT_DATA_PATH, "eshot", "points", file_name)
    for file_name in points_info["file_name"]
]

point_id_to_index = {
    point_id: idx for idx, point_id in enumerate(points_info["eshot_point_id"])
}


audio_embeddings = YOUR_MODEL(
    audio_file_paths
)  # ensure this outputs embedding matrix in same order as file paths input list
points_embeddings = YOUR_MODEL(
    points_file_paths
)  # ensure this outputs embedding matrix in same order as file paths input list

audio_class_vectors = {}
category_to_audio_embeddings = {}
category_to_point_embeddings = {}
points_class_vectors = {}

for category, audio_ids in category_to_audio_ids.items():
    audio_cat_idxs = [audio_id_to_index[audio_id] for audio_id in audio_ids]

    audio_cat_embs = audio_embeddings[audio_cat_idxs]
    category_to_audio_embeddings[category] = audio_cat_embs

    audio_cat_vector = make_class_embedding(audio_cat_embs)

    audio_class_vectors[category] = audio_cat_vector


for category, point_ids in category_to_point_ids.items():
    point_cat_idxs = [point_id_to_index[point_id] for point_id in point_ids]

    point_cat_embs = points_embeddings[point_cat_idxs]
    category_to_point_embeddings[category] = point_cat_embs

    point_cat_vector = make_class_embedding(point_cat_embs)

    points_class_vectors[category] = point_cat_vector


sorted_categories = sorted(category_to_audio_ids.keys())

audio_class_embs = np.stack([audio_class_vectors[cat] for cat in sorted_categories])
point_class_embs = np.stack([points_class_vectors[cat] for cat in sorted_categories])

# audio to points
for i, category in enumerate(sorted_categories):
    print(f"\n{'=' * 60}")
    print(f"Category: {category}")
    print(f"{'=' * 60}")
    audio_embs = category_to_audio_embeddings[category]
    audio_embs = audio_embs / np.linalg.norm(audio_embs, axis=1, keepdims=True)
    sim_mat = audio_embs @ point_class_embs.T

    classifications = np.argsort(sim_mat, axis=1)[:, ::-1]

    print("\n[Audio → Points Classification]")
    for k in [1, 5]:
        top_k_predictions = classifications[:, :k]
        correct = np.any(top_k_predictions == i, axis=1)
        accuracy = np.mean(correct)
        print(
            f"  Top-{k} Accuracy: {accuracy:.4f} ({int(correct.sum())}/{len(correct)} correct)"
        )

    point_embs = category_to_point_embeddings[category]
    point_embs = point_embs / np.linalg.norm(point_embs, axis=1, keepdims=True)
    sim_mat = point_embs @ audio_class_embs.T

    classifications = np.argsort(sim_mat, axis=1)[:, ::-1]

    print("\n[Points → Audio Classification]")
    for k in [1, 5]:
        top_k_predictions = classifications[:, :k]
        correct = np.any(top_k_predictions == i, axis=1)
        accuracy = np.mean(correct)
        print(
            f"  Top-{k} Accuracy: {accuracy:.4f} ({int(correct.sum())}/{len(correct)} correct)"
        )
```

`TODO FELIX` Please add a print statement and an example table of what's then in this final data frame

## What's Coming?

- [ ] We will publish a model with weights that was trained on the dataset. The model can embed all five modalities into a unified embedding space.
- [ ] We will publish pre-computed embeddings used to build the dataset.

# TODO

`FELIX TODO` REMOVE
% REMOVE TODO section BEFORE PUBLISH

### Felix currently working on

- [x] Add `filename` to infors csv.
- [x] Felix find Start+End time for VidGen
- [x] Split 100M into 16 shards.
- [x] Put files on LFS when finalized
- [x] Eval code for EShot

### Other stuff that needs to happen

- [ ] Prose
  - [x] Dataset attribution
  - [x] Download instructions
  - [x] Main README describe how to use the csv files:
    - [x] lfs usage
    - join tables to get info you need. Make many use-cases
  - [ ] Add contact info (ml@encord.com) or whatever
  - [Mavis] Banner?
  - [ ] Main README polish
- [ ] At the end, clean up, squash, force push.

[paper]: ./supplementary/technical_report.pdf
[download]: ./DOWNLOAD.md
