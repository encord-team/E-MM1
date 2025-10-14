[![E-MM1: The Worlds Largest Multimodal Dataset](./supplementary/banner.png)](https://data.encord.com/e-mm1/explorer)

With the E-MM1 dataset, we contribute >100M connections between data from five different modalities; Images, Videos, Audio, Point Clouds, Captions.

## Table of Contents

- [Working with the Encord Datasets](#working-with-the-encord-datasets)
- [Working with the Encord Phase 1 Dataset](#working-with-the-encord-phase-1-dataset)
  - [Phase 1 layout](#phase-1-layout)
  - [How Phase-1 groups were formed](#how-phase-1-groups-were-formed)
  - [Column conventions](#column-conventions)
    - [Phase 1 Example](#phase-1-example)
- [Working with the Encord Phase 2 (1M Human Annotated) Dataset](#working-with-the-encord-phase-2-1m-human-annotated-dataset)
  - [Layout](#layout)
  - [Column schema](#column-schema)
  - [Example: Extracting all point cloud - audio groups](#example-extracting-all-point-cloud---audio-groups)
- [EShot: A Zero-Shot Benchmark for Audio <> Point Cloud](#eshot-a-zero-shot-benchmark-for-audio--point-cloud)
  - [Directory Structure](#directory-structure)
  - [File Descriptions](#file-descriptions)
  - [Evaluation Protocol](#evaluation-protocol)
   - [Example](#example)
- [Contributing](#contributing)
  - [What's Coming?](#whats-coming)
---

## Working with the Encord Datasets

We provide two datasets:

- **Phase 1 (automated)** — very large, built via nearest-neighbour retrieval.
- **Phase 2 (annotated)** — smaller, human-verified annotations.

Both phases share the same basic structure:

- An **`infos/`** folder with one CSV **per modality** (e.g., `image.csv`, `audio.csv`, …).  
  Each file in the dataset is uniquely identified by an **`encord_{modality}_id`** column and includes where it’s saved.
- A **master grouping** that references those IDs to define which items belong together.

## Working with the Encord Phase 1 Dataset

**Whats in Phase 1?** Phase 1 contains the large-scale automated dataset built through nearest-neighbour retrieval. For each of ~6.7M captions, we retrieved the top-16 nearest neighbours across all modalities, resulting in over 100M multimodal connections.

### Phase 1 layout

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


### How Phase-1 groups were formed

We started from ~6.7M captions and retrieved the top-16 nearest neighbours **per modality** for each caption.  
Each `nn_k/data_groups.csv` contains, for every caption, the IDs of the *k-th* nearest neighbour for each modality.

### Column conventions

- `encord_{modality}_id` — unique ID for a specific file in that modality (e.g., `encord_image_id`).
- `save_folder` — relative folder under your chosen root where the asset is stored.
- `file_name` — filename of the asset.
- `encord_text_id` — ID of the caption row in `infos/text.csv`.
- `caption` — the caption text in `infos/text.csv`.


### Phase 1 Example

To download the raw underlying data, please see the [Download page][download].


This example constructs a DataFrame of first nearest-neighbour groups, substituting Encord IDs with file paths (Using the file structure as defined and setup in [Download][download])

```python
import os
from pathlib import Path
import polars as pl

ROOT_DATA_PATH = os.getenv("ROOT_DATA_PATH")

CHOSEN_MODALITIES = ["image", "audio", "video", "points"]

nn1_groups = pl.read_csv("data/encord_phase_1_dataset/nn_1/data_groups.csv")
image_info = pl.read_csv("data/encord_phase_1_dataset/infos/image.csv")
audio_info = pl.read_csv("data/encord_phase_1_dataset/infos/audio.csv")
video_info = pl.read_csv("data/encord_phase_1_dataset/infos/video.csv")
points_info = pl.read_csv("data/encord_phase_1_dataset/infos/points.csv")
text_info = pl.read_csv("data/encord_phase_1_dataset/infos/text.csv")

modality_to_info = {
    "image": image_info,
    "audio": audio_info,
    "video": video_info,
    "points": points_info,
    "text": text_info 
}

ROOT_DATA_PATH = Path("your/root/path")
SEP = str(Path("/"))  

for modality in CHOSEN_MODALITIES:
    info_df = modality_to_info[modality] 
    info_df = info_df.with_columns(
        # Here we use the file_structure as in DOWNLOAD.md
        (
            pl.lit(str(ROOT_DATA_PATH)) + SEP + 
            pl.col('save_folder') + SEP + 
            pl.lit(modality) + SEP + 
            pl.col('file_name')
        ).alias('file_path')
    )
    join_col = f'encord_{modality}_index'

    nn1_groups = nn1_groups.join(info_df.select([
        join_col,
        'file_path'
    ]), on=join_col, how='left')

nn1_groups = nn1_groups.join(text_info.select([
    'encord_text_id',
    'caption'
]), on='encord_text_id', how='left')

```

## Working with the Encord Phase 2 (1M Human Annotated) Dataset

**Whats in Phase 2?**

- **`triplets.csv`** is in **long format**. Each row is a *triplet* that links:
  1) a **caption** (`encord_text_id`),  
  2) a **modality_1 item**, and  
  3) a **modality_2 item**,  
  along with an **annotation** describing how well the caption matches the modality_2 candidate.

- **How triplets were created:**  
  We started from datasets of *(caption, modality_1)* pairs. Annotators were shown the **caption** and a **candidate modality_2 item** and asked to label the pairing as:
  - **Good Match**
  - **Partial Match**
  - **No Match**

- **`annotation_mapping.csv`** maps the `annotation` codes used in `triplets.csv` to human-readable labels (`1` → `Good Match`,`2` → `Partial Match`,`3` → `Bad Match`)


### Layout
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

### Column schema

`triplets.csv` columns:

- `encord_text_id` — ID of the caption (joins to `infos/text.csv`).
- `modality_1` — modality of item 1 (e.g., `image`, `audio`, `video`, `points`).
- `modality_2` — modality of item 2 (e.g., `image`, `audio`, `video`, `points`).
- `annotated_modality` — the modality shown to annotators with the caption (usually equals `modality_2`).
- `encord_{modality_1}_id` — Encord ID for item 1 (e.g., `encord_image_id` when `modality_1 == "image"`).
- `encord_{modality_2}_id` — Encord ID for item 2 (e.g., `encord_audio_id` when `modality_2 == "audio"`).
- `annotation` — categorical code for the label (join to `annotation_mapping.csv` for the readable value).

> The `infos/*.csv` files share the same conventions as Phase 1: each contains
> `encord_{modality}_id`, `save_folder`, and `file_name`. Follow Download Instructions to build file paths as:  
> `ROOT_DATA_PATH / save_folder / {modality} / file_name`.


### Example: Extracting all Point-Cloud <> Audio groups from Phase 2

```python
# Prerequisites: 
# - Set ROOT_DATA_PATH environment variable

import polars as pl
import os 
from itertools import permutations
from pathlib import Path

ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')

MODALITIES = ['points','audio']

triplets_df = pl.read_csv('data/encord_phase_2_dataset/triplets.csv')

modality_to_path = {
    'image': 'data/encord_phase_2_dataset/infos/image.csv',
    'audio': 'data/encord_phase_2_dataset/infos/audio.csv',
    'video': 'data/encord_phase_2_dataset/infos/video.csv',
    'points': 'data/encord_phase_2_dataset/infos/points.csv'
}

modality_to_info = {}

for modality in MODALITIES:
    info_df = pl.read_csv(modality_to_path[modality])
    # Add file path construction
    info_df = info_df.with_columns(
        (
            pl.lit(str(ROOT_DATA_PATH)) + "/" + 
            pl.col('save_folder') + "/" + 
            pl.lit(modality) + "/" + 
            pl.col('file_name')
        ).alias('file_path')
    )
    modality_to_info[modality] = info_df

modality_pairs = list(permutations(MODALITIES, 2))

processed_triplets = []
for mod1, mod2 in modality_pairs:
    pair_condition = (pl.col('modality_1') == mod1) & (pl.col('modality_2') == mod2)
    
    mod_1_mod_2_triplets = triplets_df.filter(pair_condition)

    if mod_1_mod_2_triplets.height == 0:
        continue

    mod_1_info = modality_to_info[mod1].select([
        f'encord_{mod1}_id',
        'file_path'
    ]).rename({"file_path": "modality_1_file_path"})
    
    mod_2_info = modality_to_info[mod2].select([
        f'encord_{mod2}_id',
        'file_path'
    ]).rename({"file_path": "modality_2_file_path"})
    
    mod_1_mod_2_triplets = mod_1_mod_2_triplets.join(mod_1_info, on=f'encord_{mod1}_id', how='left')
    mod_1_mod_2_triplets = mod_1_mod_2_triplets.join(mod_2_info, on=f'encord_{mod2}_id', how='left')

    processed_triplets.append(mod_1_mod_2_triplets)

output_triplets = pl.concat(processed_triplets)

# optional : get captions
text_info = pl.read_csv('data/encord_phase_2_dataset/infos/text.csv')
text_info = text_info.select(['encord_text_id','caption'])
output_triplets = output_triplets.join(text_info, on='encord_text_id', how='left')

```


## `EShot`: A Zero-Shot Benchmark for Audio <> Point Cloud

A benchmark dataset for evaluating zero-shot cross-modal classification between audio and 3D point clouds.

[Eval data download here](gs://ml-team-data-bucket/eshot)

**Whats in EShot?**

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
### File Descriptions

#### `audio/`
Directory containing all audio files. Files are referenced by their `eshot_audio_id` from the CSV files.

#### `point-clouds/`
Directory containing all point cloud files. Files are referenced by their `eshot_points_id` from the CSV files.

#### `eshot_audio_info.csv`
Complete metadata for each audio sample.

eshot_audio_id | youtube_id | start_time | end_time

- `eshot_audio_id`: Unique identifier for the audio sample
- `youtube_id`: Source YouTube video ID
- `start_time`: Start timestamp of the audio clip (seconds)
- `end_time`: End timestamp of the audio clip (seconds)

#### `eshot_points_info.csv`
Complete metadata for each point cloud sample.

**Schema**:

eshot_point_id |  object_id

- `eshot_point_id`: Unique identifier for the point cloud sample
- `object_id`: Source 3D object identifier

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

#### category_to_point_ids.json
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
felix write example code here
```


...

## Contributing

### What's Coming?

- [ ] We will publish pre-computed embeddings used to build the 100M dataset as described in Section 3.1.1 in [the paper][paper].
- [ ] We will publish a model with weights that was trained on the dataset. The model can embed all five modalities into a unified embedding space.

# TODO

% REMOVE TODO section BEFORE PUBLISH

### Felix currently working on

- [-] Add `filename` to infors csv.
- [-] Felix find Start+End time for VidGen
- [-] Split 100M into 16 shards.
- [ ] Put files on LFS when finalized
- [ ] Eval data / code for EShot

### Other stuff that needs to happen

- [ ] Prose
  - [-] Dataset attribution
  - [-] Download instructions. Needs fixing + testing
  - [ ] Main README describe how to use the csv files:
    - lfs usage
    - join tables to get info you need. Make many use-cases
  - [ ] Add contact info (ml@encord.com) or whatever
  - [Mavis] Banner?
  - [ ] Main README polish
- [ ] At the end, clean up, squash, force push.

[paper]: ./supplementary/technical_report.pdf
[download]: ./DOWNLOAD.md
