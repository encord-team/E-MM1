[![E-MM1: The Worlds Largest Multimodal Dataset](./supplementary/banner.png)](https://data.encord.com/e-mm1/explorer)

With the E-MM1 dataset, we contribute >100M connections between data from five different modalities; Images, Videos, Audio, Point Clouds, Captions.

## Getting the Data

The following describes how to use the dataset. We further provide

To download the raw underlying data, please see the [Download page][download].

## Using the data

> How do you link the DFs in LFS to the raw files stored _somewhere_ on the disc?



# Working with the Encord Datasets

We provide two datasets:

- **Phase 1 (automated)** — very large, built via nearest-neighbour retrieval.
- **Phase 2 (annotated)** — smaller, human-verified annotations.

Both phases share the same basic structure:

- An **`infos/`** folder with one CSV **per modality** (e.g., `image.csv`, `audio.csv`, …).  
  Each file in the dataset is uniquely identified by an **`encord_{modality}_id`** column and includes where it’s saved.
- A **master grouping** that references those IDs to define which items belong together.


## Phase 1 layout
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

## Column conventions

- `encord_{modality}_id` — unique ID for a file in that modality (e.g., `encord_image_id`).
- `save_folder` — relative folder under your chosen root where the asset is stored.
- `file_name` — filename of the asset.
- `encord_text_id` — ID of the caption row in `infos/text.csv`.
- *`caption` — the caption text in `infos/text.csv`.


### Phase 1 Example


If you followed the file structure / guide in [Download][download], the you can follow this example to create a dataframe of all 1st nearest neighbour groups with encord_ids replaced by file_paths:

```python


ROOT_DATA_PATH = os.getenv("ROOT_DATA_PATH")

CHOSEN_MODALIIES = ["image", "audio", "video", "points"]


nn1_groups = pl.read_csv("data/encord_phase_1_dataset/nn_1/data_groups.csv")
image_info = pl.read_csv("data/encord_phase_1_dataset/infos/image.csv")
audio_info = pl.read_csv("data/encord_phase_1_dataset/infos/audio.csv")
video_info = pl.read_csv("data/encord_phase_1_dataset/infos/video.csv")
points_info = pl.read_csv("data/encord_phase_1_dataset/infos/points.csv")
text_info = pl.read_csv("data/encord_phase_1_dataset/infos/text.csv")

 modality_to_info = {
  "image" : image_info,
  "audio" : audio_info,
  "video" : video_info,
  "points" : point_info,
  "text" : text_info 
}

from pathlib import Path
import polars as pl

ROOT_DATA_PATH = Path("your/root/path")
SEP = str(Path("/"))  


for modality in CHOSEN_MODALITIES:
    
    info_df = modality_to_info[modality] 
    info_df = info_df.with_columns(
        (
            pl.lit(str(ROOT_DATA_PATH)) + SEP + 
            pl.col('save_folder') + SEP + 
            pl.lit(modality) + SEP + 
            pl.col('file_name')
        ).alias('file_path')
    )
    join_col = f'encord_{modality}_index'

    nn1_groups = nn1_groups.join(info_df.select(
      [
        join_col,
        file_path
      ],
      on = join_col,
      how = 'left
    ))

nn1 = nn1.join(text_info.select(
  [
    'encord_text_id'
    'caprtion,
  ],
  on='encord_text_id',
  how='left'
))

```

# Working with the Encord Phase 2 (1M Human Annotated) Dataset

## Layout
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
## What’s in Phase 2?

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


## Column schema

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


### Example: Extracting all point cloud - audio groups

```python
import polar as pl 
import os 

ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')

MODALITIES = ['points','audio']

triplets_df = pl.read_csv('data/encord_phase_2_dataset/triplets.csv)

modality_to_path = {
                      'image' : 'data/encord_phase_2_dataset/infos/image.csv,
                      'audio' : 'data/encord_phase_2_dataset/infos/audio.csv,
                      'video' : 'data/encord_phase_2_dataset/infos/video.csv,
                      'points' : 'data/encord_phase_2_dataset/infos/points.csv
}

modality_to_info = {}

for modality in MODALITIES:
  modality_to_info[modality] = pl.read_csv(modality_to_path[modality])

modality_pairs = list(permutations(MODALITIES, 2))


perocessed_triplets = []
for mod1, mod2 in modality_pairs:
    pair_condition = (pl.col('modality_1') == mod1) & (pl.col('modality_2') == mod2)
    
    mod_1_mod_2_triplets = triplets.filter(pair_condition)

    if mod_1_mod_2_triplets.height == 0:
      continue

    mod_1_info = modality_to_info[mod1].select([
      f'encord_{mod1}_id',
      'file_path'
      ]
      ).rename({"file_path","modality_1_file_path"})
    mod_2_info = modality_to_info[mod2].select([
      'fencord_{mod2}_id',
      'file_path'
      ]
      ).rename({"file_path","modality_2_file_path"})
    mod_1_mod_2_triplets = mod_1_mod_2_triplets.join(mod_1_info,on=f'encord_{mod1}_id',how='left')
    mod_1_mod_2_triplets = mod_1_mod_2_triplets.join(mod_2_info,on=f'encord_{mod2}_id',how='left')

    processed_triplets.append(mod_1_mod_2_triplets)

output_triplets = pl.concat(processed_triplets)


)

```


## `EShot`: A Zero-Shot Benchmark for Audio <> Point Cloud

[Eval data download here](gs://ml-team-data-bucket/eshot)

...

## Contributing

## What's Coming?

- [ ] We will publish pre-computed embeddings used to build the 100M dataset as described in Section 3.1.1 in [the paper][paper].
- [ ] We will publish a model with weights that was trained on the dataset. The model can embed all five modalities into a unified embedding space.

# TODO

% REMOVE TODO section BEFORE PUBLISH

### Felix currently working on

- [ ] Add `filename` to infors csv.
- [ ] Felix find Start+End time for VidGen
- [ ] Split 100M into 16 shards.
- [ ] Put files on LFS when finalized
- [ ] Eval code for EShot

### Other stuff that needs to happen

- [ ] Prose
  - [ ] Dataset attribution
  - [ ] Download instructions
  - [ ] Main README describe how to use the csv files:
    - lfs usage
    - join tables to get info you need. Make many use-cases
  - [ ] Add contact info (ml@encord.com) or whatever
  - [Mavis] Banner?
  - [ ] Main README polish
- [ ] At the end, clean up, squash, force push.

[paper]: ./supplementary/technical_report.pdf
[download]: ./DOWNLOAD.md
