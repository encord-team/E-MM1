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
    'cap,
  ],
  on='encord_text_id',
  how='left'
))

```

### 1M Human Rated

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
