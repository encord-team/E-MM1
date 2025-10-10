# E-MM1: The Worlds Largest Multimodal Dataset

> LAUNCH: Oct. 15th.

With the E-MM1 dataset, we contribute >100M connections between data from five different modalities; Images, Videos, Audio, Point Clouds, Captions.

## Getting the Data

The following describes how to use the dataset. We further provide

To download the raw underlying data, please see the [Download page][download].

## Using the data

> How do you link the DFs in LFS to the raw files stored _somewhere_ on the disc?

### 100M

If you followed the file structure / guide in [Download][download], the you can do:

```python
r1_pairs = pd.read_csv("data/100m/r1.csv")
r1_pairs["file_path"] = os.getenv("ROOT_DATA_PATH") / r1_pairs["file_name"]
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

% REMOVE ME BEFORE PUBLISH

- [ ] Add `filename` to infors csv.
- [ ] Split 100M into 16 shards.
- [ ] Put files on LFS when finalized
- [ ] Felix find Start+End time for VidGen
- [ ] Prose

  - [ ] Dataset attribution
  - [ ] Download instructions
  - [ ] Main README describe how to use the csv files:
    - lfs usage
    - join tables to get info you need. Make many use-cases
  - [ ] Add contact info (ml@encord.com) or whatever
  - [ ] Banner?
  - [ ] Main README polish

- [ ] Eval code for EShot
- [ ] At the end, clean up, squash, force push.

[paper]: ./supplementary/technical_report.pdf
[download]: ./DOWNLOAD.md
