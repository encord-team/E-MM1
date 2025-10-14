# Data Download Guide
This guide provides code to download all raw data files from the various source datasets used in this project. The download scripts utilize the metadata in the infos dataframes 
(located at `/encord_phase_1_dataset/infos/` and `/encord_phase_2_dataset/infos/`) to automatically organize files into the appropriate directory structure.
Repository Structure
All downloaded files are organized into the following structure under your ROOT_DATA_PATH:

```
ROOT_FOLDER/
├── phase_1_only/
│   ├── image/
│   ├── video/
│   └── points/
├── phase_2_only/
│   ├── image/
│   ├── video/
│   └── points/
└── shared/
    ├── image/
    ├── video/
    └── points/
```

phase_1_only: Contains files exclusive to Phase 1
phase_2_only: Contains files exclusive to Phase 2
shared: Contains files that appear in both Phase 1 and Phase 2
The download scripts read the save_folder and file_name columns from each info dataframe to place files in the correct locations automatically. For each code chunk, you need to set the ROOT_DATA_PATH and the path to your chosen infos dataframe that you want to extract data from. 


## Infos DataFrame Schema


Each modality (video, audio, image, points, text) has an associated `infos` dataframe that contains metadata for downloading and organizing the raw data files. All dataframes share a common schema structure with modality-specific columns for file retrieval.

All dataframes have these columns:

`encord_{modality}_id | source_dataset | dataset_license | [modality_specific_columns] | file_name | save_folder`

### Universal Columns (All Modalities)

| Column | Type | Description |
|--------|------|-------------|
| `encord_{modality}_id` | Integer | Unique identifier for the data item within the Encord system (e.g., `encord_video_id`, `encord_image_id`) |
| `source_dataset` | String | Name of the original dataset from which this data item was sourced (e.g., "Valor", "COCO", "AudioSet") |
| `dataset_license` | String | License type of the source dataset (e.g., "MIT", "CC-BY", "Apache-2.0") |
| `file_name` | String | Name of the file to be saved locally (e.g., `4Df6eeR64Ow_14.mp4`) |
| `save_folder` | String | Target directory for organizing files. One of: `phase_1_only`, `phase_2_only`, or `shared` |


### Note on audio: 

All audio files are stored as .mp4 video files (located in the video/ directories) due to significant overlap between video and audio datasets. The scripts will handle this automatically based on the save_folder column in your dataframes.


## Install dependencies

We use [uv](https://docs.astral.sh/uv/) to setup an environment for downloading the data.
```
export ROOT_DATA_PATH=/path/to/ml-data/data
uv sync
```

# Video & Audio

## YouTube-based Datasets


Download videos or audio from YouTube using yt-dlp with time-based segmentation:
Note: We include this as a representative example of how to download videos but for bulk-downloading **All** videos, we would encourage people to look elsewhere for more robust `yt-dlp` based solutions
```python
import os
import subprocess
import polars as pl
import yt_dlp
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
DF_PATH = 'path/to/your/info/df.csv'  # Change this to the phase {1,2} {audio,video} df path e.g: `/infos/audio.csv`

# Load and filter out datasets with no start/end times provided
df = pl.read_csv(DF_PATH) 
df = df.filter(pl.col('source_dataset') != 'VidGen-1M') # VidGen uses the entire video so we will download the video separately in entirety

# Download videos/audio
for row in df.iter_rows(named=True):
    # Create output path
    output_dir = Path(ROOT_DATA_PATH) / row['save_folder'] / 'video' 
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / row['file_name']
    
    # Skip if exists
    if output_path.exists():
        continue
    
    ytid = row["youtube_id"]
    start_time = row["start_time"]
    end_time = row["end_time"]
     
    cmd = [
        'yt-dlp',
        f'https://www.youtube.com/watch?v={ytid}',
        # You may need to pass cookies in
        '--download-sections', f'*{start_time}-{end_time}',
        '-o', str(output_path),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        logger.info(f"Successfully downloaded {row['file_name']}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {row['file_name']}: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out for {row['file_name']}")
    except Exception as e:
        logger.error(f"Unexpected error downloading {row['file_name']}: {e}")
```

## VidGen-1M Videos



For VidGen-1M videos start/end times are not provided. But you can download the dataset directly from HuggingFace and extract the relevant file_id entries:
HuggingFace Dataset: `Fudan-FUXI/VIDGEN-1M`
```python
import os
import polars as pl
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import zipfile
import shutil
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
DF_PATH = 'path/to/your/video_info.csv'

def download_vidgen_videos():
    
    df = pl.read_csv(DF_PATH)

    vidgen_df = df.filter(pl.col('source_dataset') == 'VidGen-1M')
    
    if vidgen_df.height == 0:
        logger.info('dataframe contains no VidGen-1M Videos, ending early.')
        return 

    needed_file_ids = set(vidgen_df['file_id'].to_list())
    logger.info(f"Need to find {len(needed_file_ids)} videos from VidGen-1M")
    
    logger.info("Fetching list of zip files from VidGen-1M repository...")
    all_files = list_repo_files(
        repo_id="Fudan-FUXI/VIDGEN-1M",
        repo_type="dataset"
    )
    zip_files = [f for f in all_files if f.endswith('.zip')]
    logger.info(f"Found {len(zip_files)} zip files in repository")
    
    videos_found = 0
    
    for i, zip_filename in enumerate(zip_files, 1):
        logger.info(f"Processing {i}/{len(zip_files)}: {zip_filename}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            logger.info(f"  Downloading {zip_filename}...")
            zip_path = hf_hub_download(
                repo_id="Fudan-FUXI/VIDGEN-1M",
                filename=zip_filename,
                repo_type="dataset",
                cache_dir=temp_path / "cache"
            )
            
            logger.info(f"  Extracting {zip_filename}...")
            extract_dir = temp_path / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            video_files = list(extract_dir.rglob('*'))
            video_files = [f for f in video_files if f.is_file()]
            
            logger.info(f"  Found {len(video_files)} files in zip")
            
            for video_path in video_files:
                video_id = video_path.stem
                
                if video_id in needed_file_ids:
                    row = vidgen_df.filter(pl.col('file_id') == video_id).row(0, named=True)
                    
                    output_dir = Path(ROOT_DATA_PATH) / row['save_folder'] / 'video'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / row['file_name']
                    
                    if output_path.exists():
                        logger.info(f"  Skipping existing: {row['file_name']}")
                        continue
                    
                    shutil.move(str(video_path), str(output_path))
                    videos_found += 1
                    logger.info(f"Moved: {video_id} -> {output_path}")
                    
                    needed_file_ids.remove(video_id)
            
            logger.info(f"Cleaned up temporary files for {zip_filename}")
        
        logger.info(f"Progress: {videos_found}/{len(vidgen_df)} videos found")
        
        if len(needed_file_ids) == 0:
            logger.info("All videos found! Stopping early.")
            break
    
    logger.info(f"Complete! Found {videos_found}/{len(vidgen_df)} videos")
    if len(needed_file_ids) > 0:
        logger.warning(f"Missing {len(needed_file_ids)} videos: {needed_file_ids}")

if __name__ == "__main__":
    download_vidgen_videos()
```
# Image

For image we need to treat each dataset differently


## COCO


All images are from the COCO2017 Train subset: https://cocodataset.org/#home
```python
import os   
import polars as pl
from pathlib import Path
import requests
import zipfile
import shutil
from tqdm import tqdm

# Download COCO train2017 zip
ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
DF_PATH = 'path/to/your/image_info.csv'
COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"
zip_path = Path(f"{ROOT_DATA_PATH}/train2017.zip")

def download_coco(df):
    coco_df = df.filter(pl.col('source_dataset') == 'COCO')
    if coco_df.height == 0:
        print('no COCO images needed for this dataset')
        return
    else:
        print(f"Found {len(coco_df)} COCO images to process")

    print("Downloading COCO train2017...")
    response = requests.get(COCO_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

  
    extract_folder = Path(f"{ROOT_DATA_PATH}/temp_coco_extracted")
    extract_folder.mkdir(exist_ok=True)


    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(member, extract_folder)

 
    image_source_folder = extract_folder / "train2017"
    print(f"Images extracted to: {image_source_folder}")

   
    needed_file_ids = set(coco_df['file_id'].to_list())

    
    file_id_to_info = {
        row['file_id']: {
            'file_name': row['file_name'],
            'save_folder': row['save_folder']
        }
        for row in coco_df.iter_rows(named=True)
    }


    print("Copying images to target folders...")
    found_count = 0

    for file_id in tqdm(needed_file_ids, desc="Processing COCO images"):
        # COCO images are named with 12-digit zero-padded numbers
        source_filename = f"{int(file_id):012d}.jpg"
        source_path = image_source_folder / source_filename
        
        if source_path.exists():
         
            info = file_id_to_info[file_id]
            file_name = info['file_name']
            save_folder = Path(ROOT_DATA_PATH,info['save_folder'],'image')
            
    
            save_folder.mkdir(parents=True, exist_ok=True)
            target_path = save_folder / file_name
            
         
            try:
                shutil.copy2(source_path, target_path)
                found_count += 1
            except Exception as e:
                print(f"Error copying {file_id}: {e}")
        else:
            print(f"Warning: Source image not found: {source_path}")

    print(f"Copied {found_count}/{len(needed_file_ids)} images")

    zip_path.unlink()
    shutil.rmtree(extract_folder)
```


## ImageNet



All images can be downloaded from huggingface: https://huggingface.co/datasets/ILSVRC/imagenet-1k/
```python
import os
import polars as pl
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import polars as pl
from tqdm import tqdm



ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
DF_PATH = '/path/to/your/image/df.csv

df = pl.read_csv(DF_PATH)
# Create set of needed file IDs from your dataframe
print("Creating set of needed ImageNet file IDs...")

def download_imagenet(df):
    imagenet_df = df.filter(pl.col('source_dataset') == 'ImageNet')
    needed_file_ids = set(imagenet_df['file_id'].to_list())
    print(f"Need to fetch {len(needed_file_ids)} ImageNet images")
    
    if imagenet_df.height == 0:
        return

    
    file_id_to_info = {
        row['file_id']: {
            'file_name': row['file_name'],
            'save_folder': row['save_folder']
        }
        for row in imagenet_df.iter_rows(named=True)
    }

    print("Downloading ImageNet parquet files...")
    repo_id = "ILSVRC/imagenet-1k"
    repo_type = "dataset"

    parquet_files = [f for f in list_repo_files(repo_id, repo_type=repo_type) 
                    if f.startswith('data/train-') and f.endswith('.parquet')]

    print(f"Found {len(parquet_files)}/294 parquet files")

    found_count = 0


    for parquet_file in tqdm(parquet_files[:1], desc="Processing parquet files"):
 
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=parquet_file,
            repo_type=repo_type
        )
        
        parquet_df = pl.read_parquet(local_path)

        for row in tqdm(parquet_df.iter_rows(named=True), desc="Processing images"):  

            im_id = row['image']['path'].replace('.JPEG', '')
            
         
            if im_id in needed_file_ids:
   
                im_bytes = row['image']['bytes']
                
               
                info = file_id_to_info[im_id]
                file_name = info['file_name']
                save_folder = Path(ROOT_DATA_PATH,info['save_folder'],'image')
                

                save_folder.mkdir(parents=True, exist_ok=True)
                target_path = save_folder / file_name
                
                with open(target_path, 'wb') as f:
                    f.write(im_bytes)
                
                found_count += 1
                
  
                if found_count == len(needed_file_ids):
                    print(f"\nAll {found_count} images found. Stopping early.")
                    break

        print(f"Found and saved {found_count}/{len(needed_file_ids)} images")
```

## Flickr30k



All images can be downloaded from huggingface: https://huggingface.co/datasets/nlphuji/flickr30k
```python 
import polars as pl
from huggingface_hub import hf_hub_download
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm



DF_PATH = '/path/to/your/image/df.csv
df = pl.read_csv(DF_PATH)

print("Downloading Flickr30k images...")
zip_path = hf_hub_download(
    repo_id="nlphuji/flickr30k",
    filename="flickr30k-images.zip",
    repo_type="dataset"
)


extract_folder = Path("./temp_flickr30k_extracted")
extract_folder.mkdir(exist_ok=True)


print("Extracting zip file...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)


image_files = list(extract_folder.rglob("*.jpg"))
if image_files:
    image_source_folder = image_files[0].parent
else:
    image_source_folder = extract_folder

print(f"Images extracted to: {image_source_folder}")


flickr_df = df.filter(pl.col('source_dataset') == 'Flickr30k')
print(f"Found {len(flickr_df)} Flickr30k images to process")

print("Copying images...")
for row in tqdm(flickr_df.iter_rows(named=True)):
    file_id = row['file_id']
    file_name = row['file_name']
    save_folder = row['save_folder']
    
    source_path = image_source_folder / f"{file_id}.jpg"
    

    target_folder = Path(ROOT_DATA_PATH,save_folder,'image')
    target_folder.mkdir(parents=True, exist_ok=True)
    
    target_path = target_folder / target_filename
    

    if source_path.exists():
        try:
            shutil.copy2(source_path, target_path)
        except Exception as e:
            print(f"Error copying {file_id}: {e}")
    else:
        print(f"Warning: Source image not found: {source_path}")

shutil.rmtree(extract_folder)
```


### Google Conceptual Captions (GCC) and other URL-based Image Datasets


Download images using img2dataset:
```python
import os
import json
import shutil
import tempfile
import polars as pl
from pathlib import Path
from tqdm import tqdm


PROCESSES_COUNT = 16
THREAD_COUNT = 64
RESIZE_MODE = 'no'
ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
DF_PATH = '/path/to/your/image/df.csv

#STEP 1: DOWNLOAD WITH IMG2DATASET 
print("Loading dataframe...")
df = pl.read_csv(DF_PATH)
df = df.filter(pl.col('source_dataset') == 'Google Conceptual Captions')


with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
    temp_path = temp_file.name
    df.select(['file_id', 'file_name', 'save_folder']).rename({"file_id":"url"}).write_csv(temp_path)


output_dir = Path(ROOT_DATA_PATH) / 'gcc_downloads'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Downloading {len(df)} images to {output_dir}")


os.system(f"""
img2dataset --url_list {temp_path} \\
    --output_folder {output_dir} \\
    --input_format csv \\
    --processes_count {PROCESSES_COUNT} \\
    --thread_count {THREAD_COUNT} \\
    --resize_mode {RESIZE_MODE} \\
    --save_additional_columns '["file_name","save_folder"]' \\
    --enable_wandb False
""")

os.remove(temp_path)

#STEP 2: Move images to their correct folders
print("\Moving images...")
organized_dir = Path(ROOT_DATA_PATH) 


json_files = list(output_dir.glob("*/*.json"))
print(f"Found {len(json_files)} files to reorganize")

moved, errors = 0, 0
for json_file in tqdm(json_files, desc="Reorganizing"):
    try:
        with open(json_file) as f:
            meta = json.load(f)
        
        img_file = json_file.with_suffix('.jpg')
        
        if img_file.exists() and meta.get('file_name') and meta.get('save_folder'):
            target_dir = organized_dir / meta['save_folder'] / 'image'
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, target_dir / f"{meta['file_name']}")
            moved += 1
        else:
            errors += 1
    except Exception as e:
        print(e)
        errors += 1

print(f"\nComplete! Moved: {moved}, Errors: {errors}")
```
<!-- Checked to be okay -->

# Points

all point clouds are available to download from OpenShape on huggingface: https://huggingface.co/datasets/OpenShape/openshape-training-data/tree/main 
Files can be identified using the file_id column in infos/points.csv

```python
import polars as pl
from pathlib import Path
import os
from huggingface_hub import hf_hub_download
import tarfile
import shutil
from tqdm import tqdm

ROOT_DATA_PATH = os.getenv('ROOT_DATA_PATH')
REPO_ID = "OpenShape/openshape-training-data"
DF_PATH = '/path/to/your/image/points.csv

df = pl.read_csv(DF_PATH)
ROOT_DATASETS = {
    '3D-FUTURE': '3D-FUTURE.tar.gz',
    'ABO': 'ABO.tar.gz',
    'ShapeNet': 'ShapeNet.tar.gz'
}

def process_tar(tar_filename, dataset_df):
    if len(dataset_df) == 0:
        return
    
    print(f"\nProcessing {dataset_name} ({len(dataset_df)} files needed)...")
    
 
    file_id_to_folder = {row['file_id']: row['save_folder'] 
                         for row in dataset_df.iter_rows(named=True)}
    
    print(f"Downloading {REPO_ID}/{tar_filename}...")
    tar_path = hf_hub_download(repo_id=REPO_ID, filename=tar_filename, repo_type="dataset")
    print(f"Downloaded {REPO_ID}/{tar_filename}")
    found = 0
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tqdm(tar.getmembers(), desc=dataset_name):
            if member.isfile() and member.name.endswith('.npy'):
                file_id = Path(member.name).stem
                if file_id in file_id_to_folder:
                    target_path = Path(ROOT_DATA_PATH, file_id_to_folder[file_id],'points', f"{file_id}.npy")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with tar.extractfile(member) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    found += 1
    
    print(f"{dataset_name}: Copied {found} files")


for dataset_name, tar_filename in ROOT_DATASETS.items():
    process_tar(tar_filename, df)

# Process Objaverse (000-000.tar.gz to 000-159.tar.gz)

print(f"\nProcessing Objaverse ({len(df)} files needed)...")
file_id_to_folder = {row['file_id']: row['save_folder'] 
                        for row in df.iter_rows(named=True)}
found = 0
N_OBJAVERSE_TARS = 160
for i in tqdm(range(N_OBJAVERSE_TARS), desc="Objaverse tars"):
    try:
        tar_path = hf_hub_download(repo_id=REPO_ID, filename=f"Objaverse/000-{i:03d}.tar.gz", repo_type="dataset")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.npy'):
                    file_id = Path(member.name).stem
                    if file_id in file_id_to_folder:
                        target_path = Path(ROOT_DATA_PATH , file_id_to_folder[file_id] ,'points', f"{file_id}.npy")
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with tar.extractfile(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        found += 1
        
        if found == len(df):
            print(f"All files found! Stopping at tar 000-{i:03d}.")
            break
    except Exception as e:
        print(f"Error with tar 000-{i:03d}: {e}")

print("\nAll datasets processed!")
```
## Captions

All captions are available in the 'caption' column in infos/text.csv
