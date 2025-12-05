# Datasets Module

Dataset management module inspired by big-ann-benchmarks, providing unified dataset download, preprocessing, and access interfaces.

## Key Features

- **Automatic Download**: Supports automatic dataset download from Google Drive
- **Unified Interface**: Consistent access methods across all datasets
- **Memory Optimization**: Uses iterators for large datasets to avoid memory overflow
- **Standard Format**: Unified directory structure and file formats

## Quick Start

### Install Dependencies
```bash
pip install numpy gdown
```

### Basic Usage
```python
from benchmark_anns.datasets import DATASETS

# Get dataset instance
dataset = DATASETS['sift']()

# Auto-download and prepare (first time)
dataset.prepare()

# Access data
queries = dataset.get_queries()
groundtruth = dataset.get_groundtruth(k=10)

# Use iterator for large datasets
for batch in dataset.get_dataset_iterator(bs=10000):
    # Process batch
    pass
```


## Directory Structure

```
raw_data/
├── sift-small/           # SIFT Small (10K vectors, 128d)
├── sift/                 # SIFT 1M (1M vectors, 128d)
├── openimages/           # OpenImages (9M vectors, 512d)
├── sun/                  # SUN Scene (79K vectors, 512d)
├── msong/                # Million Song (992K vectors, 420d)
├── coco/                 # COCO (117K vectors, 768d)
├── glove/                # GloVe (1.19M vectors, 100d)
└── random-*/             # Random test datasets
```

Each dataset directory contains:
- `base.{format}` - Base vector data
- `query.{format}` - Query vectors
- `groundtruth.{format}` - Ground truth results

## Supported Datasets

### SIFT Series
- **sift-small**: 10K vectors, 128d - Small dataset for testing
- **sift**: 1M vectors, 128d - Standard SIFT dataset

### Image Feature Datasets
- **openimages**: 9M vectors, 512d
- **sun**: 79K vectors, 512d
- **coco**: 117K vectors, 768d

### Text/Embedding Datasets
- **glove**: 1.19M vectors, 100d - GloVe word embeddings
- **msong**: 992K vectors, 420d - Million Song Dataset

### Other Datasets
- **random-xs/s/m**: Randomly generated test datasets

## Dataset Download

### Automatic Download (Recommended)
```python
# Datasets auto-download from Google Drive
dataset = DATASETS['sift']()
dataset.prepare()
```

### Download Links

| Dataset | Size | Download Link |
|---------|------|---------------|
| SIFT-Small | ~50MB | [Google Drive](https://drive.google.com/drive/folders/1XbvrSjlP-oUZ5cixVpfSTn0zE-Cim0NK) |
| SIFT | ~500MB | [Google Drive](https://drive.google.com/drive/folders/1PngXRH9jnN86T8RNiU-QyGqOillfQE_p) |
| OpenImages | ~2GB | [Google Drive](https://drive.google.com/drive/folders/1ZkWOrja-0A6C9yh3ysFoCP6w5u7oWjQx) |
| Sun | ~160MB | [Google Drive](https://drive.google.com/drive/folders/1gNK1n-do-7d5N-Z1tuAoXe5Xq3I8fZIH) |
| COCO | ~300MB | [Google Drive](https://drive.google.com/drive/folders/1Hp6SI8YOFPdWbmC1a4_-1dZWxZH3CHMS) |
| Glove | ~500MB | [Google Drive](https://drive.google.com/drive/folders/1m06VVmXmklHr7QZzdz6w8EtYmuRGIl9s) |
| Msong | ~1.6GB | [Google Drive](https://drive.google.com/drive/folders/1TnLNJNVqyFrEzKGfQVdvUC8Al-tmjVg0) |

**Note**: 
- Downloads may take time; Google Drive may have quota limits
- Ensure stable network connection


## API Reference

### Dataset Registry

All datasets accessed via `DATASETS` dictionary:

```python
from benchmark_anns.datasets import DATASETS

# Get available datasets
print(DATASETS.keys())

# Create dataset instance
dataset = DATASETS['sift']()
```

### Dataset Base Methods

Each dataset provides these methods:

```python
# Prepare dataset (download, create directories)
dataset.prepare()

# Get data (small datasets)
base_vectors = dataset.get_dataset()

# Get data iterator (recommended for large datasets)
for batch in dataset.get_dataset_iterator(bs=10000):
    process(batch)

# Get query vectors
queries = dataset.get_queries()

# Get ground truth
gt = dataset.get_groundtruth(k=10)

# Get dataset attributes
dataset.nb           # Number of base vectors
dataset.nq           # Number of query vectors
dataset.d            # Vector dimension
dataset.distance()   # Distance metric ('euclidean', 'angular', etc.)
dataset.short_name() # Dataset short name
```

## File Formats

### `.fvecs` / `.ivecs` Format
Traditional SIFT format, each vector: `[dim: int32][data: float32/int32 × dim]`

### `.bin` / `.u8bin` / `.fbin` Format (Big-ANN)
- Header: `[n: uint32][d: uint32]`
- Data: `[vectors: dtype × n × d]`

### Ground Truth Format
Binary integer indices containing nearest neighbor indices and distances

## Adding New Datasets

1. Create dataset class in `registry.py`:

```python
class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.nb = 10000      # Number of base vectors
        self.nq = 100        # Number of queries
        self.d = 64          # Vector dimension
        self.basedir = "raw_data/mydataset"
    
    def prepare(self, skip_data=False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            download_dataset('mydataset', self.basedir)
    
    def get_dataset(self):
        return load_fvecs(os.path.join(self.basedir, "base.fvecs"))
    
    def distance(self):
        return "euclidean"
```

2. Register dataset:

```python
DATASETS['mydataset'] = MyDataset
```

3. Add download link in `download_utils.py` (if auto-download needed)

## Notes

- Use iterators instead of direct loading for large datasets
- Random datasets are generated dynamically in memory, no download needed
- Datasets auto-download on first use, ensure stable network connection

## References

- [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks)
- [Google Drive Python API (gdown)](https://github.com/wkentaro/gdown)

