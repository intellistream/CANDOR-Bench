"""
Dataset Download Utilities

Configuration and functions for downloading datasets.
"""

# Google Drive folder URLs for each dataset
DATASET_URLS = {
    "sift-small": "https://drive.google.com/drive/folders/1XbvrSjlP-oUZ5cixVpfSTn0zE-Cim0NK?usp=sharing",
    "sift": "https://drive.google.com/drive/folders/1PngXRH9jnN86T8RNiU-QyGqOillfQE_p?usp=sharing",
    "openimages": "https://drive.google.com/drive/folders/1ZkWOrja-0A6C9yh3ysFoCP6w5u7oWjQx?usp=sharing",
    "sun": "https://drive.google.com/drive/folders/1gNK1n-do-7d5N-Z1tuAoXe5Xq3I8fZIH?usp=sharing",
    "coco": "https://drive.google.com/drive/folders/1Hp6SI8YOFPdWbmC1a4_-1dZWxZH3CHMS?usp=sharing",
    "glove": "https://drive.google.com/drive/folders/1m06VVmXmklHr7QZzdz6w8EtYmuRGIl9s?usp=sharing",
    "msong": "https://drive.google.com/drive/folders/1TnLNJNVqyFrEzKGfQVdvUC8Al-tmjVg0?usp=sharing",
    "coco-0.2": "https://drive.google.com/drive/folders/1ySdDczC6suatp0qKy1GuSa9l7ahZr_o_?usp=drive_link",
    "coco-0.4": "https://drive.google.com/drive/folders/1OSVg-n5wqKknAeN9JF9i2kZ9x5kqdrHW?usp=drive_link",
    "coco-0.6": "https://drive.google.com/drive/folders/1xcPWEbu1oKjth8sjJ9CkR551b06sdEtX?usp=drive_link",
    "coco-0.8": "https://drive.google.com/drive/folders/1coMMN66hVupvRXx05_-zwchiKX9BNfTY?usp=drive_link",
    "coco-0.05": "https://drive.google.com/drive/folders/1oh4JD5OhdPRZAVFGO5TXGlkxG2mO9AYd?usp=drive_link",
    "wte-0.05": "https://drive.google.com/drive/folders/1h2HaNM-DvflauZAUpa0CMwRBEHMvgjP6?usp=drive_link",
    "wte-0.2": "https://drive.google.com/drive/folders/1NKKyoauG9yQYhfrmXpkVy_VgoIY8wERR?usp=drive_link",
    "wte-0.4": "https://drive.google.com/drive/folders/1T7uOPc6cvtv1fhvXLk-uoMAT3shzxcUL?usp=drive_link",
    "wte-0.6": "https://drive.google.com/drive/folders/1xPYSxQ0qbma1hUI2jiKsk4mXxmxDVU6S?usp=drive_link",
    "wte-0.8": "https://drive.google.com/drive/folders/1A-ZtPOghMnPOuxkiTuprFiRwqJ8M6CQB?usp=drive_link",
}


def download_dataset(dataset_name: str, basedir: str) -> bool:
    """
    Download dataset from Google Drive.

    Args:
        dataset_name: Dataset name (key in DATASET_URLS)
        basedir: Target directory

    Returns:
        True if download succeeded
    """
    if dataset_name not in DATASET_URLS:
        print(f"✗ No download URL configured for '{dataset_name}'")
        print(f"  Please download manually and place in: {basedir}")
        return False

    try:
        import gdown

        print(f"Downloading {dataset_name} to {basedir}...")
        folder_url = DATASET_URLS[dataset_name]
        gdown.download_folder(folder_url, output=basedir, quiet=False)
        print(f"✓ Download completed!")
        return True
    except ImportError:
        print("✗ gdown not installed. Install it with: pip install gdown")
        return False
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print(f"  Manual download URL: {DATASET_URLS[dataset_name]}")
        return False
