import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"
FILE_RE = re.compile(r'href="(p\d+\.psv)"')


def list_patient_files(folder_url: str) -> list[str]:
    response = requests.get(folder_url, timeout=60)
    response.raise_for_status()
    names = FILE_RE.findall(response.text)
    return sorted(set(names))


def download_files(folder: str, n_files: int, out_dir: Path) -> int:
    folder_url = urljoin(BASE_URL, f"{folder}/")
    files = list_patient_files(folder_url)
    selected = files[:n_files] if n_files > 0 else files

    target_dir = out_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for name in tqdm(selected, desc=f"Downloading {folder}"):
        dst = target_dir / name
        if dst.exists():
            continue
        url = urljoin(folder_url, name)
        with requests.get(url, timeout=60, stream=True) as r:
            r.raise_for_status()
            with dst.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        downloaded += 1
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PhysioNet 2019 challenge patient files")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--n-set-a", type=int, default=150, help="How many files to download from set A")
    parser.add_argument("--n-set-b", type=int, default=150, help="How many files to download from set B")
    args = parser.parse_args()

    dl_a = download_files("training_setA", args.n_set_a, args.out_dir)
    dl_b = download_files("training_setB", args.n_set_b, args.out_dir)

    print(f"Downloaded new files: setA={dl_a}, setB={dl_b}")
    print(f"Data location: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
