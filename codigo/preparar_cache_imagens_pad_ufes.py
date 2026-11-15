"""Pré-processa uma única vez as imagens das coortes do PAD-UFES-20 (RGB, 256 × 256)."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize


ROOT = Path(__file__).resolve().parent.parent
DERIVED_DIR = ROOT / "dataset/PAD-UFES-20/derivados_analise_v1"
DEFAULT_PRIMARY = DERIVED_DIR / "coorte_principal/metadata_analise.csv"
DEFAULT_SENSITIVITY = DERIVED_DIR / "coorte_sensibilidade/metadata_analise.csv"
DEFAULT_IMAGE_ROOT = ROOT / "dataset/PAD-UFES-20/zr7vgbcyr2-1/images"
DEFAULT_OUT = DERIVED_DIR / "cache_imagens_256"
CACHE_SIZE = 256

TRANSFORM = Compose([
    Resize(CACHE_SIZE, interpolation=InterpolationMode.BILINEAR),
    CenterCrop(CACHE_SIZE),
])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source_image:
        image = TRANSFORM(source_image.convert("RGB"))
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (CACHE_SIZE, CACHE_SIZE, 3):
        raise RuntimeError(f"Formato inesperado em {path.name}: {array.shape}")
    return array


def build_cache(primary_path: Path, sensitivity_path: Path, image_root: Path,
                out_dir: Path, workers: int) -> dict:
    primary = pd.read_csv(primary_path)
    sensitivity = pd.read_csv(sensitivity_path)
    primary_ids = set(primary["img_id"].astype(str))
    sensitivity_ids = set(sensitivity["img_id"].astype(str))
    if not primary_ids <= sensitivity_ids:
        raise RuntimeError(
            "A coorte de sensibilidade deveria conter todas as imagens da principal.")

    image_by_id = {
        path.name: path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    img_ids = sorted(sensitivity_ids)
    missing = [img_id for img_id in img_ids if img_id not in image_by_id]
    if missing:
        raise FileNotFoundError(f"Imagens sem arquivo: {missing[:5]}")

    paths = [image_by_id[img_id] for img_id in img_ids]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        arrays = list(executor.map(load_image, paths, chunksize=16))
    images = np.stack(arrays)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_path = out_dir / "imagens_256.npy"
    index_path = out_dir / "indice.csv"
    np.save(images_path, images)
    pd.DataFrame({
        "posicao": range(len(img_ids)),
        "img_id": img_ids,
        "na_coorte_principal": [img_id in primary_ids for img_id in img_ids],
    }).to_csv(index_path, index=False)

    manifest = {
        "finalidade": "cache de pré-processamento; não altera as imagens originais",
        "n_imagens": len(img_ids),
        "n_imagens_coorte_principal": len(primary_ids),
        "formato": list(images.shape),
        "dtype": str(images.dtype),
        "transformacao": "RGB; Resize(256, bilinear) no menor lado; CenterCrop(256)",
        "metadata_principal": str(primary_path.resolve().relative_to(ROOT)),
        "sha256_metadata_principal": sha256_file(primary_path),
        "metadata_sensibilidade": str(sensitivity_path.resolve().relative_to(ROOT)),
        "sha256_metadata_sensibilidade": sha256_file(sensitivity_path),
        "arquivo_imagens": str(images_path.resolve().relative_to(ROOT)),
        "sha256_imagens": sha256_file(images_path),
        "arquivo_indice": str(index_path.resolve().relative_to(ROOT)),
    }
    (out_dir / "manifesto_cache.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(
        build_cache(args.primary, args.sensitivity, args.image_root,
                    args.out_dir, args.workers),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
