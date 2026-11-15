"""Verifica os pacotes do NIH ChestX-ray14 e grava o cache de imagens em 256 × 256 (tons de cinza).

As imagens são lidas direto dos ZIPs (a leitura valida o CRC de cada arquivo), sem extração.
O hash de cada pacote é comparado com o SHA-256 publicado pelo espelho no Hugging Face.
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "dataset/NIH-ChestXray14/fonte"
ZIP_DIR = SOURCE_DIR / "zips"
DEFAULT_OUT = ROOT / "dataset/NIH-ChestXray14/cache_imagens_256"
METADATA = SOURCE_DIR / "Data_Entry_2017_v2020.csv"
HF_TREE = "https://huggingface.co/api/datasets/alkzar90/NIH-Chest-X-ray-dataset/tree/main/data/images"
CACHE_SIZE = 256
EXPECTED_SIZE = (1024, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def published_hashes() -> dict:
    with urllib.request.urlopen(HF_TREE, timeout=60) as response:
        entries = json.load(response)
    return {Path(entry["path"]).name: entry["lfs"]["oid"] for entry in entries if entry.get("lfs")}


def process_chunk(args) -> list[tuple]:
    """Lê um lote de PNGs de um ZIP; devolve (nome, modo, largura, altura, array 256 × 256)."""
    zip_path, names = args
    results = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in names:
            with archive.open(name) as member:  # erro de CRC levanta BadZipFile
                with Image.open(member) as image:
                    image.load()
                    mode, size = image.mode, image.size
                    gray = image.convert("L").resize((CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
            results.append((Path(name).name, mode, size[0], size[1], np.asarray(gray, dtype=np.uint8)))
    return results


def build(out_dir: Path, workers: int) -> dict:
    metadata = pd.read_csv(METADATA)
    expected_names = metadata["Image Index"].tolist()
    position = {name: index for index, name in enumerate(expected_names)}

    hashes = published_hashes()
    archives = sorted(ZIP_DIR.glob("images_*.zip"))
    hash_check = {}
    for archive in archives:
        local = sha256_file(archive)
        hash_check[archive.name] = {"sha256": local, "confere_com_publicado": local == hashes.get(archive.name)}
        print(f"{archive.name}: SHA-256 {'confere' if hash_check[archive.name]['confere_com_publicado'] else 'DIFERENTE'}", flush=True)
    if not all(item["confere_com_publicado"] for item in hash_check.values()):
        raise RuntimeError("Algum pacote não confere com o hash publicado.")

    tasks, members_seen, ignored_macos = [], [], 0
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            pngs = [n for n in handle.namelist() if n.lower().endswith(".png")]
        # Metadados do macOS ("__MACOSX/._*.png", 212 bytes) gerados ao compactar; não são imagens.
        names = [n for n in pngs if not n.startswith("__MACOSX/")]
        ignored_macos += len(pngs) - len(names)
        members_seen.extend((archive.name, Path(n).name) for n in names)
        tasks.extend((str(archive), names[i:i + 200]) for i in range(0, len(names), 200))

    found = [name for _, name in members_seen]
    duplicated = sorted({n for n in found if found.count(n) > 1}) if len(found) != len(set(found)) else []
    missing = sorted(set(expected_names) - set(found))
    extra = sorted(set(found) - set(expected_names))
    if missing or extra or duplicated:
        raise RuntimeError(f"Ausentes: {len(missing)}, extras: {len(extra)}, duplicados: {len(duplicated)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_path = out_dir / "imagens_256.npy"
    cache = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8,
                                      shape=(len(expected_names), CACHE_SIZE, CACHE_SIZE))
    modes, sizes, archive_of = {}, {}, dict((name, arch) for arch, name in members_seen)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for results in executor.map(process_chunk, tasks):
            for name, mode, width, height, array in results:
                cache[position[name]] = array
                modes[name] = mode
                sizes[name] = (width, height)
            done += len(results)
            if done % 10000 < 200:
                print(f"{done} imagens processadas", flush=True)
    cache.flush()

    index = pd.DataFrame({
        "posicao": range(len(expected_names)),
        "Image Index": expected_names,
        "pacote": [archive_of[name] for name in expected_names],
        "modo_original": [modes[name] for name in expected_names],
        "largura_png": [sizes[name][0] for name in expected_names],
        "altura_png": [sizes[name][1] for name in expected_names],
    })
    index.to_csv(out_dir / "indice.csv", index=False)
    wrong_size = index[(index["largura_png"] != EXPECTED_SIZE[0]) | (index["altura_png"] != EXPECTED_SIZE[1])]
    manifest = {
        "finalidade": "cache de pré-processamento; não altera os pacotes originais",
        "origem": "espelho no Hugging Face (alkzar90/NIH-Chest-X-ray-dataset) dos arquivos do NIH Clinical Center",
        "pacotes": hash_check,
        "n_imagens_csv": len(expected_names),
        "n_imagens_nos_pacotes": len(found),
        "ausentes": len(missing), "extras": len(extra), "duplicados": len(duplicated),
        "entradas_macosx_ignoradas": ignored_macos,
        "modos_originais": index["modo_original"].value_counts().to_dict(),
        "n_png_fora_de_1024x1024": len(wrong_size),
        "transformacao": "conversão para tons de cinza (L); redimensionamento bilinear para 256 × 256",
        "formato_cache": [len(expected_names), CACHE_SIZE, CACHE_SIZE],
        "arquivo_imagens": str(images_path.relative_to(ROOT)),
        "sha256_imagens": sha256_file(images_path),
        "sha256_metadata": sha256_file(METADATA),
    }
    (out_dir / "manifesto_cache.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    manifest = build(args.out_dir, args.workers)
    manifest.pop("pacotes")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
