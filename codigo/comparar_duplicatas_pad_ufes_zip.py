"""Compara os hashes auditados com as imagens dentro do ZIP original do PAD-UFES-20."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset/PAD-UFES-20"
DEFAULT_ARCHIVE = DATASET_DIR / "zr7vgbcyr2-1.zip"
DEFAULT_AUDIT = DATASET_DIR / "artefatos_protocolo/verificacao_imagens.json"
DEFAULT_METADATA = DATASET_DIR / "zr7vgbcyr2-1/metadata.csv"
DEFAULT_OUT = DATASET_DIR / "artefatos_protocolo"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    with path.open("rb") as source:
        return sha256_stream(source)


def compare_archive(archive_path: Path, audit_path: Path, metadata_path: Path,
                    out_dir: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    archive_rows = []
    candidates_by_part = defaultdict(list)
    expected_hashes = {}

    for digest, group in audit["hashes_sha256_duplicados"].items():
        for item in group["itens"]:
            local_path = Path(item["caminho"])
            path_parts = local_path.parts
            image_index = path_parts.index("images")
            part_name = path_parts[image_index + 1]
            member_name = f"{part_name}/{local_path.name}"
            candidate = {
                "hash_grupo_auditado": digest,
                "img_id": item.get("img_id", ""),
                "patient_id": item.get("patient_id", ""),
                "lesion_id": item.get("lesion_id", ""),
                "diagnostic": item.get("diagnostic", ""),
                "particao_atual": item.get("particao_atual", ""),
                "caminho_local": str(local_path),
                "arquivo_zip_aninhado": f"images/{part_name}.zip",
                "membro_zip_aninhado": member_name,
                "sha256_local": sha256_path(ROOT / local_path),
                "sha256_auditado": digest,
            }
            candidates_by_part[part_name].append(candidate)
            expected_hashes[(digest, candidate["img_id"])] = digest

    with ZipFile(archive_path) as source_zip:
        metadata_member = "metadata.csv"
        if metadata_member not in source_zip.namelist():
            raise FileNotFoundError(
                f"O arquivo {metadata_member} não existe no ZIP original.")
        with source_zip.open(metadata_member) as original_metadata:
            source_metadata_hash = sha256_stream(original_metadata)
        local_metadata_hash = sha256_path(metadata_path)

        nested_archives = sorted(
            name for name in source_zip.namelist()
            if name.startswith("images/") and name.lower().endswith(".zip")
        )
        source_image_count = 0
        for nested_name in nested_archives:
            part_name = Path(nested_name).stem
            with source_zip.open(nested_name) as nested_stream:
                with ZipFile(nested_stream) as image_zip:
                    members = set(image_zip.namelist())
                    source_image_count += sum(
                        Path(name).suffix.lower() in IMAGE_SUFFIXES for name in members
                    )
                    for candidate in candidates_by_part.get(part_name, []):
                        member_name = candidate["membro_zip_aninhado"]
                        row = dict(candidate)
                        if member_name not in members:
                            row["status_no_zip_original"] = "membro_ausente"
                            row["sha256_zip_original"] = ""
                        else:
                            with image_zip.open(member_name) as source_image:
                                row["sha256_zip_original"] = sha256_stream(
                                    source_image)
                            row["status_no_zip_original"] = "encontrado"
                        row["bytes_locais_iguais_aos_do_zip"] = (
                            row["sha256_zip_original"] == row["sha256_local"]
                            if row["sha256_zip_original"] else False
                        )
                        row["hash_do_zip_igual_ao_grupo_auditado"] = (
                            row["sha256_zip_original"] == row["sha256_auditado"]
                            if row["sha256_zip_original"] else False
                        )
                        archive_rows.append(row)

    comparison_csv = out_dir / "comparacao_duplicatas_zip_original.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(archive_rows).to_csv(comparison_csv, index=False)

    groups = defaultdict(list)
    for row in archive_rows:
        groups[row["hash_grupo_auditado"]].append(row)
    confirmed_groups = [
        digest for digest, rows in groups.items()
        if len(rows) > 1 and all(
            row["bytes_locais_iguais_aos_do_zip"]
            and row["hash_do_zip_igual_ao_grupo_auditado"]
            for row in rows
        )
    ]
    report = {
        "arquivo_zip_original": str(archive_path.resolve().relative_to(ROOT)),
        "n_arquivos_zip_aninhados": len(nested_archives),
        "n_imagens_no_zip_original": source_image_count,
        "n_imagens_duplicadas_comparadas": len(archive_rows),
        "n_grupos_hash_auditados": len(groups),
        "n_grupos_confirmados_no_zip_original": len(confirmed_groups),
        "grupos_hash_confirmados_no_zip_original": confirmed_groups,
        "sha256_metadata_local": local_metadata_hash,
        "sha256_metadata_no_zip": source_metadata_hash,
        "metadata_local_byte_a_byte_igual_ao_zip": local_metadata_hash == source_metadata_hash,
        "todas_as_imagens_comparadas_iguais_ao_zip": bool(archive_rows) and all(
            row["bytes_locais_iguais_aos_do_zip"] for row in archive_rows
        ),
        "arquivo_resultado": str(comparison_csv.resolve().relative_to(ROOT)),
    }
    report_path = out_dir / "comparacao_zip_original.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = compare_archive(args.archive, args.audit,
                             args.metadata, args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
