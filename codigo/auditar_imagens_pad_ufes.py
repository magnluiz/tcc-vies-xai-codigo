"""Audita correspondência, decodificação, dimensões e hashes das imagens PAD-UFES-20."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset/PAD-UFES-20/zr7vgbcyr2-1"
DEFAULT_CSV = DATASET_DIR / "metadata.csv"
DEFAULT_IMAGE_ROOT = DATASET_DIR / "images"
DEFAULT_OUT = ROOT / "dataset/PAD-UFES-20/artefatos_protocolo"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(ROOT))
    except ValueError:
        return str(resolved_path)


def inspect_image(path: Path) -> dict:
    record = {
        "status_leitura": "ok",
        "formato": "",
        "largura": None,
        "altura": None,
        "modo": "",
        "sha256": "",
        "erro_leitura": "",
    }
    try:
        with Image.open(path) as image:
            record["formato"] = image.format or ""
            record["largura"], record["altura"] = image.size
            record["modo"] = image.mode
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        record["status_leitura"] = "erro"
        record["erro_leitura"] = f"{type(error).__name__}: {error}"
    try:
        record["sha256"] = sha256_file(path)
    except OSError as error:
        record["sha256"] = ""
        if not record["erro_leitura"]:
            record["status_leitura"] = "erro"
            record["erro_leitura"] = f"{type(error).__name__}: {error}"
    return record


def audit_images(csv_path: Path, image_root: Path, out_dir: Path) -> dict:
    metadata = pd.read_csv(csv_path)
    if "img_id" not in metadata.columns:
        raise ValueError(
            "O CSV de metadados precisa conter a coluna 'img_id'.")

    metadata_ids = metadata["img_id"].dropna().astype(str)
    metadata_counts = metadata_ids.value_counts().to_dict()
    metadata_id_set = set(metadata_counts)
    review_fields = [
        field for field in ("patient_id", "lesion_id", "diagnostic", "biopsed")
        if field in metadata.columns
    ]
    metadata_by_id = {}
    for img_id, group in metadata.dropna(subset=["img_id"]).groupby("img_id", sort=False):
        metadata_by_id[str(img_id)] = {
            field: ";".join(sorted(group[field].dropna().astype(str).unique()))
            for field in review_fields
        }
    partition_file = out_dir / "particoes_pacientes.csv"
    partition_by_patient = {}
    if partition_file.exists():
        partitions = pd.read_csv(partition_file, dtype=str)
        if {"patient_id", "particao"}.issubset(partitions.columns):
            partition_by_patient = partitions.set_index(
                "patient_id")["particao"].to_dict()
    files = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    files_by_name = defaultdict(list)
    for path in files:
        files_by_name[path.name].append(path)

    records = []
    hash_groups = defaultdict(list)
    unreadable_paths = []
    for path in files:
        img_id = path.name if path.name in metadata_id_set else ""
        metadata_record = metadata_by_id.get(img_id, {})
        patient_id = metadata_record.get("patient_id", "")
        matching_paths = files_by_name[path.name]
        if not img_id:
            correspondence = "sem_registro"
        elif len(matching_paths) > 1:
            correspondence = "nome_ambiguo"
        elif metadata_counts[img_id] > 1:
            correspondence = "img_id_duplicado_no_metadata"
        else:
            correspondence = "unica"

        image_record = inspect_image(path)
        if image_record["status_leitura"] != "ok":
            unreadable_paths.append({
                "caminho": display_path(path),
                "img_id": img_id,
                "erro": image_record["erro_leitura"],
            })
        if image_record["sha256"]:
            hash_groups[image_record["sha256"]].append({
                "img_id": img_id,
                "caminho": display_path(path),
                **metadata_record,
                "particao_atual": partition_by_patient.get(patient_id, ""),
            })
        records.append({
            "img_id": img_id,
            "caminho": display_path(path),
            **metadata_record,
            "particao_atual": partition_by_patient.get(patient_id, ""),
            "status_correspondencia": correspondence,
            "n_registros_metadata": metadata_counts.get(img_id, 0),
            "n_arquivos_com_mesmo_nome": len(matching_paths),
            **image_record,
        })

    missing_ids = sorted(metadata_id_set - set(files_by_name))
    for img_id in missing_ids:
        metadata_record = metadata_by_id.get(img_id, {})
        patient_id = metadata_record.get("patient_id", "")
        records.append({
            "img_id": img_id,
            "caminho": "",
            **metadata_record,
            "particao_atual": partition_by_patient.get(patient_id, ""),
            "status_correspondencia": "arquivo_ausente",
            "n_registros_metadata": metadata_counts[img_id],
            "n_arquivos_com_mesmo_nome": 0,
            "status_leitura": "nao_aplicavel",
            "formato": "",
            "largura": None,
            "altura": None,
            "modo": "",
            "sha256": "",
            "erro_leitura": "",
        })

    duplicate_hashes = {
        digest: matches
        for digest, matches in hash_groups.items()
        if len(matches) > 1
    }
    duplicate_hash_review = {}
    cross_partition_hashes = []
    conflicting_label_hashes = []
    for digest, matches in duplicate_hashes.items():
        patient_ids = sorted({match["patient_id"]
                             for match in matches if match.get("patient_id")})
        diagnoses = sorted({match["diagnostic"]
                           for match in matches if match.get("diagnostic")})
        partitions = sorted({match["particao_atual"]
                            for match in matches if match.get("particao_atual")})
        if len(partitions) > 1:
            cross_partition_hashes.append(digest)
        if len(diagnoses) > 1:
            conflicting_label_hashes.append(digest)
        duplicate_hash_review[digest] = {
            "itens": matches,
            "patient_ids": patient_ids,
            "diagnosticos": diagnoses,
            "particoes": partitions,
            "rotulos_conflitantes": len(diagnoses) > 1,
            "atravessa_particoes": len(partitions) > 1,
        }
    for record in records:
        matches = hash_groups.get(
            record["sha256"], []) if record["sha256"] else []
        record["hash_duplicado_exato"] = len(matches) > 1
        record["itens_com_mesmo_hash"] = ";".join(
            match["img_id"] or match["caminho"] for match in matches
        ) if len(matches) > 1 else ""

    extra_files = [
        display_path(path) for path in files
        if path.name not in metadata_id_set
    ]
    duplicate_basenames = {
        name: [display_path(path) for path in paths]
        for name, paths in files_by_name.items()
        if len(paths) > 1
    }
    metadata_null_ids = int(metadata["img_id"].isna().sum())
    duplicate_metadata_ids = {
        img_id: int(count)
        for img_id, count in metadata_counts.items()
        if count > 1
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = out_dir / "auditoria_imagens.csv"
    pd.DataFrame(records).to_csv(audit_csv, index=False)

    needs_partition_review = bool(
        missing_ids or unreadable_paths or duplicate_basenames
        or duplicate_metadata_ids or cross_partition_hashes
    )
    report = {
        "arquivo_metadata": display_path(csv_path),
        "diretorio_imagens": display_path(image_root),
        "n_registros_metadata": len(metadata),
        "n_img_id_nao_nulos": len(metadata_ids),
        "n_img_id_unicos": len(metadata_id_set),
        "n_img_id_ausentes_no_metadata": metadata_null_ids,
        "n_arquivos_de_imagem": len(files),
        "n_ids_sem_arquivo": len(missing_ids),
        "ids_sem_arquivo": missing_ids,
        "n_arquivos_sem_registro": len(extra_files),
        "arquivos_sem_registro": extra_files,
        "n_nomes_de_arquivo_duplicados": len(duplicate_basenames),
        "nomes_de_arquivo_duplicados": duplicate_basenames,
        "n_arquivos_ilegiveis": len(unreadable_paths),
        "arquivos_ilegiveis": unreadable_paths,
        "n_hashes_sha256_duplicados": len(duplicate_hashes),
        "n_grupos_hash_com_rotulos_conflitantes": len(conflicting_label_hashes),
        "hashes_sha256_duplicados": duplicate_hash_review,
        "hashes_sha256_duplicados_que_atravessam_particoes": cross_partition_hashes,
        "revisao_de_registros_duplicados_necessaria": bool(duplicate_hashes),
        "ids_metadata_duplicados": duplicate_metadata_ids,
        "status_correspondencia_completa": not (
            missing_ids or extra_files or duplicate_basenames
            or duplicate_metadata_ids or metadata_null_ids
        ),
        "status_decodificacao_completa": not unreadable_paths,
        "particoes_devem_ser_revisadas_antes_do_treinamento": needs_partition_review,
        "exclusoes_automaticas_realizadas": False,
        "arquivo_resultado": display_path(audit_csv),
    }
    report_path = out_dir / "verificacao_imagens.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = audit_images(args.csv, args.image_root, args.out_dir)
    print(json.dumps({
        "n_arquivos_de_imagem": report["n_arquivos_de_imagem"],
        "n_arquivos_ilegiveis": report["n_arquivos_ilegiveis"],
        "n_ids_sem_arquivo": report["n_ids_sem_arquivo"],
        "n_arquivos_sem_registro": report["n_arquivos_sem_registro"],
        "n_hashes_sha256_duplicados": report["n_hashes_sha256_duplicados"],
        "particoes_devem_ser_revisadas_antes_do_treinamento": report[
            "particoes_devem_ser_revisadas_antes_do_treinamento"
        ],
        "exclusoes_automaticas_realizadas": report["exclusoes_automaticas_realizadas"],
        "arquivo_resultado": report["arquivo_resultado"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
