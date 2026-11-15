"""Cria coortes derivadas do PAD-UFES-20 sem alterar a fonte original."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset/PAD-UFES-20"
DEFAULT_METADATA = DATASET_DIR / "zr7vgbcyr2-1/metadata.csv"
DEFAULT_AUDIT = DATASET_DIR / "artefatos_protocolo/verificacao_imagens.json"
DEFAULT_ZIP_CHECK = DATASET_DIR / "artefatos_protocolo/comparacao_zip_original.json"
DEFAULT_OUT = DATASET_DIR / "derivados_analise_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cohorts(metadata_path: Path, audit_path: Path, zip_check_path: Path,
                    out_dir: Path) -> dict:
    metadata = pd.read_csv(metadata_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    zip_check = json.loads(zip_check_path.read_text(encoding="utf-8"))
    if not zip_check["metadata_local_byte_a_byte_igual_ao_zip"]:
        raise RuntimeError(
            "O metadata local difere do arquivo no ZIP original.")
    if not zip_check["todas_as_imagens_comparadas_iguais_ao_zip"]:
        raise RuntimeError(
            "As imagens locais não foram confirmadas contra o ZIP original.")
    if zip_check["n_grupos_confirmados_no_zip_original"] != audit["n_hashes_sha256_duplicados"]:
        raise RuntimeError(
            "Nem todos os grupos de hash foram confirmados no ZIP original.")
    if metadata["img_id"].isna().any() or metadata["img_id"].duplicated().any():
        raise RuntimeError(
            "img_id precisa estar preenchido e ser único na fonte.")

    image_audit = pd.read_csv(audit_path.parent / "auditoria_imagens.csv")
    hash_by_image = image_audit.set_index("img_id")["sha256"].to_dict()
    conflicting_images = set()
    canonical_by_hash = {}
    same_class_duplicate_images = set()
    for digest, group in audit["hashes_sha256_duplicados"].items():
        image_ids = sorted(item["img_id"] for item in group["itens"])
        if group["rotulos_conflitantes"]:
            conflicting_images.update(image_ids)
        else:
            canonical_by_hash[digest] = image_ids[0]
            same_class_duplicate_images.update(image_ids[1:])

    primary_drop = conflicting_images | same_class_duplicate_images
    sensitivity_drop = conflicting_images
    primary = metadata[~metadata["img_id"].isin(primary_drop)].copy()
    sensitivity = metadata[~metadata["img_id"].isin(sensitivity_drop)].copy()
    decisions = []
    for row in metadata.itertuples(index=False):
        img_id = str(getattr(row, "img_id"))
        digest = hash_by_image.get(img_id, "")
        canonical_id = canonical_by_hash.get(digest, "")
        if img_id in conflicting_images:
            primary_action = "EXCLUIR_ROTULO_CONFLITANTE"
            sensitivity_action = "EXCLUIR_ROTULO_CONFLITANTE"
            reason = "Mesmo SHA-256 associado a diagnósticos diferentes; não há desempate autorizado pela fonte."
        elif img_id in same_class_duplicate_images:
            primary_action = "EXCLUIR_DUPLICATA_EXATA"
            sensitivity_action = "MANTER_NA_SENSIBILIDADE"
            reason = f"Bytes idênticos e mesmo diagnóstico; representante canônico: {canonical_id}."
        else:
            primary_action = "MANTER"
            sensitivity_action = "MANTER"
            reason = "Sem conflito de rótulo ou duplicata exata removida pela política."
        decisions.append({
            "img_id": img_id,
            "patient_id": getattr(row, "patient_id"),
            "lesion_id": getattr(row, "lesion_id"),
            "diagnostic": getattr(row, "diagnostic"),
            "biopsed": getattr(row, "biopsed"),
            "sha256": digest,
            "canonical_img_id": canonical_id,
            "acao_coorte_principal": primary_action,
            "acao_coorte_sensibilidade": sensitivity_action,
            "justificativa": reason,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    primary_dir = out_dir / "coorte_principal"
    sensitivity_dir = out_dir / "coorte_sensibilidade"
    primary_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    primary_path = primary_dir / "metadata_analise.csv"
    sensitivity_path = sensitivity_dir / "metadata_analise.csv"
    primary.to_csv(primary_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    decisions_path = out_dir / "decisoes_coortes.csv"
    pd.DataFrame(decisions).to_csv(decisions_path, index=False)

    source_patients = set(metadata["patient_id"].astype(str))
    primary_patients = set(primary["patient_id"].astype(str))
    sensitivity_patients = set(sensitivity["patient_id"].astype(str))
    manifest = {
        "metadata_fonte": str(metadata_path.resolve().relative_to(ROOT)),
        "sha256_metadata_fonte": sha256_file(metadata_path),
        "metadata_fonte_inalterado": True,
        "zip_original_confirma_metadata_e_imagens": True,
        "politica": {
            "rotulos_conflitantes": "Excluir todos os registros dos 12 grupos da análise supervisionada principal e da sensibilidade; não escolher um diagnóstico sem adjudicação clínica/documental.",
            "duplicatas_mesmo_rotulo": "Na coorte principal, manter um registro por SHA-256, escolhendo o menor img_id em ordem lexicográfica. Na sensibilidade, manter os registros duplicados.",
            "metadata_original": "Preservado sem alterações; coortes são arquivos derivados.",
        },
        "n_grupos_hash_rotulo_conflitante": len(audit["hashes_sha256_duplicados"]) - sum(
            not group["rotulos_conflitantes"]
            for group in audit["hashes_sha256_duplicados"].values()
        ),
        "n_registros_excluidos_conflito": len(conflicting_images),
        "n_grupos_hash_mesmo_rotulo": len(canonical_by_hash),
        "n_registros_excluidos_deduplicacao_principal": len(same_class_duplicate_images),
        "n_registros_fonte": len(metadata),
        "n_pacientes_fonte": len(source_patients),
        "n_pacientes_removidos_integralmente": len(source_patients - primary_patients),
        "pacientes_removidos_integralmente": sorted(source_patients - primary_patients),
        "coorte_principal": {
            "arquivo": str(primary_path.resolve().relative_to(ROOT)),
            "n_registros": len(primary),
            "n_pacientes": len(primary_patients),
            "contagem_por_diagnostico": primary["diagnostic"].value_counts().sort_index().to_dict(),
            "n_hashes_unicos": len({hash_by_image[img_id] for img_id in primary["img_id"]}),
        },
        "coorte_sensibilidade": {
            "arquivo": str(sensitivity_path.resolve().relative_to(ROOT)),
            "n_registros": len(sensitivity),
            "n_pacientes": len(sensitivity_patients),
            "contagem_por_diagnostico": sensitivity["diagnostic"].value_counts().sort_index().to_dict(),
            "n_hashes_unicos": len({hash_by_image[img_id] for img_id in sensitivity["img_id"]}),
        },
        "decisoes_por_registro": str(decisions_path.resolve().relative_to(ROOT)),
    }
    manifest_path = out_dir / "manifesto_coortes.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--zip-check", type=Path, default=DEFAULT_ZIP_CHECK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(
        prepare_cohorts(args.metadata, args.audit,
                        args.zip_check, args.out_dir),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
