"""Gera auditoria por paciente e partições agrupadas para o PAD-UFES-20."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "dataset/PAD-UFES-20/zr7vgbcyr2-1/metadata.csv"
DEFAULT_OUT = ROOT / "dataset/PAD-UFES-20/artefatos_protocolo"
REQUIRED_COLUMNS = {"patient_id", "diagnostic",
                    "img_id", "gender", "fitspatrick"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_patient_quality(data: pd.DataFrame) -> pd.DataFrame:
    rows = data.copy()
    rows["_demographics_incomplete"] = rows[[
        "gender", "fitspatrick"]].isna().any(axis=1)
    rows["_duplicate_img_id"] = rows["img_id"].duplicated(keep=False)
    rows["_exact_duplicate"] = rows.duplicated(keep=False)

    records = []
    for patient_id, group in rows.groupby("patient_id", sort=True):
        diagnoses = sorted(group["diagnostic"].dropna().astype(str).unique())
        duplicate_img_ids = sorted(
            group.loc[group["_duplicate_img_id"],
                      "img_id"].dropna().astype(str).unique()
        )
        records.append({
            "patient_id": patient_id,
            "n_registros": len(group),
            "diagnosticos": ";".join(diagnoses),
            "n_diagnosticos": len(diagnoses),
            "n_registros_com_demografia_ausente": int(group["_demographics_incomplete"].sum()),
            "n_gender_ausente": int(group["gender"].isna().sum()),
            "n_fitspatrick_ausente": int(group["fitspatrick"].isna().sum()),
            "preenchimento_demografico_misto": bool(
                group["_demographics_incomplete"].nunique() > 1
            ),
            "demografia_inconsistente_entre_registros": bool(
                (group["gender"].nunique(dropna=True) > 1)
                or (group["fitspatrick"].nunique(dropna=True) > 1)
            ),
            "n_linhas_com_img_id_duplicado": int(group["_duplicate_img_id"].sum()),
            "img_ids_duplicados": ";".join(duplicate_img_ids),
            "n_linhas_identicas_duplicadas": int(group["_exact_duplicate"].sum()),
        })
    return pd.DataFrame(records)


def grouped_partitions(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    development_idx, test_idx = next(
        outer.split(np.zeros(len(data)),
                    data["diagnostic"], groups=data["patient_id"])
    )
    development = data.iloc[development_idx]
    development_ids = set(development["patient_id"].astype(str))
    test_ids = set(data.iloc[test_idx]["patient_id"].astype(str))
    inner = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed + 1)
    fold_rows = []
    fold_checks = []
    validation_appearances = {patient_id: 0 for patient_id in development_ids}
    for fold_number, (train_idx, validation_idx) in enumerate(inner.split(
        np.zeros(len(development)),
        development["diagnostic"],
        groups=development["patient_id"],
    ), start=1):
        train_ids = set(development.iloc[train_idx]["patient_id"].astype(str))
        validation_ids = set(
            development.iloc[validation_idx]["patient_id"].astype(str))
        fold_rows.extend(
            {"fold": fold_number, "patient_id": patient_id, "uso": "treino"}
            for patient_id in sorted(train_ids)
        )
        fold_rows.extend(
            {"fold": fold_number, "patient_id": patient_id, "uso": "validacao"}
            for patient_id in sorted(validation_ids)
        )
        for patient_id in validation_ids:
            validation_appearances[patient_id] += 1
        fold_checks.append({
            "fold": fold_number,
            "n_pacientes_treino": len(train_ids),
            "n_pacientes_validacao": len(validation_ids),
            "sem_intersecao_treino_validacao": not bool(train_ids & validation_ids),
            "cobre_todos_pacientes_desenvolvimento": (train_ids | validation_ids) == development_ids,
        })

    partitions = pd.DataFrame([
        *({"patient_id": patient_id, "particao": "desenvolvimento"}
          for patient_id in sorted(development_ids)),
        *({"patient_id": patient_id, "particao": "teste"}
          for patient_id in sorted(test_ids)),
    ])
    checks = {
        "n_pacientes_desenvolvimento": len(development_ids),
        "n_pacientes_teste": len(test_ids),
        "intersecao_desenvolvimento_teste": sorted(development_ids & test_ids),
        "sem_pacientes_compartilhados_desenvolvimento_teste": not bool(development_ids & test_ids),
        "cada_paciente_de_desenvolvimento_validado_uma_vez": all(
            count == 1 for count in validation_appearances.values()
        ),
        "folds": fold_checks,
        "uso_do_teste_na_validacao_cruzada": False,
    }
    return partitions, pd.DataFrame(fold_rows), checks


def load_reference_partitions(
    data: pd.DataFrame, reference_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    partitions = pd.read_csv(
        reference_dir / "particoes_pacientes.csv", dtype=str)
    cv_patients = pd.read_csv(
        reference_dir / "pacientes_validacao_cruzada.csv",
        dtype={"patient_id": str, "uso": str},
    )
    if not {"patient_id", "particao"}.issubset(partitions.columns):
        raise ValueError(
            "O arquivo de referência precisa conter patient_id e particao.")
    if not {"fold", "patient_id", "uso"}.issubset(cv_patients.columns):
        raise ValueError(
            "A referência de CV precisa conter fold, patient_id e uso.")

    cohort_ids = set(data["patient_id"].astype(str))
    if partitions["patient_id"].duplicated().any():
        raise ValueError(
            "A referência contém patient_id repetido nas partições.")
    if set(partitions["patient_id"]) != cohort_ids:
        raise ValueError(
            "A coorte e a referência não contêm os mesmos pacientes.")

    development_ids = set(
        partitions.loc[partitions["particao"]
                       == "desenvolvimento", "patient_id"]
    )
    test_ids = set(
        partitions.loc[partitions["particao"] == "teste", "patient_id"])
    if development_ids & test_ids or development_ids | test_ids != cohort_ids:
        raise ValueError(
            "A referência não separa desenvolvimento e teste corretamente.")
    if set(cv_patients["patient_id"]) != development_ids:
        raise ValueError(
            "A validação cruzada precisa conter todos e somente os pacientes de desenvolvimento."
        )

    validation_appearances = {patient_id: 0 for patient_id in development_ids}
    fold_checks = []
    for fold_number, fold_rows in cv_patients.groupby("fold", sort=True):
        train_ids = set(
            fold_rows.loc[fold_rows["uso"] == "treino", "patient_id"])
        validation_ids = set(
            fold_rows.loc[fold_rows["uso"] == "validacao", "patient_id"])
        for patient_id in validation_ids:
            validation_appearances[patient_id] += 1
        fold_checks.append({
            "fold": int(fold_number),
            "n_pacientes_treino": len(train_ids),
            "n_pacientes_validacao": len(validation_ids),
            "sem_intersecao_treino_validacao": not bool(train_ids & validation_ids),
            "cobre_todos_pacientes_desenvolvimento": (train_ids | validation_ids) == development_ids,
        })
    checks = {
        "n_pacientes_desenvolvimento": len(development_ids),
        "n_pacientes_teste": len(test_ids),
        "intersecao_desenvolvimento_teste": sorted(development_ids & test_ids),
        "sem_pacientes_compartilhados_desenvolvimento_teste": not bool(development_ids & test_ids),
        "cada_paciente_de_desenvolvimento_validado_uma_vez": (
            len(fold_checks) == 5
            and all(count == 1 for count in validation_appearances.values())
        ),
        "folds": fold_checks,
        "uso_do_teste_na_validacao_cruzada": bool(set(cv_patients["patient_id"]) & test_ids),
    }
    return partitions, cv_patients, checks


def write_partition_artifacts(
    data: pd.DataFrame,
    out_dir: Path,
    seed: int,
    reference_splits: Path | None = None,
) -> dict:
    if reference_splits is None:
        partitions, cv_patients, checks = grouped_partitions(data, seed)
    else:
        partitions, cv_patients, checks = load_reference_partitions(
            data, reference_splits
        )
    if not checks["sem_pacientes_compartilhados_desenvolvimento_teste"]:
        raise RuntimeError(
            "Há pacientes compartilhados entre desenvolvimento e teste.")
    if not checks["cada_paciente_de_desenvolvimento_validado_uma_vez"]:
        raise RuntimeError(
            "A validação cruzada não atribuiu cada paciente exatamente uma vez.")
    if any(
        not fold["sem_intersecao_treino_validacao"]
        or not fold["cobre_todos_pacientes_desenvolvimento"]
        for fold in checks["folds"]
    ):
        raise RuntimeError(
            "Foi encontrada interseção ou cobertura incompleta em um fold.")

    partitions.to_csv(out_dir / "particoes_pacientes.csv", index=False)
    cv_patients.to_csv(
        out_dir / "pacientes_validacao_cruzada.csv", index=False)

    distribution = []
    for partition in ("desenvolvimento", "teste"):
        patient_ids = set(
            partitions.loc[partitions["particao"] == partition, "patient_id"])
        subset = data[data["patient_id"].astype(str).isin(patient_ids)]
        for diagnosis in sorted(data["diagnostic"].dropna().astype(str).unique()):
            class_rows = subset[subset["diagnostic"].astype(str) == diagnosis]
            distribution.append({
                "particao": partition,
                "diagnostico": diagnosis,
                "n_imagens": len(class_rows),
                "percentual_imagens_na_particao": round(
                    len(class_rows) / len(subset) * 100, 2
                ) if len(subset) else 0.0,
                "n_pacientes_com_diagnostico": class_rows["patient_id"].nunique(),
            })
    pd.DataFrame(distribution).to_csv(
        out_dir / "distribuicao_classes_particoes.csv", index=False
    )
    cv_distribution = []
    for (fold, usage), rows in cv_patients.groupby(["fold", "uso"], sort=True):
        patient_ids = set(rows["patient_id"])
        subset = data[data["patient_id"].astype(str).isin(patient_ids)]
        for diagnosis in sorted(data["diagnostic"].dropna().astype(str).unique()):
            class_rows = subset[subset["diagnostic"].astype(str) == diagnosis]
            cv_distribution.append({
                "fold": fold,
                "uso": usage,
                "diagnostico": diagnosis,
                "n_imagens": len(class_rows),
                "percentual_imagens_na_particao": round(
                    len(class_rows) / len(subset) * 100, 2
                ) if len(subset) else 0.0,
                "n_pacientes_com_diagnostico": class_rows["patient_id"].nunique(),
            })
    pd.DataFrame(cv_distribution).to_csv(
        out_dir / "distribuicao_classes_validacao_cruzada.csv", index=False
    )

    return {
        "semente": seed,
        "metodo": (
            "partições agrupadas herdadas da coorte principal"
            if reference_splits is not None
            else "StratifiedGroupKFold: primeiro fold de 5 externos para teste; 5 folds agrupados no desenvolvimento"
        ),
        "diretorio_particoes_referencia": str(reference_splits.resolve().relative_to(ROOT))
        if reference_splits is not None else None,
        "grupos": "patient_id",
        **checks,
        "particoes": {
            partition: {
                "n_pacientes": len(patient_rows),
                "n_imagens": int(data["patient_id"].astype(str).isin(
                    set(patient_rows["patient_id"])
                ).sum()),
            }
            for partition, patient_rows in partitions.groupby("particao")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-splits",
        type=Path,
        help="Diretório com particoes_pacientes.csv e pacientes_validacao_cruzada.csv para herdar a divisão por paciente.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        parser.error(
            f"Colunas obrigatórias ausentes: {', '.join(missing_columns)}")
    if data[["patient_id", "diagnostic", "img_id"]].isna().any().any():
        parser.error(
            "patient_id, diagnostic e img_id não podem conter valores ausentes.")
    if data["patient_id"].nunique() < 5:
        parser.error(
            "São necessários ao menos cinco pacientes para a divisão agrupada.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    quality = build_patient_quality(data)
    quality.to_csv(
        args.out_dir / "qualidade_dados_por_paciente.csv", index=False)
    split_check = write_partition_artifacts(
        data, args.out_dir, args.seed, args.reference_splits
    )
    metadata = {
        "arquivo_metadata": str(args.csv.resolve().relative_to(ROOT)),
        "sha256_metadata": sha256_file(args.csv),
        "n_registros": len(data),
        "n_pacientes": int(data["patient_id"].nunique()),
        "diagnosticos": sorted(data["diagnostic"].astype(str).unique()),
        "n_img_id_duplicados": int(data["img_id"].duplicated(keep=False).sum()),
        "n_linhas_identicas_duplicadas": int(data.duplicated(keep=False).sum()),
    }
    (args.out_dir / "verificacao_particoes.json").write_text(
        json.dumps({"dados": metadata, **split_check},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Artefatos gravados em: {args.out_dir}")
    print(json.dumps(split_check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
