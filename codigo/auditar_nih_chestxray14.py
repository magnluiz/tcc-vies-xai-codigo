"""Auditoria dos metadados do NIH ChestX-ray14 (sem imagens): partições, sexo, idade e achados."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "dataset/NIH-ChestXray14/fonte"
DEFAULT_OUT = ROOT / "dataset/NIH-ChestXray14/auditoria"
METADATA = "Data_Entry_2017_v2020.csv"
AGE_BINS = [0, 17, 40, 60, 200]
AGE_LABELS = ["0_17", "18_40", "41_60", "acima_60"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(source_dir / METADATA)
    test = set((source_dir / "test_list.txt").read_text().split())
    train_val = set((source_dir / "train_val_list.txt").read_text().split())
    data["particao_oficial"] = np.where(
        data["Image Index"].isin(test), "teste",
        np.where(data["Image Index"].isin(train_val), "treino_val", "fora_das_listas"))
    data["faixa_etaria"] = pd.cut(data["Patient Age"], AGE_BINS, labels=AGE_LABELS,
                                  include_lowest=True).astype(str)
    findings = data["Finding Labels"].str.get_dummies("|")
    return data, findings


def findings_by_sex(data: pd.DataFrame, findings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for finding in findings.columns:
        present = findings[finding] == 1
        row = {"achado": finding, "imagens": int(present.sum()),
               "pacientes": int(data.loc[present, "Patient ID"].nunique())}
        for sex in ("M", "F"):
            of_sex = data["Patient Gender"] == sex
            row[f"pacientes_{sex}"] = int(data.loc[present & of_sex, "Patient ID"].nunique())
            row[f"prevalencia_imagens_{sex}_pct"] = round(100 * findings.loc[of_sex, finding].mean(), 2)
        for split in ("treino_val", "teste"):
            in_split = data["particao_oficial"] == split
            row[f"pacientes_{split}"] = int(data.loc[present & in_split, "Patient ID"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("imagens", ascending=False)


def audit(source_dir: Path, out_dir: Path) -> dict:
    data, findings = load(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    findings_by_sex(data, findings).to_csv(out_dir / "achados_por_sexo.csv", index=False)
    pd.crosstab([data["particao_oficial"], data["Patient Gender"]], data["faixa_etaria"]) \
        .to_csv(out_dir / "sexo_idade_por_particao.csv")

    per_patient = data.groupby("Patient ID")
    summary = {
        "arquivo_metadata": str((source_dir / METADATA).relative_to(ROOT)),
        "sha256_metadata": sha256_file(source_dir / METADATA),
        "origem": "espelho no Hugging Face (alkzar90/NIH-Chest-X-ray-dataset) dos arquivos do NIH Clinical Center",
        "n_imagens": len(data),
        "n_pacientes": int(data["Patient ID"].nunique()),
        "n_image_index_duplicado": int(data["Image Index"].duplicated().sum()),
        "imagens_por_particao": data["particao_oficial"].value_counts().to_dict(),
        "pacientes_em_mais_de_uma_particao": int((per_patient["particao_oficial"].nunique() > 1).sum()),
        "pacientes_por_sexo": per_patient["Patient Gender"].first().value_counts().to_dict(),
        "pacientes_com_sexo_inconsistente": int((per_patient["Patient Gender"].nunique() > 1).sum()),
        "idade": {key: float(value) for key, value in data["Patient Age"].describe().items()},
        "imagens_por_visao": data["View Position"].value_counts().to_dict(),
        "imagens_por_paciente": {"mediana": float(per_patient.size().median()),
                                 "maximo": int(per_patient.size().max())},
    }
    (out_dir / "resumo_auditoria.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonte", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.fonte, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
