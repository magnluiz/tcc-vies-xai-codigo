"""Coorte base do NIH ChestX-ray14 para o desenho derrame pleural × sem achados.

Regras: incidência PA; positivo = qualquer imagem PA com "Effusion" (com ou sem outros achados);
negativo = imagem PA "No Finding" de paciente sem nenhuma imagem PA com derrame; uma imagem por
paciente (o primeiro exame elegível); divisão oficial treino_val/teste; validação de 15% dos
pacientes de treino_val, estratificada por rótulo × sexo. Os conjuntos de treino com viés
controlado são sorteados depois, a partir da partição "treino".
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import auditar_nih_chestxray14 as auditoria


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "dataset/NIH-ChestXray14/derivados_v1"
SEED = 42
VALIDATION_FRACTION = 0.15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_image(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(["Patient ID", "Follow-up #"]).groupby("Patient ID", as_index=False).head(1)


def build_cohort(only_isolated: bool) -> pd.DataFrame:
    data, _ = auditoria.load(auditoria.SOURCE_DIR)
    pa = data[data["View Position"] == "PA"].copy()
    labels = pa["Finding Labels"]
    pa["derrame"] = (labels == "Effusion") if only_isolated else labels.str.contains("Effusion")
    any_effusion = set(pa.loc[labels.str.contains("Effusion"), "Patient ID"])

    positives = first_image(pa[pa["derrame"]])
    negatives = first_image(pa[(labels == "No Finding") & ~pa["Patient ID"].isin(any_effusion)])
    cohort = pd.concat([positives.assign(rotulo=1), negatives.assign(rotulo=0)], ignore_index=True)
    cohort = cohort.rename(columns={"Image Index": "img_id", "Patient ID": "patient_id",
                                    "Patient Gender": "sexo", "Patient Age": "idade",
                                    "Finding Labels": "achados", "Follow-up #": "follow_up"})
    return cohort[["img_id", "patient_id", "sexo", "idade", "faixa_etaria", "rotulo", "achados",
                   "follow_up", "particao_oficial"]]


def assign_partitions(cohort: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cohort = cohort.copy()
    cohort["particao"] = np.where(cohort["particao_oficial"] == "teste", "teste", "treino")
    development = cohort[cohort["particao"] == "treino"]
    for _, group in development.groupby(["rotulo", "sexo"]):
        n_validation = int(round(len(group) * VALIDATION_FRACTION))
        chosen = rng.choice(group.index.to_numpy(), n_validation, replace=False)
        cohort.loc[chosen, "particao"] = "validacao"
    return cohort


def checks(cohort: pd.DataFrame) -> dict:
    patients_per_partition = cohort.groupby("particao")["patient_id"].apply(set)
    partitions = list(patients_per_partition.index)
    intersections = {f"{a}&{b}": len(patients_per_partition[a] & patients_per_partition[b])
                     for i, a in enumerate(partitions) for b in partitions[i + 1:]}
    if any(intersections.values()) or cohort["patient_id"].duplicated().any():
        raise RuntimeError(f"Pacientes repetidos: {intersections}")
    return {"intersecoes_de_pacientes": intersections,
            "uma_imagem_por_paciente": not cohort["patient_id"].duplicated().any()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--derrame-isolado", action="store_true",
                        help="Análise de sensibilidade: positivos só com 'Effusion' sem outros achados.")
    args = parser.parse_args()
    name = "coorte_derrame_isolado" if args.derrame_isolado else "coorte_principal"
    out_dir = args.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = assign_partitions(build_cohort(args.derrame_isolado))
    verification = checks(cohort)
    path = out_dir / "coorte.csv"
    cohort.to_csv(path, index=False)
    counts = cohort.groupby(["particao", "rotulo", "sexo"]).size().unstack("sexo")
    counts.to_csv(out_dir / "contagens_particao_rotulo_sexo.csv")
    manifest = {
        "regras": __doc__.strip(),
        "semente": SEED,
        "metadata_fonte": str(auditoria.SOURCE_DIR.relative_to(ROOT) / auditoria.METADATA),
        "sha256_metadata_fonte": sha256_file(auditoria.SOURCE_DIR / auditoria.METADATA),
        "arquivo_coorte": str(path.relative_to(ROOT)),
        "sha256_coorte": sha256_file(path),
        "n_imagens": len(cohort),
        "contagens": {f"{partition}|rotulo={label}": row.to_dict()
                      for (partition, label), row in counts.iterrows()},
        **verification,
    }
    (out_dir / "manifesto_coorte.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(counts)
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    main()
