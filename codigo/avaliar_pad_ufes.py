"""Calcula as métricas do teste do PAD-UFES-20 a partir das predições já gravadas.

Não treina nem prediz: lê `predicoes_teste.csv` da fase final e aplica as regras da ficha
de protocolo (bootstrap por paciente e limiares de suporte por subgrupo).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "dataset/PAD-UFES-20/derivados_analise_v1/experimentos_v1"
CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
N_CLASSES = len(CLASS_NAMES)
N_BOOTSTRAP = 2000
SEED = 42
MIN_PATIENTS_GROUP = 30
MIN_PATIENTS_CELL = 10


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.bincount(labels * N_CLASSES + predictions,
                       minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    hits = np.diag(matrix).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(true_counts > 0, hits / true_counts, np.nan)
        precision = np.where(predicted_counts > 0, hits / predicted_counts, 0.0)
        f1 = np.where(precision + np.nan_to_num(recall) > 0,
                      2 * precision * np.nan_to_num(recall) / (precision + np.nan_to_num(recall)), 0.0)
    present = true_counts > 0
    return {
        "acuracia": hits.sum() / matrix.sum(),
        # Macro-F1 e acurácia balanceada só são definidos quando todas as classes estão presentes.
        "macro_f1": f1.mean() if present.all() else np.nan,
        "acuracia_balanceada": np.nanmean(recall) if present.all() else np.nan,
        "sensibilidade": recall,
        "precisao": precision,
        "f1": f1,
    }


class PatientBootstrap:
    """Reamostra pacientes com reposição, mantendo juntas as imagens de cada um."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)
        self.patients = self.data["patient_id"].unique()
        groups = self.data.groupby("patient_id").indices
        self.rows_by_patient = [groups[patient] for patient in self.patients]
        rng = np.random.default_rng(SEED)
        self.samples = [rng.integers(0, len(self.patients), len(self.patients))
                        for _ in range(N_BOOTSTRAP)]

    def replicate_rows(self):
        for sample in self.samples:
            yield np.concatenate([self.rows_by_patient[index] for index in sample])


def interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    if len(valid) == 0:
        return {"ic95_inf": None, "ic95_sup": None, "replicacoes_validas": 0}
    low, high = np.percentile(valid, [2.5, 97.5])
    return {"ic95_inf": float(low), "ic95_sup": float(high), "replicacoes_validas": int(len(valid))}


def global_metrics(data: pd.DataFrame, bootstrap: PatientBootstrap) -> tuple[dict, pd.DataFrame]:
    labels, predictions = data["label"].to_numpy(), data["pred"].to_numpy()
    point = metrics_from_confusion(confusion(labels, predictions))
    replicates = {key: [] for key in ("acuracia", "macro_f1", "acuracia_balanceada")}
    per_class = {key: [] for key in ("sensibilidade", "precisao", "f1")}
    for rows in bootstrap.replicate_rows():
        result = metrics_from_confusion(confusion(labels[rows], predictions[rows]))
        for key in replicates:
            replicates[key].append(result[key])
        for key in per_class:
            per_class[key].append(result[key])
    summary = {key: {"valor": float(point[key]), **interval(values)}
               for key, values in replicates.items()}
    class_rows = []
    for index, name in enumerate(CLASS_NAMES):
        mask = labels == index
        row = {"diagnostico": name, "n_imagens": int(mask.sum()),
               "n_pacientes": int(data.loc[mask, "patient_id"].nunique())}
        for key in per_class:
            ci = interval([values[index] for values in per_class[key]])
            row[key] = float(point[key][index])
            row[f"{key}_ic95_inf"] = ci["ic95_inf"]
            row[f"{key}_ic95_sup"] = ci["ic95_sup"]
        class_rows.append(row)
    return summary, pd.DataFrame(class_rows)


def add_groups(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["sexo"] = data["gender"].fillna("AUSENTE")
    data["fototipo"] = data["fitspatrick"].map(
        lambda value: "AUSENTE" if pd.isna(value) else f"FST {int(value)}")
    data["faixa_etaria"] = pd.cut(data["age"], [0, 50, 65, 200],
                                  labels=["ate_50", "51_65", "acima_65"]).astype(str)
    data["dados_demograficos"] = np.where(data["gender"].isna(), "ausentes", "presentes")
    return data


GROUPINGS = {
    "sexo": ["FEMALE", "MALE"],
    "fototipo": ["FST 1", "FST 2", "FST 3", "FST 4", "FST 5", "FST 6"],
    "faixa_etaria": ["ate_50", "51_65", "acima_65"],
    "dados_demograficos": ["presentes", "ausentes"],
}


def subgroup_tables(data: pd.DataFrame, bootstrap: PatientBootstrap):
    labels, predictions = data["label"].to_numpy(), data["pred"].to_numpy()
    support_rows, cell_rows, difference_rows = [], [], []
    replicate_rows = list(bootstrap.replicate_rows())

    for grouping, groups in GROUPINGS.items():
        eligible_cells = {}
        for group in groups:
            mask = (data[grouping] == group).to_numpy()
            n_patients = int(data.loc[mask, "patient_id"].nunique())
            patients_per_class = {
                name: int(data.loc[mask & (labels == index), "patient_id"].nunique())
                for index, name in enumerate(CLASS_NAMES)}
            group_ok = n_patients >= MIN_PATIENTS_GROUP
            support_rows.append({
                "agrupamento": grouping, "grupo": group, "n_imagens": int(mask.sum()),
                "n_pacientes": n_patients, "apto_inferencia": group_ok,
                "macro_f1_permitido": group_ok and min(patients_per_class.values()) >= MIN_PATIENTS_CELL,
                **{f"pacientes_{name}": count for name, count in patients_per_class.items()},
            })
            if not group_ok:
                continue
            for index, name in enumerate(CLASS_NAMES):
                if patients_per_class[name] < MIN_PATIENTS_CELL:
                    continue
                cell = mask & (labels == index)
                values = [float((predictions[rows][cell[rows]] == index).mean())
                          if cell[rows].any() else np.nan for rows in replicate_rows]
                eligible_cells[(group, index)] = (cell, values)
                cell_rows.append({
                    "agrupamento": grouping, "grupo": group, "diagnostico": name,
                    "n_imagens": int(cell.sum()), "n_pacientes": patients_per_class[name],
                    "sensibilidade": float((predictions[cell] == index).mean()),
                    **interval(values),
                })

        for position, first in enumerate(groups):
            for second in groups[position + 1:]:
                for index, name in enumerate(CLASS_NAMES):
                    if (first, index) not in eligible_cells or (second, index) not in eligible_cells:
                        continue
                    first_cell, first_values = eligible_cells[(first, index)]
                    second_cell, second_values = eligible_cells[(second, index)]
                    point = float((predictions[first_cell] == index).mean()
                                  - (predictions[second_cell] == index).mean())
                    differences = np.asarray(first_values) - np.asarray(second_values)
                    difference_rows.append({
                        "agrupamento": grouping, "grupo_a": first, "grupo_b": second,
                        "diagnostico": name, "diferenca_sensibilidade_a_menos_b": point,
                        **interval(differences.tolist()),
                    })
    return pd.DataFrame(support_rows), pd.DataFrame(cell_rows), pd.DataFrame(difference_rows)


def evaluate(cohort: str) -> dict:
    final_dir = RESULTS_DIR / f"final_{cohort}"
    predictions_path = final_dir / "predicoes_teste.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predições inexistentes: {predictions_path}")
    data = add_groups(pd.read_csv(predictions_path))
    data["patient_id"] = data["patient_id"].astype(str)
    out_dir = final_dir / "avaliacao"
    out_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = PatientBootstrap(data)
    data = bootstrap.data
    summary, per_class = global_metrics(data, bootstrap)
    matrix = pd.DataFrame(confusion(data["label"].to_numpy(), data["pred"].to_numpy()),
                          index=[f"real_{name}" for name in CLASS_NAMES],
                          columns=[f"prev_{name}" for name in CLASS_NAMES])
    support, cells, differences = subgroup_tables(data, bootstrap)

    per_class.to_csv(out_dir / "metricas_por_classe.csv", index=False)
    matrix.to_csv(out_dir / "matriz_confusao.csv")
    support.to_csv(out_dir / "suporte_subgrupos.csv", index=False)
    cells.to_csv(out_dir / "sensibilidade_por_subgrupo.csv", index=False)
    differences.to_csv(out_dir / "diferencas_entre_subgrupos.csv", index=False)
    result = {
        "coorte": cohort,
        "n_imagens_teste": len(data),
        "n_pacientes_teste": int(data["patient_id"].nunique()),
        "bootstrap": {"replicacoes": N_BOOTSTRAP, "unidade": "paciente", "semente": SEED,
                      "intervalo": "percentil 2,5–97,5"},
        "limiares": {"pacientes_por_grupo": MIN_PATIENTS_GROUP,
                     "pacientes_por_celula": MIN_PATIENTS_CELL},
        "metricas_globais": summary,
    }
    (out_dir / "metricas_globais.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coorte", choices=["principal", "sensibilidade"], default="principal")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.coorte), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
