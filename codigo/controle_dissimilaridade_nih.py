"""Controles do sinal exploratório "diferença entre os mapas médios de mulheres e homens" (NIH).

O sinal cresceu com q no experimento A, mas o modelo também piora com q, e mapas mais ruidosos
afastariam os mapas médios de quaisquer dois grupos. Para cada execução (A e B), nas imagens de
teste com derrame, compara-se a dissimilaridade por sexo com:
  - a de duas metades sorteadas com os mesmos tamanhos dos grupos de sexo (piso de ruído; média
    de 20 sorteios, sementes 0 a 19);
  - a de um atributo não manipulado: idade até a mediana × acima dela.
No experimento B o sexo é balanceado no treino; ali a dissimilaridade por sexo não deveria
crescer com q (controle negativo). Análise exploratória, não pré-registrada (Dia 11), nas sementes
42 a 46; nas sementes 47 a 51 é a hipótese confirmatória pré-registrada (ficha, ajuste 2; Dia 12),
avaliada em `controle_dissimilaridade_replica.json`.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import xai_nih_chestxray14 as xai

N_SPLITS = 20


def dissimilarity(maps: np.ndarray, first: np.ndarray) -> float:
    a, b = xai.mean_normalized(maps[first]), xai.mean_normalized(maps[~first])
    return float(1 - np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    out_path = xai.OUT_DIR / "controle_dissimilaridade.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["execucao"])
    cohort = pd.read_csv(xai.nih.COHORTS["principal"])
    test = cohort[cohort["particao"] == "teste"].sort_values("img_id").reset_index(drop=True)
    positives = test[test["rotulo"] == 1].reset_index(drop=True)
    crops = xai.crops_of(xai.load_images(positives["img_id"], device))
    female = positives["sexo"].to_numpy() == "F"
    older = positives["idade"].to_numpy() > np.median(positives["idade"])
    splits = []
    for seed in range(N_SPLITS):
        mask = np.zeros(len(positives), dtype=bool)
        mask[np.random.default_rng(seed).permutation(len(positives))[:female.sum()]] = True
        splits.append(mask)

    rows = done.to_dict("records")
    for name in xai.run_names("combinado"):
        if name in set(done["execucao"]) or not (xai.RUN_DIR / name / "execucao.json").exists():
            continue
        model = xai.load_model(name, device)
        row = {"execucao": name, "experimento": name[0],
               "q": float(name.split("_q")[1].split("_s")[0]), "semente": int(name.split("_s")[-1])}
        for method, function in (("gradcam", xai.gradcam_maps), ("ig", xai.ig_maps)):
            maps = function(model, crops)
            row[f"sexo_{method}"] = dissimilarity(maps, female)
            row[f"idade_{method}"] = dissimilarity(maps, older)
            row[f"aleatorio_{method}"] = float(np.mean([dissimilarity(maps, s) for s in splits]))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)

    table = pd.DataFrame(rows)
    # Nos modelos B com q = 1, sem o quadrado o modelo não vê derrame: só 5 a 28 dos 573 mapas de
    # Grad-CAM das imagens com derrame não são vazios (contado no Dia 11). A comparação entre
    # grupos fica indefinida, então essas execuções saem das colunas de Grad-CAM.
    degenerate = (table["experimento"] == "B") & (table["q"] == 1.0)
    table.loc[degenerate, [c for c in table.columns if c.endswith("_gradcam")]] = np.nan
    table["excesso_gradcam"] = table["sexo_gradcam"] - table["aleatorio_gradcam"]
    table["excesso_ig"] = table["sexo_ig"] - table["aleatorio_ig"]
    for conjunto, suffix in xai.SUFFIXES.items():
        # Antes da réplica treinada, os conjuntos "replica" e "combinado" ficam incompletos; o
        # resumo deles só é gravado quando todas as execuções existem (rodar de novo depois).
        faltam = set(xai.run_names(conjunto)) - set(table["execucao"])
        if faltam:
            print(conjunto, f"pulado: faltam {len(faltam)} execuções")
            continue
        summary = summarize(table[table["execucao"].isin(xai.run_names(conjunto))])
        if conjunto == "replica":
            summary["hipotese_confirmatoria"] = confirmatory(summary)
        (xai.OUT_DIR / f"controle_dissimilaridade{suffix}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(conjunto, json.dumps(summary.get("hipotese_confirmatoria", ""), ensure_ascii=False))


def summarize(table: pd.DataFrame) -> dict:
    rng = np.random.default_rng(xai.nih.SEED)
    summary = {"nota": "B com q = 1 excluído das colunas de Grad-CAM (mapas quase todos vazios)"}
    columns = [c for c in table.columns if c.split("_")[0] in ("sexo", "idade", "aleatorio", "excesso")]

    def correlation(part, column):
        valid = part[["q", column]].dropna()
        return {**xai.spearman_with_ci(valid["q"].to_numpy(), valid[column].to_numpy(), rng),
                "n_execucoes": len(valid)}

    for experiment in ("A", "B"):
        part = table[table["experimento"] == experiment]
        summary[experiment] = {
            "medianas_por_q": part.groupby("q")[columns].median().round(4).to_dict(),
            # "excesso" = sexo menos metades sorteadas (sinal acima do piso de ruído), por execução.
            "spearman_com_q": {column: correlation(part, column) for column in columns},
        }
    return summary


def confirmatory(summary: dict) -> dict:
    """Regra de decisão pré-registrada (ficha, ajuste 2): HC-A e HC-B, por método."""
    result = {}
    for method in ("gradcam", "ig"):
        a = summary["A"]["spearman_com_q"][f"excesso_{method}"]
        b = summary["B"]["spearman_com_q"][f"excesso_{method}"]
        hc_a = a["ic95"][0] > 0
        hc_b = b["ic95"][0] <= 0
        result[method] = {"HC_A": hc_a, "HC_B": hc_b, "confirmada": bool(hc_a and hc_b),
                          "rho_A": a["rho"], "ic95_A": a["ic95"], "rho_B": b["rho"], "ic95_B": b["ic95"]}
    return result


if __name__ == "__main__":
    main()
