"""Controle do sinal H/M condicionado à probabilidade prevista (NIH; análise exploratória, Dia 15).

Pergunta: a dissimilaridade entre os mapas médios de mulheres e homens com derrame (sinal H/M)
mede algo além da diferença entre as predições dos dois grupos? Com q alto, as mulheres com
derrame recebem P(derrame) perto de zero; mapas de imagens com predições diferentes tendem a
diferir por isso mesmo, e o sinal poderia ser só um reflexo da saída do modelo.

Para cada execução, nas 573 imagens de teste com derrame:
  - pareado: o logit de P(derrame) na imagem limpa (predicoes_teste.csv) é dividido em faixas
    fixas; em cada faixa sorteiam-se min(nF, nM) mulheres e o mesmo número de homens. Os dois
    grupos pareados têm, então, a mesma distribuição de predições, e a dissimilaridade restante
    não pode vir da diferença de saída;
  - verdadeiros positivos: só as imagens com P(derrame) >= 0,5, nos dois sexos;
  - sem condição: o sinal original, recalculado para conferir que reproduz
    controle_dissimilaridade.csv.
Em cada versão: D por sexo, piso de ruído (metades sorteadas com os mesmos tamanhos, média de
20 sorteios) e excesso = D - piso. Versões com menos de MIN_POR_SEXO imagens em algum sexo ficam
vazias (NaN). Nos modelos B com q = 1, o Grad-CAM sai, como no controle original.

Não pré-registrada: motivada pela simulação de banca (Organização/documentos/Simulado_Banca_TCC1).
Saídas: xai/controle_condicionado.csv (por execução) e xai/controle_condicionado{,_replica,
_combinado}.json. Retomável: execuções já gravadas são puladas.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

import controle_dissimilaridade_nih as controle
import xai_nih_chestxray14 as xai

N_SPLITS = 20
MIN_POR_SEXO = 20
# Faixas de logit(P(derrame)); as extremas pegam as predições muito confiantes.
LOGIT_EDGES = np.array([-np.inf, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, np.inf])


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def matched_subset(prob: np.ndarray, female: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Índices de um subconjunto com o mesmo número de mulheres e homens em cada faixa de logit."""
    bins = np.digitize(logit(prob), LOGIT_EDGES[1:-1])
    chosen = []
    for b in np.unique(bins):
        women = np.flatnonzero((bins == b) & female)
        men = np.flatnonzero((bins == b) & ~female)
        k = min(len(women), len(men))
        if k:
            chosen += list(rng.choice(women, k, replace=False)) + list(rng.choice(men, k, replace=False))
    return np.sort(np.array(chosen, dtype=int))


def signal(maps: np.ndarray, female: np.ndarray, subset: np.ndarray) -> dict:
    """D por sexo, piso de ruído e excesso dentro de um subconjunto de imagens."""
    n_f = int(female[subset].sum())
    n_m = len(subset) - n_f
    if n_f < MIN_POR_SEXO or n_m < MIN_POR_SEXO:
        return {"sexo": np.nan, "aleatorio": np.nan, "excesso": np.nan, "n_f": n_f, "n_m": n_m}
    part = maps[subset]
    sexo = controle.dissimilarity(part, female[subset])
    floors = []
    for seed in range(N_SPLITS):
        mask = np.zeros(len(subset), dtype=bool)
        mask[np.random.default_rng(seed).permutation(len(subset))[:n_f]] = True
        floors.append(controle.dissimilarity(part, mask))
    floor = float(np.mean(floors))
    return {"sexo": sexo, "aleatorio": floor, "excesso": sexo - floor, "n_f": n_f, "n_m": n_m}


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    out_path = xai.OUT_DIR / "controle_condicionado.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["execucao"])
    cohort = pd.read_csv(xai.nih.COHORTS["principal"])
    test = cohort[cohort["particao"] == "teste"].sort_values("img_id").reset_index(drop=True)
    positives = test[test["rotulo"] == 1].reset_index(drop=True)
    crops = xai.crops_of(xai.load_images(positives["img_id"], device))
    female = positives["sexo"].to_numpy() == "F"

    rows = done.to_dict("records")
    for name in xai.run_names("combinado"):
        if name in set(done["execucao"]) or not (xai.RUN_DIR / name / "execucao.json").exists():
            continue
        predictions = pd.read_csv(xai.RUN_DIR / name / "predicoes_teste.csv")
        prob = positives[["img_id"]].merge(predictions, on="img_id", how="left")["p_derrame"].to_numpy()
        assert not np.isnan(prob).any(), name
        matched = matched_subset(prob, female, np.random.default_rng(0))
        true_pos = np.flatnonzero(prob >= 0.5)
        everything = np.arange(len(positives))
        model = xai.load_model(name, device)
        row = {"execucao": name, "experimento": name[0],
               "q": float(name.split("_q")[1].split("_s")[0]), "semente": int(name.split("_s")[-1]),
               "p_media_F": float(prob[female].mean()), "p_media_M": float(prob[~female].mean()),
               "p_media_F_pareado": float(prob[matched][female[matched]].mean()) if female[matched].any() else np.nan,
               "p_media_M_pareado": float(prob[matched][~female[matched]].mean()) if (~female[matched]).any() else np.nan}
        for method, function in (("gradcam", xai.gradcam_maps), ("ig", xai.ig_maps)):
            maps = function(model, crops)
            for version, subset in (("todos", everything), ("pareado", matched), ("vp", true_pos)):
                for key, value in signal(maps, female, subset).items():
                    row[f"{version}_{key}_{method}" if key not in ("n_f", "n_m") else f"{version}_{key}"] = value
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float) and "excesso" in k},
              "pareado n =", row["pareado_n_f"], "+", row["pareado_n_m"], flush=True)
        del model, maps
        torch.cuda.empty_cache()  # libera a reserva entre execuções (Linux); não muda resultados

    table = pd.DataFrame(rows)
    degenerate = (table["experimento"] == "B") & (table["q"] == 1.0)
    table.loc[degenerate, [c for c in table.columns if c.endswith("_gradcam")]] = np.nan
    for conjunto, suffix in xai.SUFFIXES.items():
        summary = summarize(table[table["execucao"].isin(xai.run_names(conjunto))])
        (xai.OUT_DIR / f"controle_condicionado{suffix}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summarize(table[table["execucao"].isin(xai.run_names("combinado"))])["A"]["spearman_com_q"],
                     ensure_ascii=False, indent=1))


def summarize(table: pd.DataFrame) -> dict:
    rng = np.random.default_rng(xai.nih.SEED)
    summary = {"nota": "exploratória, não pré-registrada; B com q = 1 excluído das colunas de Grad-CAM"}
    signal_columns = [c for c in table.columns
                      if c.split("_")[0] in ("todos", "pareado", "vp") and c.split("_")[1] in ("sexo", "aleatorio", "excesso")]
    count_columns = [c for c in table.columns if c.endswith("_n_f") or c.endswith("_n_m") or c.startswith("p_media")]

    def correlation(part, column):
        valid = part[["q", column]].dropna()
        if len(valid) < 8 or valid["q"].nunique() < 3:
            return {"rho": None, "ic95": None, "n_execucoes": len(valid)}
        return {**xai.spearman_with_ci(valid["q"].to_numpy(), valid[column].to_numpy(), rng),
                "n_execucoes": len(valid)}

    for experiment in ("A", "B"):
        part = table[table["experimento"] == experiment]
        summary[experiment] = {
            "medianas_por_q": part.groupby("q")[signal_columns + count_columns].median().round(4).to_dict(),
            "spearman_com_q": {column: correlation(part, column) for column in signal_columns},
            # Com q = 1 restam 15 a 41 pares por sexo no experimento A; a tendência sem esse nível
            # (leitura decidida depois de ver os dados) mostra o que ocorre com viés moderado.
            "spearman_com_q_ate_0.9": {column: correlation(part[part["q"] < 1], column)
                                       for column in signal_columns},
        }
    return summary


if __name__ == "__main__":
    main()
