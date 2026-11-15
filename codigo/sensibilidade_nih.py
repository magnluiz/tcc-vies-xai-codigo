"""Análises de sensibilidade pré-registradas do NIH (ficha, ajuste 2; Dia 12).

S1  TCAV com 100 CAVs aleatórios (sementes 0 a 99) em todas as execuções das sementes 42 a 51.
    O CAV do conceito e o escore não mudam; muda a faixa aleatória que define a significância.
S2  IG com linha de base borrada: para cada imagem, a própria imagem com filtro gaussiano de
    sigma = 10 pixels (Sturmfels; Lundberg; Lee, 2020), 32 passos.
      (a) concentração no quadrado do experimento B, em todas as execuções;
      (b) testes de sanidade do IG (cascata e rótulos embaralhados, nas mesmas 300 imagens).
Resultados por execução ficam em cache em `xai/sensibilidade/`; `--so-analise` refaz só o resumo.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

import treinar_pad_ufes as pad
import xai_nih_chestxray14 as xai
from xai_atribuicoes import integrated_gradients, mass_fraction

N_RANDOM_CAVS = 100
BLUR_SIGMA = 10.0
BLUR_KERNEL = 61  # cobre ±3 sigma
OUT_DIR = xai.OUT_DIR / "sensibilidade"


def ig_maps_blurred(model, crops: torch.Tensor) -> np.ndarray:
    target = torch.ones(1, dtype=torch.long, device=crops.device)
    maps = []
    for start in range(0, len(crops), 16):
        batch = crops[start:start + 16]
        baselines = pad.normalize(TF.gaussian_blur(batch, BLUR_KERNEL, BLUR_SIGMA))
        for image, baseline in zip(pad.normalize(batch), baselines):
            maps.append(integrated_gradients(model, image[None], target, baseline, xai.IG_STEPS)[0].cpu().numpy())
    return np.stack(maps)


def tcav_100(name: str, model, context: xai.Context) -> dict:
    result = xai.tcav(model, context.concept_crops, context.concepts["sexo"].to_numpy(),
                      context.test_crops[context.positive], n_random=N_RANDOM_CAVS)
    return {layer: {key: value for key, value in values.items() if key != "escores_aleatorios"}
            for layer, values in result.items()}


def ig_square(name: str, model, context: xai.Context) -> pd.DataFrame:
    """Mesmas imagens e máscaras de `xai.explain_run_b`, com a linha de base borrada."""
    predictions = pd.read_csv(xai.RUN_DIR / name / "predicoes_teste.csv")
    negative = np.flatnonzero(predictions["rotulo"].to_numpy() == 0)
    corners = predictions.loc[negative, ["quadrado_linha", "quadrado_coluna"]].to_numpy()
    device = context.test_crops.device
    crops = xai.nih.paste(context.test_crops[negative], torch.as_tensor(corners, device=device),
                          torch.ones(len(negative), dtype=torch.bool, device=device))
    masks = np.zeros((len(negative), xai.nih.CROP, xai.nih.CROP), dtype=np.float32)
    for index, (row, column) in enumerate(corners):
        masks[index, row:row + xai.nih.PATCH, column:column + xai.nih.PATCH] = 1.0
    clean, patched = predictions.loc[negative, "p_derrame"], predictions.loc[negative, "p_derrame_com_quadrado"]
    rows = predictions.loc[negative, ["img_id"]].reset_index(drop=True)
    rows["virou_derrame"] = ((clean < 0.5) & (patched >= 0.5)).to_numpy()
    rows["conc_quadrado_ig_borrada"] = mass_fraction(ig_maps_blurred(model, crops), masks) / xai.PATCH_AREA
    return rows


def sanity_blurred(context: xai.Context, device) -> dict:
    """Como `xai.sanity`, só para o IG, com a linha de base borrada."""
    subset = np.sort(np.random.default_rng(xai.nih.SEED).choice(len(context.test), xai.N_IG_SANITY, replace=False))
    crops = context.test_crops[torch.as_tensor(subset, device=device)]
    reference = ig_maps_blurred(xai.load_model(xai.SANITY_RUN, device), crops)
    result = {"modelo_referencia": xai.SANITY_RUN, "n_imagens": len(subset), "cascata": []}
    torch.manual_seed(xai.nih.SEED)
    model = xai.load_model(xai.SANITY_RUN, device)
    for layer in xai.CASCADE:
        xai.pad_randomize(getattr(model, layer))
        if layer == "conv1":
            xai.pad_randomize(model.bn1)
        result["cascata"].append({"camada_randomizada": layer,
                                  **xai.describe(xai.rank_correlations(reference, ig_maps_blurred(model, crops)))})
    shuffled = xai.load_model(xai.SHUFFLED_RUN, device)
    result["rotulos_embaralhados"] = xai.describe(xai.rank_correlations(reference, ig_maps_blurred(shuffled, crops)))
    return result


def compute(device) -> None:
    context = xai.Context(device)
    for folder in ("tcav100", "ig_borrada"):
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)
    for name in xai.run_names("combinado"):
        if not (xai.RUN_DIR / name / "execucao.json").exists():
            continue
        tcav_path = OUT_DIR / "tcav100" / f"{name}.json"
        square_path = OUT_DIR / "ig_borrada" / f"{name}.csv"
        if tcav_path.exists() and (name.startswith("A") or square_path.exists()):
            continue
        model = xai.load_model(name, device)
        if not tcav_path.exists():
            tcav_path.write_text(json.dumps(tcav_100(name, model, context), ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        if name.startswith("B") and not square_path.exists():
            ig_square(name, model, context).to_csv(square_path, index=False)
        print(name, flush=True)
    sanity_path = OUT_DIR / "sanidade_ig_borrada.json"
    if not sanity_path.exists():
        sanity_path.write_text(json.dumps(sanity_blurred(context, device), ensure_ascii=False, indent=2),
                               encoding="utf-8")


def summarize() -> dict:
    rng = np.random.default_rng(xai.nih.SEED)
    summary = {}
    for conjunto, suffix in xai.SUFFIXES.items():
        records = []
        for name in xai.run_names(conjunto):
            tcav_path = OUT_DIR / "tcav100" / f"{name}.json"
            if not tcav_path.exists():
                continue
            original = json.loads((xai.OUT_DIR / "por_execucao" / f"{name}.json").read_text(encoding="utf-8"))
            record = {"execucao": name, "experimento": name[0], "q": float(name.split("_q")[1].split("_s")[0]),
                      "conc_quadrado_ig_preta": original.get("conc_quadrado_ig")}
            for layer, values in json.loads(tcav_path.read_text(encoding="utf-8")).items():
                record[f"tcav_{layer}"] = values["escore"]
                record[f"tcav_{layer}_significativo_100"] = values["significativo"]
                record[f"tcav_{layer}_significativo_20"] = original["tcav_detalhe"][layer]["significativo"]
            square_path = OUT_DIR / "ig_borrada" / f"{name}.csv"
            if square_path.exists():
                record["conc_quadrado_ig_borrada"] = float(np.nanmean(pd.read_csv(square_path)["conc_quadrado_ig_borrada"]))
            records.append(record)
        if not records:
            continue
        runs = pd.DataFrame(records)
        runs.to_csv(OUT_DIR / f"execucoes{suffix}.csv", index=False)
        part = {}
        flags = [c for c in runs.columns if "significativo" in c]
        for experiment in ("A", "B"):
            subset = runs[runs["experimento"] == experiment]
            part[experiment] = {"n_execucoes": len(subset),
                                "fracao_significativa_por_q": subset.groupby("q")[flags].mean().to_dict()}
        b = runs[(runs["experimento"] == "B")].dropna(subset=["conc_quadrado_ig_borrada"])
        if len(b) > 2:
            part["B"]["ig_borrada"] = {
                "medianas_por_q": b.groupby("q")[["conc_quadrado_ig_preta", "conc_quadrado_ig_borrada"]]
                    .median().round(3).to_dict(),
                "spearman_com_q": xai.spearman_with_ci(b["q"].to_numpy(), b["conc_quadrado_ig_borrada"].to_numpy(), rng),
                "auc_por_imagem_identifica_virou": {},
            }
            for level in xai.nih.LEVELS[1:]:
                images = pd.concat([pd.read_csv(OUT_DIR / "ig_borrada" / f"{name}.csv")
                                    for name in b.loc[b["q"] == level, "execucao"]]).dropna()
                part["B"]["ig_borrada"]["auc_por_imagem_identifica_virou"][str(level)] = (
                    float(roc_auc_score(images["virou_derrame"], images["conc_quadrado_ig_borrada"]))
                    if images["virou_derrame"].nunique() == 2 else None)
        summary[conjunto] = part
    sanity_path = OUT_DIR / "sanidade_ig_borrada.json"
    if sanity_path.exists():
        summary["sanidade_ig_borrada"] = json.loads(sanity_path.read_text(encoding="utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--so-analise", action="store_true")
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.so_analise:
        compute(torch.device("cuda"))
    summary = summarize()
    (OUT_DIR / "resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
