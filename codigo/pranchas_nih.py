"""Pranchas de mapas de atribuição do NIH ChestX-ray14 (semente 42 de cada nível).

  prancha_A_exemplos.png      4 imagens de teste com derrame (2 mulheres, 2 homens, sorteadas),
                              Grad-CAM e IG da classe derrame nos modelos A com q = 0,5 e q = 1,0;
  prancha_A_mapas_medios.png  mapas médios (soma 1) de mulheres e de homens com derrame e a
                              diferença H − M, para q = 0,5 e q = 1,0;
  prancha_B_exemplos.png      3 imagens negativas de teste (sorteadas) com o quadrado inserido,
                              Grad-CAM e IG da classe derrame nos quatro níveis de q.
As imagens são sorteadas, não escolhidas, para que a prancha mostre casos típicos.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

import xai_nih_chestxray14 as xai

SEED = 42
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
HEAT = "inferno"


def br(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")
DIVERGING = "RdBu_r"


def show(ax, image, heat=None, title=None, cmap=HEAT, symmetric=False) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if heat is not None:
        if symmetric:
            limit = np.percentile(np.abs(heat), 99.5) or 1.0
            ax.imshow(heat, cmap=cmap, vmin=-limit, vmax=limit, alpha=0.6)
        else:
            top = np.percentile(heat, 99.5) or 1.0
            scaled = np.clip(heat / top, 0, 1)
            ax.imshow(scaled, cmap=cmap, vmin=0, vmax=1, alpha=0.25 + 0.55 * scaled)
    if title:
        ax.set_title(title, fontsize=8, color=INK_SECONDARY)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def probabilities(model, crops: torch.Tensor) -> np.ndarray:
    return xai.nih.probability(model, lambda a, b: crops[a:b], len(crops))


def maps_for(name, crops, device) -> dict:
    model = xai.load_model(name, device)
    return {"gradcam": xai.gradcam_maps(model, crops), "ig": xai.ig_maps(model, crops),
            "p": probabilities(model, crops)}


def plate_a_examples(test, device) -> None:
    rng = np.random.default_rng(SEED)
    picks = []
    for sex in ("F", "M"):
        members = np.flatnonzero((test["rotulo"] == 1) & (test["sexo"] == sex))
        picks.extend(rng.choice(members, 2, replace=False))
    rows = test.iloc[picks].reset_index(drop=True)
    crops = xai.crops_of(xai.load_images(rows["img_id"], device))
    images = crops[:, 0].cpu().numpy()
    runs = {q: maps_for(f"A_q{q:g}_s{SEED}", crops, device) for q in (0.5, 1.0)}
    columns = [("gradcam", 0.5), ("gradcam", 1.0), ("ig", 0.5), ("ig", 1.0)]
    figure, axes = plt.subplots(len(rows), 5, figsize=(11, 2.35 * len(rows)), facecolor=SURFACE)
    for i, row in rows.iterrows():
        sex = "mulher" if row["sexo"] == "F" else "homem"
        show(axes[i, 0], images[i], title=f"{sex}, {int(row['idade'])} anos, derrame")
        for j, (method, q) in enumerate(columns, start=1):
            label = "Grad-CAM" if method == "gradcam" else "IG"
            show(axes[i, j], images[i], runs[q][method][i],
                 title=f"{label}, q = {br(q, 2 if q < 1 else 0)}\nP(derrame) = {br(runs[q]['p'][i])}")
    figure.suptitle("Experimento A: mapas da classe derrame em imagens sorteadas do teste\n"
                    "modelo sem viés (q = 0,5) e com viés máximo (q = 1); mapa vazio = nenhuma evidência a favor de derrame",
                    fontsize=10, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(xai.OUT_DIR / "prancha_A_exemplos.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def plate_a_means(test, device) -> None:
    positives = test[test["rotulo"] == 1].reset_index(drop=True)
    crops = xai.crops_of(xai.load_images(positives["img_id"], device))
    background = crops[:, 0].mean(dim=0).cpu().numpy()
    female = positives["sexo"].to_numpy() == "F"
    figure, axes = plt.subplots(2, 6, figsize=(13.5, 5.8), facecolor=SURFACE)
    for col_offset, q in ((0, 0.5), (3, 1.0)):
        model = xai.load_model(f"A_q{q:g}_s{SEED}", device)
        for row, (method, function) in enumerate((("gradcam", xai.gradcam_maps), ("ig", xai.ig_maps))):
            maps = function(model, crops)
            mean_f, mean_m = xai.mean_normalized(maps[female]), xai.mean_normalized(maps[~female])
            label = "Grad-CAM" if method == "gradcam" else "IG"
            dissimilarity = 1 - np.corrcoef(mean_f.ravel(), mean_m.ravel())[0, 1]
            q_text = br(q, 2 if q < 1 else 0)
            show(axes[row, col_offset], background, mean_f, f"{label}, q = {q_text}\nmulheres (n = {female.sum()})")
            show(axes[row, col_offset + 1], background, mean_m, f"{label}, q = {q_text}\nhomens (n = {(~female).sum()})")
            show(axes[row, col_offset + 2], background, mean_m - mean_f,
                 f"diferença H − M\n1 − r = {br(dissimilarity, 3)}", cmap=DIVERGING, symmetric=True)
    figure.suptitle("Experimento A: mapas médios da classe derrame nas imagens de teste com derrame\n"
                    "vermelho = mais atribuição nos homens; azul = nas mulheres",
                    fontsize=10, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(xai.OUT_DIR / "prancha_A_mapas_medios.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def plate_b(test, device) -> None:
    predictions = pd.read_csv(xai.RUN_DIR / f"B_q0.5_s{SEED}" / "predicoes_teste.csv")
    rng = np.random.default_rng(SEED)
    picks = rng.choice(np.flatnonzero(predictions["rotulo"] == 0), 3, replace=False)
    rows = predictions.iloc[picks].reset_index(drop=True)
    corners = torch.as_tensor(rows[["quadrado_linha", "quadrado_coluna"]].to_numpy(), device=device)
    clean = xai.crops_of(xai.load_images(rows["img_id"], device))
    crops = xai.nih.paste(clean, corners, torch.ones(len(rows), dtype=torch.bool, device=device))
    images = crops[:, 0].cpu().numpy()
    levels = xai.nih.LEVELS
    runs = {q: maps_for(f"B_q{q:g}_s{SEED}", crops, device) for q in levels}
    figure, axes = plt.subplots(2 * len(rows), len(levels) + 1,
                                figsize=(11, 2.3 * 2 * len(rows)), facecolor=SURFACE)
    for i in range(len(rows)):
        row_c, col_c = rows.loc[i, ["quadrado_linha", "quadrado_coluna"]]
        for r, method in enumerate(("gradcam", "ig")):
            ax_row = axes[2 * i + r]
            label = "Grad-CAM" if method == "gradcam" else "IG"
            show(ax_row[0], images[i], title=f"sem achados + quadrado\n{label}")
            for j, q in enumerate(levels, start=1):
                show(ax_row[j], images[i], runs[q][method][i],
                     title=f"q = {br(q, 2 if q < 1 else 0)}\nP(derrame) = {br(runs[q]['p'][i])}")
            for ax in ax_row:
                ax.add_patch(Rectangle((col_c - 0.5, row_c - 0.5), xai.nih.PATCH, xai.nih.PATCH,
                                       fill=False, edgecolor="#2a78d6", linewidth=1))
    figure.suptitle("Experimento B: mapas da classe derrame em imagens negativas sorteadas,\n"
                    "com o quadrado inserido (contorno azul), por intensidade do atalho no treino",
                    fontsize=10, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(xai.OUT_DIR / "prancha_B_exemplos.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    cohort = pd.read_csv(xai.nih.COHORTS["principal"])
    test = cohort[cohort["particao"] == "teste"].sort_values("img_id").reset_index(drop=True)
    for filename, plate in (("prancha_A_exemplos.png", plate_a_examples),
                            ("prancha_B_exemplos.png", plate_b),
                            ("prancha_A_mapas_medios.png", plate_a_means)):
        if not (xai.OUT_DIR / filename).exists():  # para refazer, apague o arquivo
            plate(test, device)


if __name__ == "__main__":
    main()
