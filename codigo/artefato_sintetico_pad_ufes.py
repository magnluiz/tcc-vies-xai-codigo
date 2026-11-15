"""Experimento de artefato sintético com intensidade controlada no PAD-UFES-20.

Um quadrado verde (20 × 20 pixels, num canto da imagem) é inserido nas imagens de treino com
probabilidade q nas lesões NEV e 1 − q nas demais; q = 0,5 equivale a nenhuma correlação.
Para cada nível e semente, treina-se a ResNet-18 com a receita final e mede-se:
  - a dependência real do modelo no artefato (verdade de referência): quanto inserir o
    quadrado em imagens de teste não NEV aumenta a probabilidade de NEV e quantas viram NEV;
  - o sinal das explicações: fração da atribuição (Grad-CAM e IG) que cai no quadrado.
"""

import argparse
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

import treinar_pad_ufes as treino
from xai_atribuicoes import GradCAM, integrated_gradients, mass_fraction


OUT_DIR = treino.RESULTS_DIR / "artefato_sintetico"
TARGET = treino.CLASS_NAMES.index("NEV")
LEVELS = [0.5, 0.75, 0.9, 1.0]
N_SEEDS = 5
PATCH = 20
PATCH_COLOR = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1)
CORNER_MARGIN = (8, 40)  # distância do canto, em pixels, sorteada nesse intervalo
IG_STEPS = 32

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_GRADCAM = "#2a78d6"
COLOR_IG = "#eb6834"


def sample_corners(n: int, generator: torch.Generator, device) -> torch.Tensor:
    """Canto superior esquerdo (linha, coluna) do quadrado, num dos quatro cantos do recorte."""
    low, high = CORNER_MARGIN
    offset = torch.randint(low, high + 1, (n, 2), generator=generator, device=device)
    far = torch.rand(n, 2, generator=generator, device=device) < 0.5
    far_position = treino.CROP - PATCH - offset
    return torch.where(far, far_position, offset)


def paste(batch: torch.Tensor, corners: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Insere o quadrado nas imagens selecionadas (lote em [0, 1], já recortado)."""
    batch = batch.clone()
    color = PATCH_COLOR.to(batch.device, batch.dtype)
    for index in torch.nonzero(selected).flatten().tolist():
        row, column = corners[index].tolist()
        batch[index, :, row:row + PATCH, column:column + PATCH] = color
    return batch


def masks_for(corners: torch.Tensor) -> np.ndarray:
    masks = np.zeros((len(corners), treino.CROP, treino.CROP), dtype=np.float32)
    for index, (row, column) in enumerate(corners.tolist()):
        masks[index, row:row + PATCH, column:column + PATCH] = 1.0
    return masks


def train_run(q, seed, images, positions, labels, schedule, device):
    rng = np.random.default_rng(seed)
    probability = np.where(labels == TARGET, q, 1 - q)
    has_artifact = torch.as_tensor(rng.random(len(labels)) < probability, device=device)
    treino.set_seed(seed)
    model = treino.build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule[0],
                                  weight_decay=treino.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type)
    generator = torch.Generator(device=device).manual_seed(seed)
    positions_all = torch.as_tensor(positions, device=device)
    labels_all = torch.as_tensor(labels.astype(np.int64), device=device)
    for lr in schedule:
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        order = torch.randperm(len(positions), generator=generator, device=device)
        for start in range(0, len(order), treino.BATCH_SIZE):
            batch_index = order[start:start + treino.BATCH_SIZE]
            if len(batch_index) < 2:
                continue
            crops = treino.augment(treino.to_input(images[positions_all[batch_index]]), generator)
            corners = sample_corners(len(batch_index), generator, device)
            crops = paste(crops, corners, has_artifact[batch_index])
            with torch.autocast(device.type, dtype=torch.float16):
                logits = model(treino.normalize(crops))
            loss = F.cross_entropy(logits.float(), labels_all[batch_index])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model.eval(), float(has_artifact.float().mean())


@torch.no_grad()
def probabilities(model, crops: torch.Tensor) -> torch.Tensor:
    outputs = []
    for start in range(0, len(crops), 128):
        with torch.autocast("cuda", dtype=torch.float16):
            outputs.append(model(treino.normalize(crops[start:start + 128])).float())
    return torch.softmax(torch.cat(outputs), dim=1)


def attribution_fractions(model, crops, classes, masks, device) -> dict:
    cam = GradCAM(model)
    gradcam_maps = []
    for start in range(0, len(crops), 64):
        gradcam_maps.append(cam(treino.normalize(crops[start:start + 64]),
                                classes[start:start + 64]).cpu().numpy())
    cam.remove()
    baseline = treino.normalize(torch.zeros(1, 3, treino.CROP, treino.CROP, device=device))[0]
    ig_maps = []
    for start in range(0, len(crops), 16):
        ig_maps.append(integrated_gradients(
            model, treino.normalize(crops[start:start + 16]), classes[start:start + 16],
            baseline, IG_STEPS).cpu().numpy())
    return {"gradcam": mass_fraction(np.concatenate(gradcam_maps), masks),
            "ig": mass_fraction(np.concatenate(ig_maps), masks)}


def evaluate_run(model, images, test_positions, test_labels, device, seed):
    clean = treino.center_crop(treino.to_input(images[torch.as_tensor(test_positions, device=device)]))
    generator = torch.Generator(device=device).manual_seed(10_000 + seed)
    corners = sample_corners(len(clean), generator, device)
    with_artifact = paste(clean, corners, torch.ones(len(clean), dtype=torch.bool, device=device))
    p_clean, p_artifact = probabilities(model, clean), probabilities(model, with_artifact)
    pred_clean = p_clean.argmax(1).cpu().numpy()
    pred_artifact = p_artifact.argmax(1).cpu().numpy()

    others = test_labels != TARGET
    delta = (p_artifact[:, TARGET] - p_clean[:, TARGET]).cpu().numpy()
    flipped = (pred_artifact == TARGET) & (pred_clean != TARGET)

    masks = masks_for(corners)
    index = torch.as_tensor(np.flatnonzero(others), device=device)
    predicted = torch.as_tensor(pred_artifact[others], device=device)
    nev = torch.full_like(predicted, TARGET)
    explained = with_artifact[index]
    fractions_pred = attribution_fractions(model, explained, predicted, masks[others], device)
    fractions_nev = attribution_fractions(model, explained, nev, masks[others], device)

    per_image = pd.DataFrame({
        "delta_p_nev": delta[others], "virou_nev": flipped[others],
        "fracao_gradcam_prevista": fractions_pred["gradcam"],
        "fracao_ig_prevista": fractions_pred["ig"],
        "fracao_gradcam_nev": fractions_nev["gradcam"],
        "fracao_ig_nev": fractions_nev["ig"],
    })
    summary = {
        "macro_f1_teste_limpo": float(f1_score(test_labels, pred_clean, average="macro")),
        "taxa_virou_nev": float(flipped[others].mean()),
        "delta_p_nev_medio": float(delta[others].mean()),
        **{f"{column}_media": float(per_image[column].mean())
           for column in per_image.columns if column.startswith("fracao")},
    }
    return summary, per_image


def run_experiment(device) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = json.loads((treino.RESULTS_DIR / "cv_principal/decisao_selecao.json")
                          .read_text(encoding="utf-8"))
    schedule = decision["esquema_lr_treino_final"]
    metadata, partitions, _ = treino.load_cohort("principal")
    development = set(partitions.loc[partitions["particao"] == "desenvolvimento", "patient_id"])
    test = set(partitions.loc[partitions["particao"] == "teste", "patient_id"])
    images, position_by_id = treino.load_cache(device)
    _, train_positions, train_labels = treino.rows_for(metadata, development, position_by_id)
    _, test_positions, test_labels = treino.rows_for(metadata, test, position_by_id)

    runs, images_rows = [], []
    for q in LEVELS:
        for repetition in range(N_SEEDS):
            seed = treino.SEED + repetition
            started = time.time()
            model, artifact_share = train_run(q, seed, images, train_positions, train_labels,
                                              schedule, device)
            summary, per_image = evaluate_run(model, images, test_positions, test_labels,
                                              device, seed)
            runs.append({"q": q, "semente": seed, "fracao_treino_com_artefato": artifact_share,
                         **summary, "segundos": time.time() - started})
            images_rows.append(per_image.assign(q=q, semente=seed))
            print(f"q={q} semente={seed}: virou NEV {summary['taxa_virou_nev']:.3f}, "
                  f"Grad-CAM {summary['fracao_gradcam_prevista_media']:.3f}, "
                  f"IG {summary['fracao_ig_prevista_media']:.3f}, "
                  f"macro-F1 limpo {summary['macro_f1_teste_limpo']:.3f}", flush=True)
    return pd.DataFrame(runs), pd.concat(images_rows, ignore_index=True)


def analyse(runs: pd.DataFrame, per_image: pd.DataFrame) -> dict:
    area = PATCH * PATCH / treino.CROP ** 2
    result = {"area_relativa_do_quadrado": area, "n_execucoes": len(runs)}
    for method in ("gradcam", "ig"):
        for target in ("prevista", "nev"):
            column = f"fracao_{method}_{target}"
            by_q = spearmanr(runs["q"], runs[f"{column}_media"]).statistic
            by_reliance = spearmanr(runs["taxa_virou_nev"], runs[f"{column}_media"]).statistic
            valid = per_image.dropna(subset=[column])
            valid = valid[valid["q"] > 0.5]
            auc = roc_auc_score(valid["virou_nev"], valid[column]) if valid["virou_nev"].nunique() == 2 else None
            result[column] = {"spearman_com_q": float(by_q),
                              "spearman_com_taxa_virou_nev": float(by_reliance),
                              "auc_por_imagem_identifica_virou_nev_q_maior_0_5": auc}
    return result


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot(runs: pd.DataFrame, path) -> None:
    area = PATCH * PATCH / treino.CROP ** 2
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6), facecolor=SURFACE)
    x = np.arange(len(LEVELS))
    jitter = np.linspace(-0.12, 0.12, N_SEEDS)
    grouped = runs.groupby("q")

    for position, (q, rows) in enumerate(grouped):
        left.scatter(position + jitter[:len(rows)], rows["taxa_virou_nev"], s=22,
                     color=COLOR_GRADCAM, alpha=0.45, linewidths=0)
    left.plot(x, grouped["taxa_virou_nev"].median().to_numpy(), color=COLOR_GRADCAM,
              linewidth=2, marker="o", markersize=6)
    left.set_title("Dependência real: imagens não NEV que viram NEV", fontsize=10,
                   color=INK, loc="left")
    left.set_ylabel("Proporção do teste", color=INK_SECONDARY)

    for color, column, label, offset in ((COLOR_GRADCAM, "fracao_gradcam_prevista_media", "Grad-CAM", -0.06),
                                         (COLOR_IG, "fracao_ig_prevista_media", "IG", 0.06)):
        for position, (q, rows) in enumerate(grouped):
            right.scatter(position + offset + jitter[:len(rows)] / 2, rows[column], s=18,
                          color=color, alpha=0.45, linewidths=0)
        medians = grouped[column].median().to_numpy()
        right.plot(x + offset, medians, color=color, linewidth=2, marker="o", markersize=6,
                   label=label)
        right.annotate(label, (x[-1] + offset, medians[-1]), xytext=(8, 0),
                       textcoords="offset points", va="center", fontsize=9, color=INK_SECONDARY)
    right.axhline(area, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    right.annotate("área do quadrado", (x[0] - 0.3, area), xytext=(0, 4), textcoords="offset points",
                   fontsize=8, color=MUTED)
    right.set_title("Sinal da explicação: atribuição no quadrado", fontsize=10, color=INK, loc="left")
    right.set_ylabel("Fração da atribuição total", color=INK_SECONDARY)
    right.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_SECONDARY)

    for ax in (left, right):
        style_axes(ax)
        ax.set_xticks(x, [f"q = {q:g}" for q in LEVELS])
        ax.set_xlabel("Correlação artefato–NEV no treino", color=INK_SECONDARY)
        ax.set_xlim(-0.5, len(LEVELS) - 0.3)
        ax.set_ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, per_image = run_experiment(device)
    runs.to_csv(OUT_DIR / "execucoes.csv", index=False)
    per_image.to_csv(OUT_DIR / "por_imagem.csv", index=False)
    result = analyse(runs, per_image)
    result["configuracao"] = {"classe_alvo": "NEV", "niveis_q": LEVELS, "sementes": N_SEEDS,
                              "quadrado_pixels": PATCH, "cor": "verde (0, 1, 0)",
                              "passos_ig": IG_STEPS, "linha_de_base_ig": "imagem preta",
                              "avaliacao": "teste da coorte principal, imagens não NEV com o quadrado"}
    (OUT_DIR / "analise.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    plot(runs, OUT_DIR / "artefato_sintetico.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
