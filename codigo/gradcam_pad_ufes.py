"""Grad-CAM e testes de sanidade (Adebayo et al., 2018) no modelo final do PAD-UFES-20.

Usa o modelo e as predições já gravados pela fase final; não refaz a predição do teste.
Produz: amostra documentada de mapas (acertos e erros por classe), randomização em cascata
dos pesos e comparação com um modelo treinado em rótulos embaralhados.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

import treinar_pad_ufes as treino
from xai_atribuicoes import GradCAM


FINAL_DIR = treino.RESULTS_DIR / "final_principal"
OUT_DIR = FINAL_DIR / "gradcam"
SAMPLES_PER_CELL = 3
# Ordem da randomização em cascata: da saída para a entrada.
CASCADE = ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]

# Tokens do modo claro (impressão) e rampa sequencial azul de uma só cor.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"
HEAT = LinearSegmentedColormap.from_list(
    "azul_sequencial", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = treino.build_model(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def compute_maps(model, images, positions, classes, device, batch=64) -> np.ndarray:
    cam = GradCAM(model)
    maps = []
    for start in range(0, len(positions), batch):
        index = torch.as_tensor(positions[start:start + batch], device=device)
        inputs = treino.normalize(treino.center_crop(treino.to_input(images[index])))
        target = torch.as_tensor(classes[start:start + batch], device=device)
        maps.append(cam(inputs, target).cpu().numpy())
    cam.remove()
    return np.concatenate(maps)


def rank_correlations(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    values = []
    for first, second in zip(reference, other):
        if first.std() == 0 or second.std() == 0:
            values.append(np.nan)  # mapa constante: correlação indefinida
        else:
            values.append(spearmanr(first.ravel(), second.ravel()).statistic)
    return np.asarray(values)


def randomize(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
        if isinstance(child, torch.nn.BatchNorm2d):
            child.reset_running_stats()


def cascade_test(model_path, images, positions, classes, reference, device) -> pd.DataFrame:
    torch.manual_seed(treino.SEED)
    model = load_model(model_path, device)
    rows = []
    for layer in CASCADE:
        randomize(getattr(model, layer))
        if layer == "conv1":
            randomize(model.bn1)
        correlations = rank_correlations(
            reference, compute_maps(model, images, positions, classes, device))
        rows.append({"etapa": f"até {layer}", "camada_randomizada": layer,
                     "mediana": float(np.nanmedian(correlations)),
                     "q1": float(np.nanpercentile(correlations, 25)),
                     "q3": float(np.nanpercentile(correlations, 75)),
                     "n_validos": int(np.isfinite(correlations).sum()),
                     "n_mapas_constantes": int(np.isnan(correlations).sum())})
    return pd.DataFrame(rows)


def random_label_test(metadata, partitions, images, position_by_id, test_positions,
                      classes, reference, epochs, device) -> dict:
    """Treina o mesmo modelo com rótulos de desenvolvimento embaralhados."""
    development = set(partitions.loc[partitions["particao"] == "desenvolvimento", "patient_id"])
    _, train_positions, train_labels = treino.rows_for(metadata, development, position_by_id)
    shuffled = np.random.default_rng(treino.SEED).permutation(train_labels)
    treino.set_seed(treino.SEED)
    model = treino.build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=treino.LEARNING_RATE,
                                  weight_decay=treino.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type)
    generator = torch.Generator(device=device).manual_seed(treino.SEED)
    for _ in range(epochs):
        treino.train_one_epoch(model, optimizer, scaler, images, train_positions,
                               shuffled, None, generator)
    train_accuracy = float((treino.predict(model, images, train_positions).argmax(1)
                            == shuffled).mean())
    correlations = rank_correlations(
        reference, compute_maps(model, images, test_positions, classes, device))
    return {"epocas": epochs, "acuracia_treino_rotulos_embaralhados": train_accuracy,
            "mediana": float(np.nanmedian(correlations)),
            "q1": float(np.nanpercentile(correlations, 25)),
            "q3": float(np.nanpercentile(correlations, 75)),
            "n_validos": int(np.isfinite(correlations).sum())}


def choose_sample(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(treino.SEED)
    chosen = []
    for index, name in enumerate(treino.CLASS_NAMES):
        for correct in (True, False):
            cell = predictions[(predictions["label"] == index)
                               & ((predictions["pred"] == index) == correct)]
            take = min(SAMPLES_PER_CELL, len(cell))
            picked = cell.iloc[sorted(rng.choice(len(cell), take, replace=False))] if take else cell
            chosen.append(picked.assign(acerto=correct))
    return pd.concat(chosen)


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_sample(sample, images, maps, path: Path) -> None:
    columns = 2 * SAMPLES_PER_CELL
    figure, axes = plt.subplots(len(treino.CLASS_NAMES), columns,
                                figsize=(columns * 1.9, len(treino.CLASS_NAMES) * 2.25),
                                facecolor=SURFACE)
    for ax in axes.ravel():
        ax.axis("off")
    for row, name in enumerate(treino.CLASS_NAMES):
        for correct, offset in ((True, 0), (False, SAMPLES_PER_CELL)):
            cell = sample[(sample["diagnostic"] == name) & (sample["acerto"] == correct)]
            for column, (_, record) in enumerate(cell.iterrows()):
                ax = axes[row, offset + column]
                picture = treino.center_crop(images[int(record["posicao"])]).permute(1, 2, 0).cpu().numpy()
                heat = maps[record["indice_mapa"]]
                ax.imshow(picture)
                # Transparência proporcional ao mapa: valores próximos de zero não cobrem a imagem.
                ax.imshow(heat, cmap=HEAT, alpha=0.65 * heat, vmin=0, vmax=1)
                predicted = treino.CLASS_NAMES[int(record["pred"])]
                ax.set_title(f"{name} → {predicted} ({record['confianca']:.2f})",
                             fontsize=7.5, color=INK if correct else INK_SECONDARY, pad=3)
        axes[row, 0].text(-0.12, 0.5, name, transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
    figure.text(0.28, 0.995, "Previsões corretas", ha="center", va="top", fontsize=11, color=INK)
    figure.text(0.76, 0.995, "Previsões incorretas", ha="center", va="top", fontsize=11, color=INK)
    figure.tight_layout(rect=(0.02, 0, 1, 0.975))
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def plot_sanity(cascade: pd.DataFrame, random_labels: dict, path: Path) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(8.0, 3.4), sharey=True, gridspec_kw={"width_ratios": [4, 1]},
        facecolor=SURFACE)
    steps = ["original"] + [f"até {layer}" for layer in CASCADE]
    medians = [1.0] + cascade["mediana"].tolist()
    q1 = [1.0] + cascade["q1"].tolist()
    q3 = [1.0] + cascade["q3"].tolist()
    x = np.arange(len(steps))
    left.fill_between(x, q1, q3, color=SERIES, alpha=0.18, linewidth=0)
    left.plot(x, medians, color=SERIES, linewidth=2, marker="o", markersize=5)
    left.set_xticks(x, steps, fontsize=8)
    left.set_ylabel("Correlação de Spearman com\no mapa do modelo treinado", color=INK_SECONDARY)
    left.set_title("Randomização em cascata dos pesos", fontsize=10, color=INK, loc="left")
    right.errorbar([0], [random_labels["mediana"]],
                   yerr=[[random_labels["mediana"] - random_labels["q1"]],
                         [random_labels["q3"] - random_labels["mediana"]]],
                   color=SERIES, marker="o", markersize=5, linewidth=2, capsize=0)
    right.set_xticks([0], ["rótulos\nembaralhados"], fontsize=8)
    right.set_xlim(-0.8, 0.8)
    right.set_title("Rótulos aleatórios", fontsize=10, color=INK, loc="left")
    for ax in (left, right):
        style_axes(ax)
        ax.axhline(0, color=BASELINE, linewidth=0.8)
        ax.set_ylim(-0.3, 1.05)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epocas-rotulos-aleatorios", type=int, default=None,
                        help="Padrão: o mesmo número de épocas do treino final.")
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    execution = json.loads((FINAL_DIR / "execucao_final.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(FINAL_DIR / "predicoes_teste.csv")
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    metadata, partitions, _ = treino.load_cohort("principal")
    images, position_by_id = treino.load_cache(device)
    predictions["posicao"] = predictions["img_id"].map(position_by_id)
    logits = predictions[[f"logit_{name}" for name in treino.CLASS_NAMES]].to_numpy()
    predictions["confianca"] = torch.softmax(torch.from_numpy(logits), dim=1).amax(dim=1).numpy()
    predictions["indice_mapa"] = np.arange(len(predictions))
    positions = predictions["posicao"].to_numpy()
    classes = predictions["pred"].to_numpy()  # mapas da classe prevista pelo modelo treinado

    model_path = FINAL_DIR / "modelo_final.pt"
    reference = compute_maps(load_model(model_path, device), images, positions, classes, device)
    np.save(OUT_DIR / "mapas_teste.npy", reference.astype(np.float16))

    sample = choose_sample(predictions)
    columns = ["img_id", "patient_id", "diagnostic", "pred", "acerto", "confianca",
               "gender", "fitspatrick", "age", "biopsed"]
    listing = sample[columns].copy()
    listing["pred"] = listing["pred"].map(lambda value: treino.CLASS_NAMES[int(value)])
    listing.to_csv(OUT_DIR / "amostra_gradcam.csv", index=False)
    plot_sample(sample, images, reference, OUT_DIR / "gradcam_amostra.png")

    cascade = cascade_test(model_path, images, positions, classes, reference, device)
    cascade.to_csv(OUT_DIR / "sanidade_randomizacao_cascata.csv", index=False)
    epochs = args.epocas_rotulos_aleatorios or execution["epocas"]
    random_labels = random_label_test(metadata, partitions, images, position_by_id, positions,
                                      classes, reference, epochs, device)
    plot_sanity(cascade, random_labels, OUT_DIR / "sanidade_gradcam.png")
    summary = {"n_mapas": len(reference), "classe_do_mapa": "prevista pelo modelo treinado",
               "metrica": "correlação de Spearman por imagem (224 × 224)",
               "cascata": cascade.to_dict(orient="records"), "rotulos_embaralhados": random_labels}
    (OUT_DIR / "sanidade_gradcam.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
