"""Marcas de caneta cirúrgica no PAD-UFES-20: dependência do modelo e sinal da XAI.

Usa a anotação visual (`marcas_caneta/anotacao_marcas_caneta.csv`). Para cada imagem com
marca de tinta azul/violeta detectável, a tinta é apagada por inpainting e mede-se a mudança
na predição (dependência real). O controle aplica a mesma máscara, na mesma posição, a uma
imagem sem marca. As predições são sempre fora da amostra: modelos por fold (receita final)
para o desenvolvimento e o modelo final para o teste.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import treinar_pad_ufes as treino
from xai_atribuicoes import GradCAM, integrated_gradients, mass_fraction


MARKS_DIR = treino.DERIVED_DIR / "marcas_caneta"
FOLD_MODELS_DIR = treino.RESULTS_DIR / "modelos_fold_receita_final"
FINAL_MODEL = treino.RESULTS_DIR / "final_principal/modelo_final.pt"
MIN_MASK_FRACTION = 0.002  # abaixo disso a tinta não foi captada pela regra de cor
IG_STEPS = 32
ACK = treino.CLASS_NAMES.index("ACK")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_REAL = "#2a78d6"
COLOR_PLACEBO = "#eb6834"


# Raio (pixels) do disco central excluído da máscara; 0 desativa. Definido em main().
CENTRAL_RADIUS = 0


def ink_mask(crop_rgb: np.ndarray) -> np.ndarray:
    """Pixels de tinta azul/violeta (mesma regra da triagem), dilatados em 2 pixels.

    Com CENTRAL_RADIUS > 0, ignora o disco central, onde fica a lesão, para que tons
    violáceos da própria lesão não sejam tratados como tinta.
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((hue >= 105) & (hue <= 165) & (saturation >= 70)
            & (value >= 40) & (value <= 230)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if CENTRAL_RADIUS > 0:
        rows, columns = np.ogrid[:mask.shape[0], :mask.shape[1]]
        center = (mask.shape[0] - 1) / 2
        mask[(rows - center) ** 2 + (columns - center) ** 2 <= CENTRAL_RADIUS ** 2] = 0
    return mask


def inpaint(crop_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(crop_rgb, mask, 3, cv2.INPAINT_TELEA)


def train_fold_models(metadata, cv_patients, images, position_by_id, device) -> None:
    decision = json.loads((treino.RESULTS_DIR / "cv_principal/decisao_selecao.json")
                          .read_text(encoding="utf-8"))
    schedule = decision["esquema_lr_treino_final"]
    FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for fold in sorted(cv_patients["fold"].unique()):
        path = FOLD_MODELS_DIR / f"fold_{fold}.pt"
        if path.exists():
            continue
        rows = cv_patients[(cv_patients["fold"] == fold) & (cv_patients["uso"] == "treino")]
        _, positions, labels = treino.rows_for(metadata, set(rows["patient_id"]), position_by_id)
        treino.set_seed(treino.SEED)
        model = treino.build_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=schedule[0],
                                      weight_decay=treino.WEIGHT_DECAY)
        scaler = torch.amp.GradScaler(device.type)
        generator = torch.Generator(device=device).manual_seed(treino.SEED + int(fold))
        for lr in schedule:
            for group in optimizer.param_groups:
                group["lr"] = lr
            treino.train_one_epoch(model, optimizer, scaler, images, positions, labels,
                                   None, generator)
        torch.save(model.state_dict(), path)
        print(f"modelo do fold {fold} treinado", flush=True)


def load(path: Path, device) -> torch.nn.Module:
    model = treino.build_model(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def model_for_each_image(annotations, cv_patients) -> pd.Series:
    """Caminho do modelo que não viu a imagem: fold de validação ou modelo final."""
    validation = cv_patients[cv_patients["uso"] == "validacao"].set_index("patient_id")["fold"]
    def choose(row):
        if row["particao"] == "teste":
            return str(FINAL_MODEL)
        return str(FOLD_MODELS_DIR / f"fold_{int(validation[row['patient_id']])}.pt")
    return annotations.apply(choose, axis=1)


def to_tensor(crops: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255).to(device)


@torch.no_grad()
def softmax(model, crops: torch.Tensor) -> np.ndarray:
    return torch.softmax(model(treino.normalize(crops)), dim=1).cpu().numpy()


def attributions(model, crops: torch.Tensor, classes: torch.Tensor, device) -> tuple:
    cam = GradCAM(model)
    gradcam = cam(treino.normalize(crops), classes).cpu().numpy()
    cam.remove()
    baseline = treino.normalize(torch.zeros(1, 3, treino.CROP, treino.CROP, device=device))[0]
    ig = integrated_gradients(model, treino.normalize(crops), classes, baseline, IG_STEPS)
    return gradcam, ig.cpu().numpy()


def run(device) -> pd.DataFrame:
    annotations = pd.read_csv(MARKS_DIR / "anotacao_marcas_caneta.csv")
    annotations["patient_id"] = annotations["patient_id"].astype(str)
    metadata, _, cv_patients = treino.load_cohort("principal")
    images, position_by_id = treino.load_cache(device)
    train_fold_models(metadata, cv_patients, images, position_by_id, device)
    annotations["modelo"] = model_for_each_image(annotations, cv_patients)
    annotations["label"] = annotations["diagnostic"].map(treino.CLASS_NAMES.index)

    def crop_of(img_id) -> np.ndarray:
        tensor = treino.center_crop(images[position_by_id[img_id]])
        return tensor.permute(1, 2, 0).cpu().numpy().copy()

    marked = annotations[annotations["marca_caneta"] == "sim"].copy()
    unmarked = annotations[annotations["marca_caneta"] == "nao"]
    rng = np.random.default_rng(treino.SEED)
    rows = []
    for model_path, group in marked.groupby("modelo"):
        model = load(Path(model_path), device)
        # Controles vêm de imagens sem marca avaliadas pelo mesmo modelo (também fora da amostra).
        controls = unmarked[unmarked["modelo"] == model_path]
        for _, record in group.iterrows():
            original = crop_of(record["img_id"])
            mask = ink_mask(original)
            fraction = float(mask.mean())
            row = {"img_id": record["img_id"], "patient_id": record["patient_id"],
                   "diagnostic": record["diagnostic"], "particao": record["particao"],
                   "fracao_mascara": fraction, "tinta_detectada": fraction >= MIN_MASK_FRACTION}
            if fraction < MIN_MASK_FRACTION:
                rows.append(row)
                continue
            control = controls.iloc[int(rng.integers(len(controls)))]
            control_crop = crop_of(control["img_id"])
            batch = np.stack([original, inpaint(original, mask),
                              control_crop, inpaint(control_crop, mask)])
            probabilities = softmax(model, to_tensor(batch, device))
            predicted = int(probabilities[0].argmax())
            control_predicted = int(probabilities[2].argmax())
            gradcam, ig = attributions(model, to_tensor(batch[:1], device),
                                       torch.tensor([predicted], device=device), device)
            masks = mask[None].astype(np.float32)
            row.update({
                "pred_original": treino.CLASS_NAMES[predicted],
                "pred_sem_tinta": treino.CLASS_NAMES[int(probabilities[1].argmax())],
                "mudou_predicao": bool(probabilities[1].argmax() != predicted),
                "queda_prob_prevista": float(probabilities[0, predicted] - probabilities[1, predicted]),
                "delta_prob_ack": float(probabilities[0, ACK] - probabilities[1, ACK]),
                "controle_img_id": control["img_id"],
                "controle_mudou_predicao": bool(probabilities[3].argmax() != control_predicted),
                "controle_queda_prob_prevista": float(probabilities[2, control_predicted]
                                                      - probabilities[3, control_predicted]),
                "fracao_gradcam_tinta": float(mass_fraction(gradcam, masks)[0]),
                "fracao_ig_tinta": float(mass_fraction(ig, masks)[0]),
            })
            rows.append(row)
        print(f"{Path(model_path).name}: {len(group)} imagens com marca", flush=True)
    return pd.DataFrame(rows)


def prediction_association(device) -> dict:
    """Entre lesões que não são ACK, a marca aumenta a chance de o modelo prever ACK?"""
    annotations = pd.read_csv(MARKS_DIR / "anotacao_marcas_caneta.csv")
    oof = pd.read_csv(treino.RESULTS_DIR / "cv_principal/predicoes_fora_do_fold.csv")
    oof = oof[oof["config"] == "ce_padrao"][["img_id", "pred"]]
    test = pd.read_csv(treino.RESULTS_DIR / "final_principal/predicoes_teste.csv")[["img_id", "pred"]]
    data = annotations.merge(pd.concat([oof, test]), on="img_id")
    data = data[(data["diagnostic"] != "ACK") & (data["marca_caneta"] != "duvida")]
    data["previu_ack"] = data["pred"] == ACK
    result = {}
    for diagnosis, rows in [("todas_nao_ACK", data)] + list(data.groupby("diagnostic")):
        by_mark = rows.groupby("marca_caneta")["previu_ack"].agg(["mean", "size"])
        result[diagnosis] = {mark: {"taxa_previu_ack": float(values["mean"]), "n": int(values["size"])}
                             for mark, values in by_mark.iterrows()}
    return result


def detected_rows(results: pd.DataFrame) -> pd.DataFrame:
    detected = results[results["tinta_detectada"].astype(bool)].copy()
    for column in ("mudou_predicao", "controle_mudou_predicao"):
        detected[column] = detected[column].astype(bool)
    return detected


def summarize(results: pd.DataFrame) -> dict:
    detected = detected_rows(results)
    summary = {
        "n_imagens_com_marca": len(results),
        "n_tinta_detectada_pela_regra_de_cor": len(detected),
        "fracao_mediana_da_mascara": float(detected["fracao_mascara"].median()),
        "dependencia_real": {
            "taxa_mudou_predicao": float(detected["mudou_predicao"].mean()),
            "queda_media_prob_prevista": float(detected["queda_prob_prevista"].mean()),
            "delta_medio_prob_ack": float(detected["delta_prob_ack"].mean()),
        },
        "controle_placebo": {
            "taxa_mudou_predicao": float(detected["controle_mudou_predicao"].mean()),
            "queda_media_prob_prevista": float(detected["controle_queda_prob_prevista"].mean()),
        },
    }
    detected["concentracao_gradcam"] = detected["fracao_gradcam_tinta"] / detected["fracao_mascara"]
    detected["concentracao_ig"] = detected["fracao_ig_tinta"] / detected["fracao_mascara"]
    for method in ("gradcam", "ig"):
        column = f"concentracao_{method}"
        summary[method] = {
            "concentracao_mediana": float(detected[column].median()),
            "spearman_com_queda_prob": float(spearmanr(detected[column],
                                                       detected["queda_prob_prevista"]).statistic),
            "auc_identifica_mudou_predicao": float(roc_auc_score(detected["mudou_predicao"],
                                                                 detected[column]))
            if detected["mudou_predicao"].nunique() == 2 else None,
        }
    summary["ic95_bootstrap_pacientes"] = bootstrap_intervals(detected)
    summary["raio_central_excluido"] = CENTRAL_RADIUS
    return summary


def bootstrap_intervals(detected: pd.DataFrame, replicates: int = 2000) -> dict:
    """IC 95% (percentis) reamostrando pacientes, com as imagens de cada um juntas."""
    groups = detected.groupby("patient_id").indices
    patients = list(groups)
    rng = np.random.default_rng(treino.SEED)
    values = {key: [] for key in ("diferenca_taxa_mudou_vs_controle", "diferenca_queda_vs_controle",
                                  "spearman_gradcam", "spearman_ig", "auc_gradcam", "auc_ig")}
    for _ in range(replicates):
        sample = detected.iloc[np.concatenate([groups[p] for p in rng.choice(patients, len(patients))])]
        values["diferenca_taxa_mudou_vs_controle"].append(
            sample["mudou_predicao"].mean() - sample["controle_mudou_predicao"].mean())
        values["diferenca_queda_vs_controle"].append(
            (sample["queda_prob_prevista"] - sample["controle_queda_prob_prevista"]).mean())
        for method in ("gradcam", "ig"):
            column = f"concentracao_{method}"
            values[f"spearman_{method}"].append(
                spearmanr(sample[column], sample["queda_prob_prevista"]).statistic)
            if sample["mudou_predicao"].nunique() == 2:
                values[f"auc_{method}"].append(roc_auc_score(sample["mudou_predicao"], sample[column]))
    return {key: [float(v) for v in np.percentile(series, [2.5, 50, 97.5])]
            for key, series in values.items() if series}


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot(results: pd.DataFrame, path: Path) -> None:
    detected = detected_rows(results)
    detected["concentracao_gradcam"] = detected["fracao_gradcam_tinta"] / detected["fracao_mascara"]
    detected["concentracao_ig"] = detected["fracao_ig_tinta"] / detected["fracao_mascara"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for position, (column, color, label) in enumerate(
            [("queda_prob_prevista", COLOR_REAL, "tinta apagada"),
             ("controle_queda_prob_prevista", COLOR_PLACEBO, "controle (sem marca)")]):
        values = detected[column].to_numpy()
        left.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=10,
                     color=color, alpha=0.35, linewidths=0)
        left.plot([position - 0.28, position + 0.28], [values.mean()] * 2, color=INK, linewidth=2)
        left.annotate(f"média {values.mean():.3f}", (position + 0.3, values.mean()), xytext=(4, 6),
                      textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    left.set_xticks([0, 1], ["tinta apagada", "controle\n(mesma máscara, sem marca)"])
    left.axhline(0, color=BASELINE, linewidth=0.8)
    left.set_ylabel("Queda na probabilidade\nda classe prevista", color=INK_SECONDARY)
    left.set_title("Dependência real das marcas", fontsize=10, color=INK, loc="left")

    for position, (column, color, label) in enumerate(
            [("concentracao_gradcam", COLOR_REAL, "Grad-CAM"), ("concentracao_ig", COLOR_PLACEBO, "IG")]):
        for offset, changed in ((-0.22, False), (0.22, True)):
            values = detected.loc[detected["mudou_predicao"] == changed, column].to_numpy()
            right.scatter(position + offset + rng.uniform(-0.08, 0.08, len(values)), values, s=10,
                          color=color, alpha=0.35 if changed else 0.18, linewidths=0)
            if len(values):
                right.plot([position + offset - 0.12, position + offset + 0.12],
                           [np.median(values)] * 2, color=INK, linewidth=2)
    right.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    right.annotate("proporcional à área", (1.5, 1), xytext=(0, 4), textcoords="offset points",
                   fontsize=8, color=MUTED, ha="right")
    right.set_xticks([-0.22, 0.22, 0.78, 1.22], ["manteve", "mudou", "manteve", "mudou"])
    for position, label in ((0, "Grad-CAM"), (1, "IG")):
        right.annotate(label, (position, 0), xycoords=("data", "axes fraction"), xytext=(0, -30),
                       textcoords="offset points", ha="center", fontsize=9, color=INK)
    right.set_xlim(-0.6, 1.6)
    right.set_yscale("log")
    right.set_ylabel("Atribuição na tinta ÷ área da tinta", color=INK_SECONDARY)
    right.set_title("Sinal da explicação, por efeito de apagar", fontsize=10, color=INK, loc="left")
    for ax in (left, right):
        style_axes(ax)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def plot_examples(results: pd.DataFrame, images_path: Path, device, path: Path) -> None:
    images, position_by_id = treino.load_cache(device)
    detected = detected_rows(results).sort_values("queda_prob_prevista", ascending=False)
    chosen = pd.concat([detected.head(3), detected.tail(3)])
    figure, axes = plt.subplots(len(chosen), 3, figsize=(6.0, 2.0 * len(chosen)), facecolor=SURFACE)
    for row, (_, record) in enumerate(chosen.iterrows()):
        crop = treino.center_crop(images[position_by_id[record["img_id"]]]).permute(1, 2, 0).cpu().numpy().copy()
        mask = ink_mask(crop)
        overlay = crop.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([235, 104, 52])).astype(np.uint8)
        for column, (picture, title) in enumerate([
                (crop, f"{record['diagnostic']} → {record['pred_original']}"),
                (overlay, "máscara da tinta"),
                (inpaint(crop, mask), f"sem tinta → {record['pred_sem_tinta']} "
                                      f"(queda {record['queda_prob_prevista']:.2f})")]):
            axes[row, column].imshow(picture)
            axes[row, column].set_title(title, fontsize=7.5, color=INK, pad=3)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--so-analise", action="store_true",
                        help="Reaproveita por_imagem.csv e refaz apenas resumo e figuras.")
    parser.add_argument("--raio-central", type=int, default=0,
                        help="Exclui da máscara de tinta um disco central com este raio (pixels).")
    args = parser.parse_args()
    global CENTRAL_RADIUS
    CENTRAL_RADIUS = args.raio_central
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    out_dir = MARKS_DIR / ("experimento" if CENTRAL_RADIUS == 0
                           else f"experimento_fora_do_centro_r{CENTRAL_RADIUS}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.so_analise:
        results = pd.read_csv(out_dir / "por_imagem.csv")
    else:
        results = run(device)
        results.to_csv(out_dir / "por_imagem.csv", index=False)
    summary = summarize(results)
    summary["associacao_marca_previsao_ack"] = prediction_association(device)
    (out_dir / "resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    plot(results, out_dir / "marcas_caneta.png")
    plot_examples(results, None, device, out_dir / "exemplos_inpainting.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
