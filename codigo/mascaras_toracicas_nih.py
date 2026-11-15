"""Máscaras da região torácica ampliada para as imagens de teste da coorte NIH.

Segue a seção 7 da ficha de protocolo do NIH:
  - PSPNet do ChestX-Det (`torchxrayvision`), entrada 512 ampliada a partir do cache de 256;
  - união de pulmão esquerdo, pulmão direito, coração e mediastino (sigmoide > 0,5);
  - redução para 256, recorte central de 224, fecho convexo da união e dilatação de 10 pixels.
    O fecho convexo cobre as bases e os seios costofrênicos, que o PSPNet tende a excluir
    quando há derrame. A estrutura "Facies Diaphragmatica" foi descartada porque se estende
    pelo abdome até a borda da imagem (a região cobriria cerca de 82% do recorte);
  - máscara com menos de 15% do recorte = falha de segmentação (excluída da métrica de região).
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

ROOT = Path(__file__).resolve().parent.parent
NIH_DIR = ROOT / "dataset/NIH-ChestXray14"
CACHE_DIR = NIH_DIR / "cache_imagens_256"
COHORT = NIH_DIR / "derivados_v1/coorte_principal/coorte.csv"
OUT_DIR = NIH_DIR / "derivados_v1/mascaras_toracicas"
STRUCTURES = ["Left Lung", "Right Lung", "Heart", "Mediastinum"]
CROP = 224
DILATION = 10
MIN_AREA = 0.15
BATCH = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def segment(model, images: np.ndarray, device) -> np.ndarray:
    channels = [model.targets.index(name) for name in STRUCTURES]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATION + 1, 2 * DILATION + 1))
    offset = (256 - CROP) // 2
    masks = np.zeros((len(images), CROP, CROP), dtype=np.uint8)
    for start in range(0, len(images), BATCH):
        batch = torch.from_numpy(images[start:start + BATCH]).to(device).float()
        # Escala do torchxrayvision: [0, 255] → [-1024, 1024].
        x = (2 * batch / 255 - 1).mul_(1024).unsqueeze(1)
        x = F.interpolate(x, size=512, mode="bilinear", align_corners=False)
        probabilities = torch.sigmoid(model(x)[:, channels])
        union = (probabilities > 0.5).any(dim=1, keepdim=True).float()
        union = F.interpolate(union, size=256, mode="nearest")[:, 0].cpu().numpy().astype(np.uint8)
        for index, mask in enumerate(union):
            cropped = mask[offset:offset + CROP, offset:offset + CROP]
            contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue  # sem segmentação: área zero, contada como falha
            hull = np.zeros_like(cropped)
            cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(contours))], 1)
            masks[start + index] = cv2.dilate(hull, kernel)
    return masks


def plot_examples(images, masks, rows, path) -> None:
    offset = (256 - CROP) // 2
    picks = np.linspace(0, len(rows) - 1, 12).astype(int)
    figure, axes = plt.subplots(2, 6, figsize=(12, 4.4))
    for ax, index in zip(axes.flat, picks):
        image = images[index][offset:offset + CROP, offset:offset + CROP]
        ax.imshow(image, cmap="gray")
        ax.contour(masks[index], levels=[0.5], colors="#eb6834", linewidths=1)
        row = rows.iloc[index]
        ax.set_title(f"{'derrame' if row['rotulo'] else 'sem achados'} · {row['sexo']}", fontsize=8)
        ax.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    device = torch.device("cuda")
    cohort = pd.read_csv(COHORT)
    test = cohort[cohort["particao"] == "teste"].sort_values("img_id").reset_index(drop=True)
    index = pd.read_csv(CACHE_DIR / "indice.csv")
    position = dict(zip(index["Image Index"], index["posicao"]))
    cache = np.load(CACHE_DIR / "imagens_256.npy", mmap_mode="r")
    images = np.stack([cache[position[img_id]] for img_id in test["img_id"]])

    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    masks = segment(model, images, device)
    area = masks.reshape(len(masks), -1).mean(axis=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "mascaras_teste_224.npy", masks)
    table = test[["img_id", "rotulo", "sexo"]].assign(
        fracao_area=area, falha_segmentacao=area < MIN_AREA)
    table.to_csv(OUT_DIR / "mascaras_teste_224.csv", index=False)
    plot_examples(images, masks, test, OUT_DIR / "exemplos_mascaras.png")
    manifest = {
        "finalidade": "região torácica ampliada para a métrica de concentração (ficha NIH, seção 7)",
        "modelo": "torchxrayvision chestx_det.PSPNet", "torchxrayvision": xrv.__version__,
        "estruturas": STRUCTURES, "limiar_sigmoide": 0.5, "recorte": CROP, "fecho_convexo": True,
        "dilatacao_pixels": DILATION, "area_minima": MIN_AREA,
        "n_imagens": len(masks), "n_falhas": int((area < MIN_AREA).sum()),
        "fracao_area_mediana": float(np.median(area)),
        "fracao_area_por_rotulo": table.groupby("rotulo")["fracao_area"].median().to_dict(),
        "sha256_mascaras": sha256(OUT_DIR / "mascaras_teste_224.npy"),
    }
    (OUT_DIR / "manifesto_mascaras.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
