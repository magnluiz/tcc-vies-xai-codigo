"""Executa um teste técnico curto do pipeline PAD-UFES-20, sem avaliar desempenho."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


ROOT = Path(__file__).resolve().parent.parent
DERIVED_DIR = ROOT / "dataset/PAD-UFES-20/derivados_analise_v1"
DEFAULT_METADATA = DERIVED_DIR / "coorte_principal/metadata_analise.csv"
DEFAULT_CV = DERIVED_DIR / "coorte_principal/particoes/pacientes_validacao_cruzada.csv"
DEFAULT_IMAGE_ROOT = ROOT / "dataset/PAD-UFES-20/zr7vgbcyr2-1/images"
DEFAULT_OUT = DERIVED_DIR / "smoke_test/model_pipeline_smoke.json"
CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]


def run_smoke_test(metadata_path: Path, cv_path: Path, image_root: Path,
                   result_path: Path) -> dict:
    torch.manual_seed(42)
    metadata = pd.read_csv(metadata_path)
    cv_patients = pd.read_csv(cv_path, dtype={"patient_id": str, "uso": str})
    training_patients = set(
        cv_patients.loc[
            (cv_patients["fold"] == 1) & (cv_patients["uso"] == "treino"),
            "patient_id",
        ]
    )
    training_rows = metadata[
        metadata["patient_id"].astype(str).isin(training_patients)
    ]

    image_by_id = {
        path.name: path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    selected = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        candidates = training_rows.loc[
            training_rows["diagnostic"] == class_name
        ].sort_values("img_id")
        if candidates.empty:
            raise ValueError(
                f"A classe {class_name} não aparece no fold de treino.")
        row = candidates.iloc[0]
        image_path = image_by_id.get(str(row["img_id"]))
        if image_path is None:
            raise FileNotFoundError(
                f"Imagem sem arquivo correspondente: {row['img_id']}")
        selected.append((str(row["img_id"]), class_index, image_path))

    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    labels = []
    for _, class_index, path in selected:
        with Image.open(path) as source_image:
            images.append(transform(source_image.convert("RGB")))
        labels.append(class_index)
    batch = torch.stack(images)
    target = torch.tensor(labels, dtype=torch.long)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4)
    before = model.fc.weight.detach().clone()
    optimizer.zero_grad()
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()

    weight_delta = float((model.fc.weight.detach() - before).abs().max())
    if tuple(batch.shape) != (len(CLASS_NAMES), 3, 224, 224):
        raise RuntimeError(
            f"Dimensões inesperadas do lote: {tuple(batch.shape)}")
    if tuple(logits.shape) != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise RuntimeError(
            f"Dimensões inesperadas dos logits: {tuple(logits.shape)}")
    if not torch.isfinite(loss) or weight_delta <= 0:
        raise RuntimeError(
            "A perda não é finita ou os pesos não foram atualizados.")

    result = {
        "status": "PASSOU",
        "finalidade": "teste técnico do pipeline; não é resultado experimental",
        "coorte": str(metadata_path.resolve().relative_to(ROOT)),
        "fold": 1,
        "pacientes_de_origem": "treino",
        "img_ids": [img_id for img_id, _, _ in selected],
        "classes": CLASS_NAMES,
        "formato_do_lote": list(batch.shape),
        "formato_dos_logits": list(logits.shape),
        "loss_finita": bool(torch.isfinite(loss)),
        "loss": float(loss.detach()),
        "max_delta_peso_camada_final": weight_delta,
        "pesos_pre_treinados": "ResNet18_Weights.IMAGENET1K_V1",
        "dispositivo": "cpu",
        "seed": 42,
        "torch": torch.__version__,
        "arquivo_resultado": str(result_path.resolve().relative_to(ROOT)),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(
        run_smoke_test(args.metadata, args.cv, args.image_root, args.out),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
