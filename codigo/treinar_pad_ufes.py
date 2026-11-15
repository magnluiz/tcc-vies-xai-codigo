"""Treina a ResNet-18 no PAD-UFES-20 conforme a ficha de protocolo experimental.

Fases:
  cv     validação cruzada agrupada (5 folds) com entropia cruzada padrão e ponderada;
  final  treino no desenvolvimento inteiro com a configuração escolhida e predição
         do teste, uma única vez por coorte.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parent.parent
DERIVED_DIR = ROOT / "dataset/PAD-UFES-20/derivados_analise_v1"
CACHE_DIR = DERIVED_DIR / "cache_imagens_256"
RESULTS_DIR = DERIVED_DIR / "experimentos_v1"
COHORTS = {
    "principal": DERIVED_DIR / "coorte_principal",
    "sensibilidade": DERIVED_DIR / "coorte_sensibilidade",
}
CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CONFIGS = ("ce_padrao", "ce_ponderada")
SEED = 42
CROP = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
LR_PATIENCE = 2
STOP_PATIENCE = 5
MAX_ROTATION_DEGREES = 15.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cohort(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_dir = COHORTS[name]
    metadata = pd.read_csv(cohort_dir / "metadata_analise.csv")
    partitions = pd.read_csv(
        cohort_dir / "particoes/particoes_pacientes.csv", dtype=str)
    cv_patients = pd.read_csv(
        cohort_dir / "particoes/pacientes_validacao_cruzada.csv",
        dtype={"patient_id": str, "uso": str},
    )
    metadata["patient_id"] = metadata["patient_id"].astype(str)
    metadata["label"] = metadata["diagnostic"].map(
        {name: index for index, name in enumerate(CLASS_NAMES)})
    if metadata["label"].isna().any():
        raise ValueError("Há diagnósticos fora das seis classes esperadas.")
    metadata["label"] = metadata["label"].astype(int)
    return metadata, partitions, cv_patients


def load_cache(device: torch.device) -> tuple[torch.Tensor, dict]:
    images = np.load(CACHE_DIR / "imagens_256.npy")
    index = pd.read_csv(CACHE_DIR / "indice.csv")
    position_by_id = dict(zip(index["img_id"], index["posicao"]))
    # Todas as imagens (≈ 450 MB em uint8) ficam na GPU; a conversão para float é feita por lote.
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
    return tensor, position_by_id


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Espelhamentos e rotação sorteados por imagem, seguidos do recorte central."""
    n = batch.shape[0]
    device = batch.device
    angles = (torch.rand(n, generator=generator, device=device) * 2 - 1) \
        * math.radians(MAX_ROTATION_DEGREES)
    flip_x = torch.where(torch.rand(n, generator=generator, device=device) < 0.5, -1.0, 1.0)
    flip_y = torch.where(torch.rand(n, generator=generator, device=device) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = cos * flip_x
    theta[:, 0, 1] = -sin * flip_y
    theta[:, 1, 0] = sin * flip_x
    theta[:, 1, 1] = cos * flip_y
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    rotated = F.grid_sample(batch, grid, mode="bilinear",
                            padding_mode="reflection", align_corners=False)
    return center_crop(rotated)


def center_crop(batch: torch.Tensor) -> torch.Tensor:
    offset = (batch.shape[-1] - CROP) // 2
    return batch[..., offset:offset + CROP, offset:offset + CROP]


def to_input(batch_uint8: torch.Tensor) -> torch.Tensor:
    return batch_uint8.float().div_(255)


def normalize(batch: torch.Tensor) -> torch.Tensor:
    return (batch - MEAN.to(batch.device)) / STD.to(batch.device)


def build_model(device: torch.device) -> torch.nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.to(device)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    if (counts == 0).any():
        raise ValueError("Uma classe não aparece no treino; pesos indefinidos.")
    inverse = 1.0 / counts
    return torch.tensor(inverse / inverse.mean(), dtype=torch.float32)


def train_one_epoch(model, optimizer, scaler, images, positions, labels,
                    weights, generator) -> float:
    model.train()
    device = images.device
    order = torch.randperm(len(positions), generator=generator, device=device)
    positions_t = torch.as_tensor(positions, device=device)[order]
    labels_t = torch.as_tensor(np.asarray(labels, dtype=np.int64).copy(), device=device)[order]
    total_loss, total = 0.0, 0
    for start in range(0, len(positions_t), BATCH_SIZE):
        batch_positions = positions_t[start:start + BATCH_SIZE]
        if len(batch_positions) < 2:
            continue  # BatchNorm não aceita lote de uma imagem.
        target = labels_t[start:start + BATCH_SIZE]
        inputs = normalize(augment(to_input(images[batch_positions]), generator))
        with torch.autocast(device.type, dtype=torch.float16):
            logits = model(inputs)
        loss = F.cross_entropy(logits.float(), target, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(target)
        total += len(target)
    return total_loss / total


@torch.no_grad()
def predict(model, images, positions) -> np.ndarray:
    model.eval()
    device = images.device
    positions_t = torch.as_tensor(positions, device=device)
    outputs = []
    for start in range(0, len(positions_t), 128):
        inputs = normalize(center_crop(to_input(images[positions_t[start:start + 128]])))
        with torch.autocast(device.type, dtype=torch.float16):
            outputs.append(model(inputs).float())
    return torch.cat(outputs).cpu().numpy()


def evaluate(logits: np.ndarray, labels: np.ndarray) -> dict:
    predictions = logits.argmax(axis=1)
    # A perda de validação é sempre a entropia cruzada sem pesos, para ser comparável entre configurações.
    loss = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels.astype(np.int64))))
    return {
        "perda": loss,
        "macro_f1": float(f1_score(labels, predictions, labels=range(len(CLASS_NAMES)),
                                   average="macro", zero_division=0)),
        "acuracia_balanceada": float(balanced_accuracy_score(labels, predictions)),
        "acuracia": float((predictions == labels).mean()),
    }


def is_better(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if candidate["macro_f1"] != best["macro_f1"]:
        return candidate["macro_f1"] > best["macro_f1"]
    return candidate["perda"] < best["perda"]


def rows_for(metadata: pd.DataFrame, patient_ids: set, position_by_id: dict):
    rows = metadata[metadata["patient_id"].isin(patient_ids)].sort_values("img_id")
    positions = np.array([position_by_id[img_id] for img_id in rows["img_id"]])
    return rows, positions, rows["label"].to_numpy()


def run_cv_fold(config, fold, metadata, cv_patients, images, position_by_id, device):
    fold_rows = cv_patients[cv_patients["fold"] == fold]
    train_ids = set(fold_rows.loc[fold_rows["uso"] == "treino", "patient_id"])
    validation_ids = set(fold_rows.loc[fold_rows["uso"] == "validacao", "patient_id"])
    if train_ids & validation_ids:
        raise RuntimeError(f"Fold {fold}: pacientes em treino e validação.")
    _, train_positions, train_labels = rows_for(metadata, train_ids, position_by_id)
    validation_rows, validation_positions, validation_labels = rows_for(
        metadata, validation_ids, position_by_id)

    set_seed(SEED)
    model = build_model(device)
    weights = class_weights(train_labels).to(device) if config == "ce_ponderada" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=LR_PATIENCE)
    scaler = torch.amp.GradScaler(device.type)
    generator = torch.Generator(device=device).manual_seed(SEED + fold)

    history, best, best_logits, epochs_without_improvement = [], None, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        started = time.time()
        train_loss = train_one_epoch(model, optimizer, scaler, images, train_positions,
                                     train_labels, weights, generator)
        logits = predict(model, images, validation_positions)
        metrics = evaluate(logits, validation_labels)
        history.append({"config": config, "fold": fold, "epoca": epoch, "lr": lr,
                        "perda_treino": train_loss, **{f"val_{k}": v for k, v in metrics.items()},
                        "segundos": time.time() - started})
        if is_better(metrics, best):
            best = {**metrics, "epoca": epoch}
            best_logits = logits
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(metrics["macro_f1"])
        if epochs_without_improvement >= STOP_PATIENCE:
            break

    oof = validation_rows[["img_id", "patient_id", "diagnostic", "label"]].copy()
    oof.insert(0, "fold", fold)
    oof.insert(0, "config", config)
    oof["pred"] = best_logits.argmax(axis=1)
    for index, name in enumerate(CLASS_NAMES):
        oof[f"logit_{name}"] = best_logits[:, index]
    summary = {"config": config, "fold": fold, "n_imagens_treino": len(train_positions),
               "n_imagens_validacao": len(validation_positions),
               "pesos_classe": weights.tolist() if weights is not None else None,
               "epocas_executadas": len(history), "melhor": best}
    return history, oof, summary


def choose_config(summaries: pd.DataFrame) -> dict:
    stats = summaries.groupby("config")["macro_f1"].agg(["mean", "std"])
    chosen = stats["mean"].idxmax()
    other = stats.drop(index=chosen)
    difference = float(stats.loc[chosen, "mean"] - other["mean"].max())
    return {
        "config_escolhida": chosen,
        "macro_f1_medio_por_config": stats["mean"].to_dict(),
        "desvio_padrao_por_config": stats["std"].to_dict(),
        "diferenca_entre_medias": difference,
        "escolha_robusta": bool(difference >= float(stats["std"].max())),
    }


def phase_cv(device: torch.device) -> None:
    out_dir = RESULTS_DIR / "cv_principal"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata, partitions, cv_patients = load_cohort("principal")
    test_ids = set(partitions.loc[partitions["particao"] == "teste", "patient_id"])
    if set(cv_patients["patient_id"]) & test_ids:
        raise RuntimeError("Pacientes de teste aparecem na validação cruzada.")
    images, position_by_id = load_cache(device)

    histories, oofs, summaries = [], [], []
    for config in CONFIGS:
        for fold in sorted(cv_patients["fold"].unique()):
            history, oof, summary = run_cv_fold(config, int(fold), metadata, cv_patients,
                                                images, position_by_id, device)
            best = summary["melhor"]
            print(f"{config} fold {fold}: melhor época {best['epoca']}, "
                  f"macro-F1 {best['macro_f1']:.3f}, {summary['epocas_executadas']} épocas")
            histories.extend(history)
            oofs.append(oof)
            summaries.append(summary)

    pd.DataFrame(histories).to_csv(out_dir / "historico_epocas.csv", index=False)
    pd.concat(oofs).to_csv(out_dir / "predicoes_fora_do_fold.csv", index=False)
    flat = pd.DataFrame([{"config": s["config"], "fold": s["fold"], **s["melhor"]}
                         for s in summaries])
    flat.to_csv(out_dir / "resumo_folds.csv", index=False)
    decision = choose_config(flat)
    chosen = flat[flat["config"] == decision["config_escolhida"]]
    decision["epocas_treino_final"] = int(math.ceil(chosen["epoca"].median()))
    decision["esquema_lr_treino_final"] = median_lr_schedule(
        pd.DataFrame(histories), decision["config_escolhida"], decision["epocas_treino_final"])
    (out_dir / "decisao_selecao.json").write_text(
        json.dumps({"folds": summaries, **decision, **environment(device)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def median_lr_schedule(history: pd.DataFrame, config: str, epochs: int) -> list[float]:
    rows = history[history["config"] == config]
    schedule = []
    for epoch in range(1, epochs + 1):
        per_fold = []
        for _, fold_rows in rows.groupby("fold"):
            available = fold_rows[fold_rows["epoca"] <= epoch]
            per_fold.append(float(available.sort_values("epoca")["lr"].iloc[-1]))
        schedule.append(float(np.median(per_fold)))
    return schedule


def phase_final(cohort: str, device: torch.device, force: bool) -> None:
    decision_path = RESULTS_DIR / "cv_principal/decisao_selecao.json"
    if not decision_path.exists():
        raise RuntimeError("Execute a fase cv antes da fase final.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    out_dir = RESULTS_DIR / f"final_{cohort}"
    predictions_path = out_dir / "predicoes_teste.csv"
    if predictions_path.exists() and not force:
        raise RuntimeError(
            f"O teste da coorte {cohort} já foi avaliado ({predictions_path}). "
            "O protocolo prevê uma única avaliação; use --forcar só com justificativa registrada.")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata, partitions, _ = load_cohort(cohort)
    development_ids = set(partitions.loc[partitions["particao"] == "desenvolvimento", "patient_id"])
    test_ids = set(partitions.loc[partitions["particao"] == "teste", "patient_id"])
    if development_ids & test_ids:
        raise RuntimeError("Pacientes compartilhados entre desenvolvimento e teste.")
    images, position_by_id = load_cache(device)
    _, train_positions, train_labels = rows_for(metadata, development_ids, position_by_id)
    test_rows, test_positions, test_labels = rows_for(metadata, test_ids, position_by_id)

    config = decision["config_escolhida"]
    schedule = decision["esquema_lr_treino_final"]
    set_seed(SEED)
    model = build_model(device)
    weights = class_weights(train_labels).to(device) if config == "ce_ponderada" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule[0], weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type)
    generator = torch.Generator(device=device).manual_seed(SEED)
    history = []
    for epoch, lr in enumerate(schedule, start=1):
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = train_one_epoch(model, optimizer, scaler, images, train_positions,
                               train_labels, weights, generator)
        history.append({"epoca": epoch, "lr": lr, "perda_treino": loss})
        print(f"final {cohort} época {epoch}/{len(schedule)}: perda {loss:.4f}, lr {lr:g}")

    torch.save(model.state_dict(), out_dir / "modelo_final.pt")
    logits = predict(model, images, test_positions)
    predictions = test_rows.copy()
    predictions["label"] = test_labels
    predictions["pred"] = logits.argmax(axis=1)
    for index, name in enumerate(CLASS_NAMES):
        predictions[f"logit_{name}"] = logits[:, index]
    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame(history).to_csv(out_dir / "historico_treino_final.csv", index=False)
    (out_dir / "execucao_final.json").write_text(json.dumps({
        "coorte": cohort, "config": config, "epocas": len(schedule), "esquema_lr": schedule,
        "pesos_classe": weights.tolist() if weights is not None else None,
        "n_imagens_desenvolvimento": len(train_positions), "n_imagens_teste": len(test_positions),
        "n_pacientes_teste": len(test_ids), "forcado": force, **environment(device),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Predições do teste gravadas em {predictions_path}")


def environment(device: torch.device) -> dict:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dispositivo": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "semente": SEED,
        "precisao_mista": True,
        "cudnn_deterministico": torch.backends.cudnn.deterministic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fase", choices=["cv", "final"])
    parser.add_argument("--coorte", choices=sorted(COHORTS), default="principal")
    parser.add_argument("--forcar", action="store_true",
                        help="Permite refazer a predição do teste (registrado no JSON).")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA indisponível; o protocolo prevê execução na GPU.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    if args.fase == "cv":
        phase_cv(device)
    else:
        phase_final(args.coorte, device, args.forcar)


if __name__ == "__main__":
    main()
