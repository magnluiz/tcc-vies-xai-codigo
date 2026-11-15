"""Treinos do NIH ChestX-ray14 conforme a ficha de protocolo congelada (Dia 10).

Fases:
  piloto        q = 0,5, semente 42, até 20 épocas; fixa o número de épocas E pela AUC na
                validação (o teste não é usado);
  experimentos  experimento A (sexo correlacionado ao derrame) e B (quadrado xadrez
                correlacionado ao derrame), 4 níveis de q × 5 sementes cada, mais o modelo
                com rótulos embaralhados do teste de sanidade. Cada execução grava o modelo
                e as predições do teste; execuções já gravadas são puladas;
  replica       os mesmos 40 treinos com as sementes 47 a 51 (réplica confirmatória
                pré-registrada, ficha, ajuste 2), sem novo modelo de rótulos embaralhados.
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
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

import treinar_pad_ufes as pad


ROOT = Path(__file__).resolve().parent.parent
NIH_DIR = ROOT / "dataset/NIH-ChestXray14"
CACHE_DIR = NIH_DIR / "cache_imagens_256"
COHORTS = {
    "principal": NIH_DIR / "derivados_v1/coorte_principal/coorte.csv",
    "derrame_isolado": NIH_DIR / "derivados_v1/coorte_derrame_isolado/coorte.csv",
}
RESULTS_DIR = NIH_DIR / "experimentos_v1"
SEED = 42
LEVELS = [0.5, 0.75, 0.9, 1.0]
N_SEEDS = 5
# Réplica confirmatória pré-registrada (ficha, ajuste 2; Dia 12): sementes novas, mesma receita.
SEED_SETS = {"original": range(SEED, SEED + N_SEEDS), "replica": range(47, 52)}
N_POSITIVE = 1200
N_NEGATIVE = 2400
PILOT_MAX_EPOCHS = 20
CROP = 224
MAX_SHIFT = 16  # recorte aleatório de 224 dentro de 256
MAX_ROTATION_DEGREES = 10.0
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATCH = 20
PATCH_CELL = 4
CORNER_MARGIN = (8, 40)


class Data:
    """Imagens das coortes na GPU (uint8, 1 canal) e as tabelas de cada partição."""

    def __init__(self, device: torch.device):
        main = pd.read_csv(COHORTS["principal"])
        isolated = pd.read_csv(COHORTS["derrame_isolado"])
        self.train = main[main["particao"] == "treino"].sort_values("img_id").reset_index(drop=True)
        self.validation = main[main["particao"] == "validacao"].sort_values("img_id").reset_index(drop=True)
        self.test = main[main["particao"] == "teste"].sort_values("img_id").reset_index(drop=True)
        self.test_isolated = isolated[isolated["particao"] == "teste"].sort_values("img_id").reset_index(drop=True)
        for left, right in ((self.train, self.validation), (self.train, self.test),
                            (self.validation, self.test), (self.train, self.test_isolated)):
            if set(left["patient_id"]) & set(right["patient_id"]):
                raise RuntimeError("Paciente compartilhado entre partições.")

        index = pd.read_csv(CACHE_DIR / "indice.csv")
        cache_position = dict(zip(index["Image Index"], index["posicao"]))
        ids = sorted(set(self.train["img_id"]) | set(self.validation["img_id"])
                     | set(self.test["img_id"]) | set(self.test_isolated["img_id"]))
        self.row = {img_id: row for row, img_id in enumerate(ids)}
        cache = np.load(CACHE_DIR / "imagens_256.npy", mmap_mode="r")
        positions = np.array([cache_position[img_id] for img_id in ids])
        order = np.argsort(positions)  # leitura sequencial do memmap
        images = np.empty((len(ids), 256, 256), dtype=np.uint8)
        images[order] = cache[positions[order]]
        self.images = torch.from_numpy(images).unsqueeze(1).to(device)

    def rows(self, table: pd.DataFrame) -> np.ndarray:
        return np.array([self.row[img_id] for img_id in table["img_id"]])


def to_input(batch_uint8: torch.Tensor) -> torch.Tensor:
    """Cinza em [0, 1] replicado nos três canais."""
    return batch_uint8.float().div_(255).expand(-1, 3, -1, -1)


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Rotação de até 10 graus e recorte aleatório de 224, sem espelhamento."""
    n, device = batch.shape[0], batch.device
    angles = (torch.rand(n, generator=generator, device=device) * 2 - 1) \
        * math.radians(MAX_ROTATION_DEGREES)
    shifts = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (n, 2), generator=generator, device=device)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = torch.cos(angles)
    theta[:, 0, 1] = -torch.sin(angles)
    theta[:, 1, 0] = torch.sin(angles)
    theta[:, 1, 1] = torch.cos(angles)
    theta[:, :, 2] = shifts.float() * 2 / batch.shape[-1]
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    moved = F.grid_sample(batch, grid, mode="bilinear", padding_mode="reflection",
                          align_corners=False)
    return pad.center_crop(moved)


def checkerboard(device) -> torch.Tensor:
    cells = torch.arange(PATCH, device=device) // PATCH_CELL
    return ((cells.view(-1, 1) + cells.view(1, -1)) % 2).float()


def sample_corners(n: int, generator: torch.Generator, device) -> torch.Tensor:
    """Canto superior esquerdo (linha, coluna) do quadrado, num dos quatro cantos do recorte."""
    low, high = CORNER_MARGIN
    offset = torch.randint(low, high + 1, (n, 2), generator=generator, device=device)
    far = torch.rand(n, 2, generator=generator, device=device) < 0.5
    return torch.where(far, CROP - PATCH - offset, offset)


def paste(batch: torch.Tensor, corners: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Insere o quadrado xadrez nas imagens selecionadas (lote em [0, 1], já recortado)."""
    batch = batch.clone()
    pattern = checkerboard(batch.device).to(batch.dtype)
    for index in torch.nonzero(selected).flatten().tolist():
        row, column = corners[index].tolist()
        batch[index, :, row:row + PATCH, column:column + PATCH] = pattern
    return batch


def build_model(device: torch.device) -> torch.nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def draw(table: pd.DataFrame, counts: dict, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for (label, sex), n in counts.items():
        cell = table[(table["rotulo"] == label) & (table["sexo"] == sex)]
        if n > len(cell):
            raise ValueError(f"Célula rótulo={label} sexo={sex}: pede {n}, há {len(cell)}.")
        parts.append(cell.iloc[np.sort(rng.choice(len(cell), n, replace=False))])
    return pd.concat(parts).sort_values("img_id").reset_index(drop=True)


def training_set(experiment: str, q: float, seed: int, train: pd.DataFrame) -> pd.DataFrame:
    """Treino da execução, com contagens exatas por célula (ficha, seções 3 e 4)."""
    rng = np.random.default_rng(seed)
    sex_q = q if experiment == "A" else 0.5
    male_positive = round(N_POSITIVE * sex_q)
    female_negative = round(N_NEGATIVE * sex_q)
    chosen = draw(train, {(1, "M"): male_positive, (1, "F"): N_POSITIVE - male_positive,
                          (0, "F"): female_negative, (0, "M"): N_NEGATIVE - female_negative}, rng)
    chosen["quadrado"] = False
    if experiment == "B":
        for label, share in ((1, q), (0, 1 - q)):
            members = np.flatnonzero(chosen["rotulo"] == label)
            picked = rng.choice(members, round(len(members) * share), replace=False)
            chosen.loc[picked, "quadrado"] = True
    return chosen


def train(model, data: Data, chosen: pd.DataFrame, labels: np.ndarray, epochs: int, seed: int,
          device, on_epoch=None) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type)
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = torch.as_tensor(data.rows(chosen), device=device)
    targets = torch.as_tensor(labels.astype(np.int64), device=device)
    with_patch = torch.as_tensor(chosen["quadrado"].to_numpy().copy(), device=device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started, total_loss = time.time(), 0.0
        order = torch.randperm(len(rows), generator=generator, device=device)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            if len(batch) < 2:
                continue  # BatchNorm não aceita lote de uma imagem.
            crops = augment(to_input(data.images[rows[batch]]), generator)
            if with_patch.any():
                crops = paste(crops, sample_corners(len(batch), generator, device), with_patch[batch])
            with torch.autocast(device.type, dtype=torch.float16):
                logits = model(pad.normalize(crops))
            loss = F.cross_entropy(logits.float(), targets[batch])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(batch)
        record = {"epoca": epoch, "perda_treino": total_loss / len(rows),
                  "segundos": time.time() - started}
        if on_epoch is not None:
            record.update(on_epoch(model))
        history.append(record)
    return history


@torch.no_grad()
def probability(model, crops_fn, n: int) -> np.ndarray:
    """P(derrame) para n imagens; crops_fn(início, fim) devolve o lote recortado em [0, 1]."""
    model.eval()
    outputs = []
    for start in range(0, n, 128):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(pad.normalize(crops_fn(start, min(start + 128, n)))).float()
        outputs.append(torch.softmax(logits, dim=1)[:, 1])
    return torch.cat(outputs).cpu().numpy()


def clean_probability(model, data: Data, table: pd.DataFrame) -> np.ndarray:
    rows = torch.as_tensor(data.rows(table), device=data.images.device)
    return probability(model, lambda a, b: pad.center_crop(to_input(data.images[rows[a:b]])), len(rows))


def phase_pilot(data: Data, device) -> None:
    out_dir = RESULTS_DIR / "piloto"
    decision_path = out_dir / "decisao_epocas.json"
    if decision_path.exists():
        raise RuntimeError(f"O piloto já foi executado ({decision_path}).")
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = training_set("A", 0.5, SEED, data.train)
    validation_labels = data.validation["rotulo"].to_numpy()

    def validate(model):
        p = clean_probability(model, data, data.validation)
        auc = float(roc_auc_score(validation_labels, p))
        print(f"  AUC na validação: {auc:.4f}", flush=True)
        return {"val_auc": auc}

    pad.set_seed(SEED)
    model = build_model(device)
    history = train(model, data, chosen, chosen["rotulo"].to_numpy(), PILOT_MAX_EPOCHS, SEED,
                    device, on_epoch=validate)
    history = pd.DataFrame(history)
    history.to_csv(out_dir / "historico_piloto.csv", index=False)
    best = history.loc[history["val_auc"].idxmax()]
    decision = {"epocas": int(best["epoca"]), "val_auc_na_epoca": float(best["val_auc"]),
                "regra": "época de maior AUC na validação; q = 0,5, semente 42 (ficha, seção 5)",
                "n_treino": len(chosen), **pad.environment(device)}
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def run_one(name: str, experiment: str, q: float, seed: int, data: Data, epochs: int, device,
            shuffle_labels: bool = False) -> None:
    out_dir = RESULTS_DIR / "execucoes" / name
    if (out_dir / "execucao.json").exists():  # gravado por último: execução completa
        print(f"{name}: já executada, pulando.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    chosen = training_set(experiment, q, seed, data.train)
    labels = chosen["rotulo"].to_numpy()
    if shuffle_labels:
        labels = np.random.default_rng(seed).permutation(labels)
    pad.set_seed(seed)
    model = build_model(device)
    history = train(model, data, chosen, labels, epochs, seed, device)
    model.eval()
    torch.save(model.state_dict(), out_dir / "modelo.pt")
    chosen[["img_id", "rotulo", "sexo", "quadrado"]].to_csv(out_dir / "treino.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "historico.csv", index=False)

    test = data.test[["img_id", "patient_id", "rotulo", "sexo", "idade"]].copy()
    test["p_derrame"] = clean_probability(model, data, data.test)
    if experiment == "B":
        # Intervenção: o quadrado em todas as imagens do teste, com cantos fixados pela semente.
        rows = torch.as_tensor(data.rows(data.test), device=device)
        generator = torch.Generator(device=device).manual_seed(10_000 + seed)
        corners = sample_corners(len(rows), generator, device)
        everyone = torch.ones(len(rows), dtype=torch.bool, device=device)
        test["p_derrame_com_quadrado"] = probability(
            model, lambda a, b: paste(pad.center_crop(to_input(data.images[rows[a:b]])),
                                      corners[a:b], everyone[a:b]), len(rows))
        test["quadrado_linha"], test["quadrado_coluna"] = corners.cpu().numpy().T
    test.to_csv(out_dir / "predicoes_teste.csv", index=False)
    isolated = data.test_isolated[["img_id", "patient_id", "rotulo", "sexo"]].copy()
    isolated["p_derrame"] = clean_probability(model, data, data.test_isolated)
    isolated.to_csv(out_dir / "predicoes_teste_derrame_isolado.csv", index=False)
    validation_auc = float(roc_auc_score(data.validation["rotulo"],
                                         clean_probability(model, data, data.validation)))
    test_auc = float(roc_auc_score(test["rotulo"], test["p_derrame"]))
    summary = {"execucao": name, "experimento": experiment, "q": q, "semente": seed,
               "rotulos_embaralhados": shuffle_labels, "epocas": epochs, "n_treino": len(chosen),
               "homens_entre_positivos": float((chosen.loc[chosen["rotulo"] == 1, "sexo"] == "M").mean()),
               "quadrado_entre_positivos": float(chosen.loc[chosen["rotulo"] == 1, "quadrado"].mean()),
               "quadrado_entre_negativos": float(chosen.loc[chosen["rotulo"] == 0, "quadrado"].mean()),
               "val_auc": validation_auc, "teste_auc": test_auc,
               "segundos": time.time() - started, **pad.environment(device)}
    summary["semente"] = seed
    (out_dir / "execucao.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"{name}: AUC val {validation_auc:.3f}, teste {test_auc:.3f}, "
          f"{summary['segundos']:.0f} s", flush=True)


def phase_experiments(data: Data, device, seeds: range) -> None:
    decision_path = RESULTS_DIR / "piloto/decisao_epocas.json"
    if not decision_path.exists():
        raise RuntimeError("Execute a fase piloto antes dos experimentos.")
    epochs = json.loads(decision_path.read_text(encoding="utf-8"))["epocas"]
    for experiment in ("A", "B"):
        for q in LEVELS:
            for seed in seeds:
                run_one(f"{experiment}_q{q:g}_s{seed}", experiment, q, seed, data, epochs, device)
    if seeds == SEED_SETS["original"]:
        run_one("sanidade_rotulos_embaralhados", "A", 0.5, SEED, data, epochs, device,
                shuffle_labels=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fase", choices=["piloto", "experimentos", "replica"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA indisponível; o protocolo prevê execução na GPU.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    data = Data(device)
    print(f"Imagens na GPU: {tuple(data.images.shape)}", flush=True)
    if args.fase == "piloto":
        phase_pilot(data, device)
    else:
        phase_experiments(data, device, SEED_SETS["original" if args.fase == "experimentos" else "replica"])


if __name__ == "__main__":
    main()
