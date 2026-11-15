"""Explicações e análise dos treinos do NIH ChestX-ray14 (ficha congelada, seções 6 a 10).

Para cada execução gravada por `treinar_nih_chestxray14.py`:
  - experimento A: Grad-CAM e IG da classe derrame em todo o teste (concentração na região
    torácica e, como sinal exploratório, diferença entre os mapas médios de mulheres e homens
    com derrame) e TCAV do conceito "masculino" nas camadas layer3 e layer4;
  - experimento B: Grad-CAM e IG da classe derrame nas imagens negativas do teste com o
    quadrado inserido (concentração no quadrado) e, como controle negativo exploratório, o
    mesmo TCAV.
Depois: medidas de viés, testes de sanidade e correlações de Spearman entre execuções.
Resultados por execução ficam em cache; `--so-analise` refaz só a análise. `--conjunto` escolhe
as sementes: "original" (42 a 46), "replica" (47 a 51, réplica pré-registrada, ficha, ajuste 2)
ou "combinado" (as 10); as saídas da réplica e do combinado levam esse sufixo no nome.
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
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import treinar_nih_chestxray14 as nih
import treinar_pad_ufes as pad
from xai_atribuicoes import GradCAM, integrated_gradients, mass_fraction


OUT_DIR = nih.RESULTS_DIR / "xai"
RUN_DIR = nih.RESULTS_DIR / "execucoes"
MASK_DIR = nih.NIH_DIR / "derivados_v1/mascaras_toracicas"
IG_STEPS = 32
TCAV_LAYERS = ("layer3", "layer4")
N_CONCEPT = 250
N_RANDOM_CAVS = 20
N_BOOTSTRAP = 2000
N_IG_SANITY = 300
CASCADE = ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]
SANITY_RUN = "A_q0.5_s42"
SHUFFLED_RUN = "sanidade_rotulos_embaralhados"
PATCH_AREA = nih.PATCH ** 2 / nih.CROP ** 2

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_GRADCAM = "#2a78d6"
COLOR_IG = "#eb6834"
COLOR_TCAV3 = "#1f9e89"
COLOR_TCAV4 = "#7b5ea7"


def load_images(img_ids, device) -> torch.Tensor:
    index = pd.read_csv(nih.CACHE_DIR / "indice.csv")
    position = dict(zip(index["Image Index"], index["posicao"]))
    cache = np.load(nih.CACHE_DIR / "imagens_256.npy", mmap_mode="r")
    positions = np.array([position[img_id] for img_id in img_ids])
    order = np.argsort(positions)
    images = np.empty((len(positions), 256, 256), dtype=np.uint8)
    images[order] = cache[positions[order]]
    return torch.from_numpy(images).unsqueeze(1).to(device)


def crops_of(images: torch.Tensor) -> torch.Tensor:
    return pad.center_crop(nih.to_input(images))


def load_model(name: str, device) -> torch.nn.Module:
    model = nih.build_model(device)
    model.load_state_dict(torch.load(RUN_DIR / name / "modelo.pt", map_location=device))
    return model.eval()


def gradcam_maps(model, crops: torch.Tensor) -> np.ndarray:
    cam = GradCAM(model)
    maps = []
    for start in range(0, len(crops), 64):
        batch = pad.normalize(crops[start:start + 64])
        maps.append(cam(batch, torch.ones(len(batch), dtype=torch.long, device=batch.device)).cpu().numpy())
    cam.remove()
    return np.concatenate(maps)


def ig_maps(model, crops: torch.Tensor) -> np.ndarray:
    baseline = pad.normalize(torch.zeros(1, 3, nih.CROP, nih.CROP, device=crops.device))[0]
    maps = []
    for start in range(0, len(crops), 16):
        batch = pad.normalize(crops[start:start + 16])
        maps.append(integrated_gradients(
            model, batch, torch.ones(len(batch), dtype=torch.long, device=batch.device),
            baseline, IG_STEPS).cpu().numpy())
    return np.concatenate(maps)


def mean_normalized(maps: np.ndarray) -> np.ndarray:
    totals = maps.reshape(len(maps), -1).sum(axis=1)
    valid = totals > 0
    return (maps[valid] / totals[valid, None, None]).mean(axis=0)


def sex_dissimilarity(maps: np.ndarray, rows: pd.DataFrame) -> float:
    """1 − Pearson entre os mapas médios (soma 1) de mulheres e homens com derrame."""
    positive = rows["rotulo"].to_numpy() == 1
    female = mean_normalized(maps[positive & (rows["sexo"].to_numpy() == "F")])
    male = mean_normalized(maps[positive & (rows["sexo"].to_numpy() == "M")])
    return float(1 - np.corrcoef(female.ravel(), male.ravel())[0, 1])


class Activations:
    """Guarda a saída (e o gradiente, quando houver) das camadas do TCAV."""

    def __init__(self, model):
        self.outputs = {}
        self.handles = [getattr(model, name).register_forward_hook(self._hook(name))
                        for name in TCAV_LAYERS]

    def _hook(self, name):
        def hook(_module, _inputs, output):
            if output.requires_grad:
                output.retain_grad()
            self.outputs[name] = output
        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def concept_activations(model, crops: torch.Tensor) -> dict:
    hooks = Activations(model)
    collected = {name: [] for name in TCAV_LAYERS}
    for start in range(0, len(crops), 64):
        model(pad.normalize(crops[start:start + 64]))
        for name in TCAV_LAYERS:
            collected[name].append(hooks.outputs[name].flatten(1).float().cpu().numpy())
    hooks.remove()
    return {name: np.concatenate(values) for name, values in collected.items()}


def fit_cav(features: np.ndarray, labels: np.ndarray, train_index: np.ndarray,
            held_out: np.ndarray) -> tuple[np.ndarray, float]:
    classifier = LogisticRegression(C=1.0, max_iter=2000)
    classifier.fit(features[train_index], labels[train_index])
    accuracy = float(classifier.score(features[held_out], labels[held_out]))
    vector = classifier.coef_[0]
    return vector / np.linalg.norm(vector), accuracy


def directional_derivatives(model, crops: torch.Tensor, cavs: dict) -> dict:
    """Derivada de (logit derrame − logit sem achados) na direção de cada CAV, por imagem."""
    hooks = Activations(model)
    directions = {name: torch.as_tensor(np.stack(vectors), dtype=torch.float32, device=crops.device)
                  for name, vectors in cavs.items()}
    collected = {name: [] for name in TCAV_LAYERS}
    for start in range(0, len(crops), 32):
        batch = pad.normalize(crops[start:start + 32]).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        (logits[:, 1] - logits[:, 0]).sum().backward()
        for name in TCAV_LAYERS:
            gradient = hooks.outputs[name].grad.flatten(1)
            collected[name].append((gradient @ directions[name].T).cpu().numpy())
    hooks.remove()
    return {name: np.concatenate(values) for name, values in collected.items()}


def tcav(model, concept_crops, concept_sex, positive_crops, n_random: int = N_RANDOM_CAVS) -> dict:
    labels = (concept_sex == "M").astype(int)
    train_index, held_out = train_test_split(np.arange(len(labels)), test_size=0.2,
                                             stratify=labels, random_state=nih.SEED)
    activations = concept_activations(model, concept_crops)
    cavs, accuracy = {}, {}
    for name in TCAV_LAYERS:
        vector, accuracy[name] = fit_cav(activations[name], labels, train_index, held_out)
        cavs[name] = [vector]
        for seed in range(n_random):
            random_labels = np.zeros(len(labels), dtype=int)
            random_labels[np.random.default_rng(seed).permutation(len(labels))[:len(labels) // 2]] = 1
            cavs[name].append(fit_cav(activations[name], random_labels, train_index, held_out)[0])
    derivatives = directional_derivatives(model, positive_crops, cavs)
    result = {}
    for name in TCAV_LAYERS:
        scores = (derivatives[name] > 0).mean(axis=0)
        low, high = np.percentile(scores[1:], [2.5, 97.5])
        result[name] = {"escore": float(scores[0]),
                        "derivada_media": float(derivatives[name][:, 0].mean()),
                        "escores_aleatorios": scores[1:].tolist(),
                        "faixa_aleatoria_2_5_97_5": [float(low), float(high)],
                        "significativo": bool(scores[0] < low or scores[0] > high),
                        "acuracia_cav_20pct": accuracy[name]}
    return result


def concept_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """Imagens de conceito do TCAV: 250 sem achados de cada sexo, da validação (semente 42)."""
    validation = cohort[(cohort["particao"] == "validacao") & (cohort["rotulo"] == 0)]
    rng = np.random.default_rng(nih.SEED)
    parts = [validation[validation["sexo"] == sex].iloc[
        np.sort(rng.choice((validation["sexo"] == sex).sum(), N_CONCEPT, replace=False))]
        for sex in ("F", "M")]
    return pd.concat(parts).reset_index(drop=True)


class Context:
    """Imagens e tabelas compartilhadas por todas as execuções."""

    def __init__(self, device):
        cohort = pd.read_csv(nih.COHORTS["principal"])
        self.test = cohort[cohort["particao"] == "teste"].sort_values("img_id").reset_index(drop=True)
        self.concepts = concept_table(cohort)
        self.test_crops = crops_of(load_images(self.test["img_id"], device))
        self.concept_crops = crops_of(load_images(self.concepts["img_id"], device))
        self.positive = np.flatnonzero(self.test["rotulo"].to_numpy() == 1)
        masks = pd.read_csv(MASK_DIR / "mascaras_teste_224.csv")
        if not masks["img_id"].equals(self.test["img_id"]):
            raise RuntimeError("As máscaras não estão na ordem do teste.")
        self.thorax = np.load(MASK_DIR / "mascaras_teste_224.npy").astype(np.float32)
        self.thorax_area = masks["fracao_area"].to_numpy()
        self.thorax_ok = ~masks["falha_segmentacao"].to_numpy()


def explain_run_a(name: str, model, context: Context) -> tuple[dict, pd.DataFrame]:
    rows = context.test[["img_id", "rotulo", "sexo"]].copy()
    summary = {}
    for method, function in (("gradcam", gradcam_maps), ("ig", ig_maps)):
        maps = function(model, context.test_crops)
        concentration = mass_fraction(maps, context.thorax) / context.thorax_area
        concentration[~context.thorax_ok] = np.nan
        rows[f"conc_torax_{method}"] = concentration
        summary[f"conc_torax_{method}"] = float(np.nanmean(concentration))
        summary[f"excluidas_{method}"] = int(np.isnan(concentration).sum())
        summary[f"dissimilaridade_sexo_{method}"] = sex_dissimilarity(maps, rows)
    result = tcav(model, context.concept_crops, context.concepts["sexo"].to_numpy(),
                  context.test_crops[context.positive])
    for layer, values in result.items():
        summary[f"tcav_{layer}"] = values["escore"]
        summary[f"tcav_{layer}_derivada"] = values["derivada_media"]
    summary["tcav_detalhe"] = result
    return summary, rows


def explain_run_b(name: str, model, context: Context) -> tuple[dict, pd.DataFrame]:
    predictions = pd.read_csv(RUN_DIR / name / "predicoes_teste.csv")
    if not predictions["img_id"].equals(context.test["img_id"]):
        raise RuntimeError("Predições fora da ordem do teste.")
    negative = np.flatnonzero(predictions["rotulo"].to_numpy() == 0)
    corners = predictions.loc[negative, ["quadrado_linha", "quadrado_coluna"]].to_numpy()
    device = context.test_crops.device
    crops = nih.paste(context.test_crops[negative], torch.as_tensor(corners, device=device),
                      torch.ones(len(negative), dtype=torch.bool, device=device))
    masks = np.zeros((len(negative), nih.CROP, nih.CROP), dtype=np.float32)
    for index, (row, column) in enumerate(corners):
        masks[index, row:row + nih.PATCH, column:column + nih.PATCH] = 1.0
    rows = predictions.loc[negative, ["img_id", "sexo"]].reset_index(drop=True)
    clean, patched = predictions.loc[negative, "p_derrame"], predictions.loc[negative, "p_derrame_com_quadrado"]
    rows["virou_derrame"] = ((clean < 0.5) & (patched >= 0.5)).to_numpy()
    rows["delta_p"] = (patched - clean).to_numpy()
    summary = {}
    for method, function in (("gradcam", gradcam_maps), ("ig", ig_maps)):
        concentration = mass_fraction(function(model, crops), masks) / PATCH_AREA
        rows[f"conc_quadrado_{method}"] = concentration
        summary[f"conc_quadrado_{method}"] = float(np.nanmean(concentration))
        summary[f"excluidas_{method}"] = int(np.isnan(concentration).sum())
    result = tcav(model, context.concept_crops, context.concepts["sexo"].to_numpy(),
                  context.test_crops[context.positive])
    for layer, values in result.items():
        summary[f"tcav_{layer}"] = values["escore"]
    summary["tcav_detalhe"] = result
    return summary, rows


SUFFIXES = {"original": "", "replica": "_replica", "combinado": "_combinado"}


def run_names(conjunto: str = "original") -> list[str]:
    sets = ("original", "replica") if conjunto == "combinado" else (conjunto,)
    seeds = [seed for key in sets for seed in nih.SEED_SETS[key]]
    return [f"{experiment}_q{q:g}_s{seed}" for experiment in ("A", "B") for q in nih.LEVELS
            for seed in seeds]


def explain_all(context: Context, device, names: list[str]) -> None:
    out = OUT_DIR / "por_execucao"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (out / f"{name}.json").exists():
            continue
        if not (RUN_DIR / name / "execucao.json").exists():
            print(f"{name}: treino ainda não gravado, pulando.")
            continue
        model = load_model(name, device)
        explain = explain_run_a if name.startswith("A") else explain_run_b
        summary, rows = explain(name, model, context)
        rows.to_csv(out / f"{name}_por_imagem.csv", index=False)
        (out / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        brief = {k: round(v, 3) for k, v in summary.items() if isinstance(v, float)}
        print(f"{name}: {brief}", flush=True)


def rank_correlations(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Spearman por imagem entre dois conjuntos de mapas (NaN se algum mapa é constante)."""
    first = rankdata(reference.reshape(len(reference), -1), axis=1)
    second = rankdata(other.reshape(len(other), -1), axis=1)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.sqrt((first ** 2).sum(axis=1) * (second ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, (first * second).sum(axis=1) / denominator, np.nan)


def describe(correlations: np.ndarray) -> dict:
    return {"mediana": float(np.nanmedian(correlations)),
            "q1": float(np.nanpercentile(correlations, 25)),
            "q3": float(np.nanpercentile(correlations, 75)),
            "n_validos": int(np.isfinite(correlations).sum())}


def sanity(context: Context, device) -> dict:
    path = OUT_DIR / "sanidade.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    subset = np.sort(np.random.default_rng(nih.SEED).choice(len(context.test), N_IG_SANITY, replace=False))
    inputs = {"gradcam": (gradcam_maps, context.test_crops),
              "ig": (ig_maps, context.test_crops[torch.as_tensor(subset, device=device)])}
    reference_model = load_model(SANITY_RUN, device)
    reference = {method: function(reference_model, crops) for method, (function, crops) in inputs.items()}
    result = {"modelo_referencia": SANITY_RUN, "n_imagens_ig": N_IG_SANITY, "cascata": {}}
    torch.manual_seed(nih.SEED)
    model = load_model(SANITY_RUN, device)
    for method in inputs:
        result["cascata"][method] = []
    for layer in CASCADE:
        pad_randomize(getattr(model, layer))
        if layer == "conv1":
            pad_randomize(model.bn1)
        for method, (function, crops) in inputs.items():
            result["cascata"][method].append(
                {"camada_randomizada": layer, **describe(rank_correlations(reference[method], function(model, crops)))})
    shuffled = load_model(SHUFFLED_RUN, device)
    result["rotulos_embaralhados"] = {
        method: describe(rank_correlations(reference[method], function(shuffled, crops)))
        for method, (function, crops) in inputs.items()}
    result["rotulos_embaralhados"]["val_auc"] = json.loads(
        (RUN_DIR / SHUFFLED_RUN / "execucao.json").read_text(encoding="utf-8"))["val_auc"]
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def pad_randomize(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
        if isinstance(child, torch.nn.BatchNorm2d):
            child.reset_running_stats()


def bias_metrics(predictions: pd.DataFrame) -> dict:
    predicted = predictions["p_derrame"].to_numpy() >= 0.5
    label = predictions["rotulo"].to_numpy() == 1
    sex = predictions["sexo"].to_numpy()
    result = {"auc": float(roc_auc_score(label, predictions["p_derrame"]))}
    accuracy = {}
    for group in ("F", "M"):
        in_group = sex == group
        result[f"auc_{group}"] = float(roc_auc_score(label[in_group], predictions["p_derrame"][in_group]))
        result[f"sensibilidade_{group}"] = float(predicted[in_group & label].mean())
        result[f"fpr_{group}"] = float(predicted[in_group & ~label].mean())
        accuracy[f"acc_{group}_derrame"] = result[f"sensibilidade_{group}"]
        accuracy[f"acc_{group}_sem_achados"] = 1 - result[f"fpr_{group}"]
    result.update(accuracy)
    result["acuracia_pior_grupo"] = min(accuracy.values())
    result["dif_sensibilidade"] = result["sensibilidade_M"] - result["sensibilidade_F"]
    result["dif_fpr"] = result["fpr_M"] - result["fpr_F"]
    return result


def bootstrap_bias(predictions: pd.DataFrame, rng: np.random.Generator) -> dict:
    """IC de 95% (uma imagem por paciente, então reamostrar imagens = reamostrar pacientes)."""
    predicted = predictions["p_derrame"].to_numpy() >= 0.5
    label = predictions["rotulo"].to_numpy() == 1
    male = predictions["sexo"].to_numpy() == "M"
    draws = rng.integers(0, len(predictions), (N_BOOTSTRAP, len(predictions)))
    p, y, m = predicted[draws], label[draws], male[draws]

    def rate(mask):
        return (p & mask).sum(axis=1) / mask.sum(axis=1)

    sens_m, sens_f = rate(y & m), rate(y & ~m)
    spec_m, spec_f = 1 - rate(~y & m), 1 - rate(~y & ~m)
    worst = np.minimum.reduce([sens_m, sens_f, spec_m, spec_f])
    return {"dif_sensibilidade_ic95": np.percentile(sens_m - sens_f, [2.5, 97.5]).tolist(),
            "acuracia_pior_grupo_ic95": np.percentile(worst, [2.5, 97.5]).tolist()}


def run_table(names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(nih.SEED)
    concepts = set(concept_table(pd.read_csv(nih.COHORTS["principal"]))["img_id"])
    records = []
    for name in names:
        xai_path = OUT_DIR / "por_execucao" / f"{name}.json"
        if not xai_path.exists():
            continue
        execution = json.loads((RUN_DIR / name / "execucao.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(RUN_DIR / name / "predicoes_teste.csv")
        record = {"execucao": name, "experimento": execution["experimento"], "q": execution["q"],
                  "semente": execution["semente"], "val_auc": execution["val_auc"],
                  **bias_metrics(predictions), **bootstrap_bias(predictions, rng)}
        isolated = pd.read_csv(RUN_DIR / name / "predicoes_teste_derrame_isolado.csv")
        record["dif_sensibilidade_derrame_isolado"] = bias_metrics(isolated)["dif_sensibilidade"]
        # Ajuste após o congelamento (Dia 10): a validação, sem as imagens de conceito, como
        # teste alternativo sem a mistura de época de aquisição do teste oficial.
        validation_path = RUN_DIR / name / "predicoes_validacao.csv"
        if validation_path.exists():
            validation = pd.read_csv(validation_path)
            validation = validation[~validation["img_id"].isin(concepts)]
            record.update({f"{key}_validacao": value for key, value in bias_metrics(validation).items()})
        if execution["experimento"] == "B":
            negative = predictions[predictions["rotulo"] == 0]
            record["taxa_virou_derrame"] = float(((negative["p_derrame"] < 0.5)
                                                  & (negative["p_derrame_com_quadrado"] >= 0.5)).mean())
            record["delta_p_medio"] = float((negative["p_derrame_com_quadrado"] - negative["p_derrame"]).mean())
        xai = json.loads(xai_path.read_text(encoding="utf-8"))
        record.update({k: v for k, v in xai.items() if not isinstance(v, dict)})
        for layer in TCAV_LAYERS:
            record[f"tcav_{layer}_significativo"] = xai["tcav_detalhe"][layer]["significativo"]
            record[f"acuracia_cav_{layer}"] = xai["tcav_detalhe"][layer]["acuracia_cav_20pct"]
        records.append(record)
    return pd.DataFrame(records)


def spearman_with_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    rho = spearmanr(x, y).statistic
    draws = []
    for _ in range(N_BOOTSTRAP):
        index = rng.integers(0, len(x), len(x))
        if np.ptp(x[index]) > 0 and np.ptp(y[index]) > 0:
            draws.append(spearmanr(x[index], y[index]).statistic)
    return {"rho": float(rho), "ic95": np.nanpercentile(draws, [2.5, 97.5]).tolist()}


def paired_difference(x, first, second, rng) -> dict:
    """IC por bootstrap das execuções para rho(first, x) − rho(second, x)."""
    observed = spearmanr(x, first).statistic - spearmanr(x, second).statistic
    draws = []
    for _ in range(N_BOOTSTRAP):
        index = rng.integers(0, len(x), len(x))
        if np.ptp(x[index]) > 0 and np.ptp(first[index]) > 0 and np.ptp(second[index]) > 0:
            draws.append(spearmanr(x[index], first[index]).statistic
                         - spearmanr(x[index], second[index]).statistic)
    return {"diferenca": float(observed), "ic95": np.nanpercentile(draws, [2.5, 97.5]).tolist()}


def analyse(runs: pd.DataFrame) -> dict:
    rng = np.random.default_rng(nih.SEED)
    result = {}
    a = runs[runs["experimento"] == "A"]
    if len(a) > 2:
        q = a["q"].to_numpy()
        real = a["dif_sensibilidade"].to_numpy()
        signals = ["conc_torax_gradcam", "conc_torax_ig", "dissimilaridade_sexo_gradcam",
                   "dissimilaridade_sexo_ig", "tcav_layer3", "tcav_layer4",
                   "tcav_layer3_derivada", "tcav_layer4_derivada"]
        result["A"] = {
            "n_execucoes": len(a),
            "H1_verificacao_manipulacao": {
                "dif_sensibilidade_vs_q": spearman_with_ci(q, real, rng),
                "acuracia_pior_grupo_vs_q": spearman_with_ci(q, a["acuracia_pior_grupo"].to_numpy(), rng),
                "dif_fpr_vs_q": spearman_with_ci(q, a["dif_fpr"].to_numpy(), rng),
                "sensibilidade_derrame_isolado_vs_q": spearman_with_ci(
                    q, a["dif_sensibilidade_derrame_isolado"].to_numpy(), rng)},
            "sensibilidade_validacao_sem_conceitos": {
                column: spearman_with_ci(q, a[f"{column}_validacao"].to_numpy(), rng)
                for column in ("dif_sensibilidade", "acuracia_pior_grupo", "dif_fpr")
                if f"{column}_validacao" in a},
            "sinais": {signal: {"vs_q": spearman_with_ci(q, a[signal].to_numpy(), rng),
                                "vs_dif_sensibilidade": spearman_with_ci(real, a[signal].to_numpy(), rng)}
                       for signal in signals},
            "H3_tcav_menos_pixel_rho_com_q": {
                f"{tcav_signal}_menos_{pixel}": paired_difference(q, a[tcav_signal].to_numpy(), a[pixel].to_numpy(), rng)
                for tcav_signal in ("tcav_layer3", "tcav_layer4")
                for pixel in ("conc_torax_gradcam", "conc_torax_ig")},
            "tcav_significativo_por_q": a.groupby("q")[["tcav_layer3_significativo", "tcav_layer4_significativo"]]
                .mean().to_dict(),
            "acuracia_cav_media": {layer: float(a[f"acuracia_cav_{layer}"].mean()) for layer in TCAV_LAYERS},
        }
    b = runs[runs["experimento"] == "B"]
    if len(b) > 2:
        q = b["q"].to_numpy()
        real = b["taxa_virou_derrame"].to_numpy()
        per_level_auc = {}
        for method in ("gradcam", "ig"):
            per_level_auc[method] = {}
            for level in nih.LEVELS[1:]:
                images = pd.concat([pd.read_csv(OUT_DIR / "por_execucao" / f"{name}_por_imagem.csv")
                                    for name in b.loc[b["q"] == level, "execucao"]])
                images = images.dropna(subset=[f"conc_quadrado_{method}"])
                per_level_auc[method][str(level)] = (
                    float(roc_auc_score(images["virou_derrame"].astype(bool), images[f"conc_quadrado_{method}"]))
                    if images["virou_derrame"].nunique() == 2 else None)
        result["B"] = {
            "n_execucoes": len(b),
            "H1_verificacao_manipulacao": {"taxa_virou_vs_q": spearman_with_ci(q, real, rng)},
            "sinais": {signal: {"vs_q": spearman_with_ci(q, b[signal].to_numpy(), rng),
                                "vs_taxa_virou": spearman_with_ci(real, b[signal].to_numpy(), rng)}
                       for signal in ("conc_quadrado_gradcam", "conc_quadrado_ig")},
            "auc_por_imagem_identifica_virou": per_level_auc,
            "controle_negativo_exploratorio_tcav_sexo": {
                layer: spearman_with_ci(q, b[f"tcav_{layer}"].to_numpy(), rng) for layer in TCAV_LAYERS},
        }
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


def series(ax, runs, column, color, label, offset=0.0, label_dy=0) -> None:
    x = np.arange(len(nih.LEVELS))
    grouped = runs.groupby("q")[column]
    for position, (_, values) in enumerate(grouped):
        ax.scatter(np.full(len(values), position + offset) + np.linspace(-0.05, 0.05, len(values)),
                   values, s=16, color=color, alpha=0.45, linewidths=0)
    medians = grouped.median().to_numpy()
    ax.plot(x + offset, medians, color=color, linewidth=2, marker="o", markersize=6, label=label)
    ax.annotate(label, (x[-1] + offset, medians[-1]), xytext=(8, label_dy), textcoords="offset points",
                va="center", fontsize=8, color=INK_SECONDARY)


def finish(ax, title, ylabel) -> None:
    style_axes(ax)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_xticks(np.arange(len(nih.LEVELS)), [f"q = {q:g}" for q in nih.LEVELS])
    ax.set_xlim(-0.4, len(nih.LEVELS) - 0.1)


def plot(runs: pd.DataFrame, suffix: str = "") -> None:
    a = runs[runs["experimento"] == "A"]
    if len(a):
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), facecolor=SURFACE)
        series(axes[0], a, "dif_sensibilidade", INK_SECONDARY, "H − M")
        axes[0].axhline(0, color=BASELINE, linewidth=1)
        finish(axes[0], "Viés real: diferença de sensibilidade", "Sensibilidade H − M")
        series(axes[1], a, "conc_torax_gradcam", COLOR_GRADCAM, "Grad-CAM", -0.08)
        series(axes[1], a, "conc_torax_ig", COLOR_IG, "IG", 0.08)
        axes[1].axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        finish(axes[1], "Por pixel: concentração no tórax", "Fração da atribuição ÷ fração da área")
        series(axes[2], a, "tcav_layer3", COLOR_TCAV3, "layer3", -0.08, label_dy=-12)
        series(axes[2], a, "tcav_layer4", COLOR_TCAV4, "layer4", 0.08, label_dy=8)
        axes[2].axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        axes[2].set_ylim(0, 1)
        finish(axes[2], "Por conceito: TCAV de \"masculino\"", "Escore TCAV (derrame)")
        for ax in axes:
            ax.set_xlabel("Correlação sexo–derrame no treino", color=INK_SECONDARY)
        figure.tight_layout()
        figure.savefig(OUT_DIR / f"experimento_A{suffix}.png", dpi=200, facecolor=SURFACE)
        plt.close(figure)
    b = runs[runs["experimento"] == "B"]
    if len(b):
        figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE)
        series(axes[0], b, "taxa_virou_derrame", INK_SECONDARY, "negativos")
        axes[0].set_ylim(bottom=0)
        finish(axes[0], "Dependência real: negativos que viram derrame", "Proporção do teste")
        series(axes[1], b, "conc_quadrado_gradcam", COLOR_GRADCAM, "Grad-CAM", -0.08)
        series(axes[1], b, "conc_quadrado_ig", COLOR_IG, "IG", 0.08)
        axes[1].axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        axes[1].set_ylim(bottom=0)
        finish(axes[1], "Por pixel: concentração no quadrado", "Fração da atribuição ÷ fração da área")
        for ax in axes:
            ax.set_xlabel("Correlação quadrado–derrame no treino", color=INK_SECONDARY)
        figure.tight_layout()
        figure.savefig(OUT_DIR / f"experimento_B{suffix}.png", dpi=200, facecolor=SURFACE)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--so-analise", action="store_true")
    parser.add_argument("--conjunto", choices=list(SUFFIXES), default="original")
    args = parser.parse_args()
    names, suffix = run_names(args.conjunto), SUFFIXES[args.conjunto]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    if not args.so_analise:
        context = Context(device)
        explain_all(context, device, names)
        if (RUN_DIR / SHUFFLED_RUN / "modelo.pt").exists():
            result["sanidade"] = sanity(context, device)
    elif (OUT_DIR / "sanidade.json").exists():
        result["sanidade"] = json.loads((OUT_DIR / "sanidade.json").read_text(encoding="utf-8"))
    runs = run_table(names)
    runs.to_csv(OUT_DIR / f"execucoes{suffix}.csv", index=False)
    result.update(analyse(runs))
    (OUT_DIR / f"analise{suffix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    plot(runs, suffix)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
