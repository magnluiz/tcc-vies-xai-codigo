"""Predições da partição de validação para cada execução do NIH (ajuste após o congelamento).

O teste oficial do NIH concentra os pacientes mais recentes (IDs acima de 28.000) entre os
negativos, e essas imagens têm outro contraste; no treino e na validação o rótulo não depende da
época. A validação (mesma distribuição do treino, nunca usada em treino) serve de teste
alternativo nas medidas de viés. As 500 imagens de conceito do TCAV são excluídas na análise.
"""

import argparse

import pandas as pd
import torch

import treinar_nih_chestxray14 as nih
import xai_nih_chestxray14 as xai


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    cohort = pd.read_csv(nih.COHORTS["principal"])
    validation = cohort[cohort["particao"] == "validacao"].sort_values("img_id").reset_index(drop=True)
    crops = xai.crops_of(xai.load_images(validation["img_id"], device))
    names = xai.run_names("combinado") + [xai.SHUFFLED_RUN]
    for name in names:
        path = xai.RUN_DIR / name / "predicoes_validacao.csv"
        if path.exists() or not (xai.RUN_DIR / name / "execucao.json").exists():
            continue
        model = xai.load_model(name, device)
        table = validation[["img_id", "patient_id", "rotulo", "sexo", "idade"]].copy()
        table["p_derrame"] = nih.probability(model, lambda a, b: crops[a:b], len(crops))
        table.to_csv(path, index=False)
        print(name, flush=True)


if __name__ == "__main__":
    main()
