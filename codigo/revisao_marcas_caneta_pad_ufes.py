"""Conferência manual da anotação de marcas de caneta do PAD-UFES-20 (piloto).

A anotação `marcas_caneta/anotacao_marcas_caneta.csv` foi feita por revisão visual com auxílio de
um assistente de IA e precisa ser conferida pelo autor. Este script tem dois modos:

  gerar     monta `marcas_caneta/revisao/revisar.html`, uma página local (abre no navegador, sem
            internet) que mostra uma imagem por vez, às cegas: todas as marcadas "sim" e "duvida"
            e uma amostra aleatória das "nao" (--n-nao, semente 42), em ordem embaralhada.
            Teclas: S = tem marca, N = não tem, D = dúvida, seta esquerda = voltar. O progresso
            fica salvo no navegador; o botão "Exportar CSV" baixa `revisao_marcas_caneta.csv`.
  analisar  compara o CSV exportado com a anotação original: tabela de concordância, precisão e
            revocação da anotação original (tomando a conferência como referência) e kappa de
            Cohen. Grava `marcas_caneta/revisao/concordancia.json`.

Uso (da raiz do projeto):
  .venv\\Scripts\\python.exe codigo\\revisao_marcas_caneta_pad_ufes.py gerar
  .venv\\Scripts\\python.exe codigo\\revisao_marcas_caneta_pad_ufes.py analisar CAMINHO\\revisao_marcas_caneta.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MARKS_DIR = ROOT / "dataset/PAD-UFES-20/derivados_analise_v1/marcas_caneta"
IMAGE_ROOT = ROOT / "dataset/PAD-UFES-20/zr7vgbcyr2-1/images"
REVIEW_DIR = MARKS_DIR / "revisao"
SEED = 42

PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Conferência das marcas de caneta</title>
<style>
 body{font-family:system-ui,sans-serif;background:#1b1b1b;color:#eee;margin:0;text-align:center}
 #topo{padding:10px;background:#2a2a2a;display:flex;gap:16px;justify-content:center;align-items:center;flex-wrap:wrap}
 img{max-width:96vw;max-height:74vh;margin-top:12px;background:#000}
 button{font-size:16px;padding:8px 16px;border:0;border-radius:6px;cursor:pointer}
 .s{background:#2e7d32;color:#fff}.n{background:#b23b3b;color:#fff}.d{background:#b08900;color:#fff}.x{background:#555;color:#fff}
 #barra{height:6px;background:#444}#feito{height:6px;background:#4caf50;width:0}
 #resp{font-weight:bold}
</style></head><body>
<div id="barra"><div id="feito"></div></div>
<div id="topo">
 <span id="pos"></span><span>Resposta: <span id="resp">-</span></span>
 <button class="s" onclick="marcar('sim')">S: tem marca</button>
 <button class="n" onclick="marcar('nao')">N: não tem</button>
 <button class="d" onclick="marcar('duvida')">D: dúvida</button>
 <button class="x" onclick="ir(-1)">&larr; voltar</button>
 <button class="x" onclick="ir(1)">avançar &rarr;</button>
 <button class="x" onclick="pendente()">próxima sem resposta</button>
 <button class="x" onclick="exportar()">Exportar CSV</button>
</div>
<img id="foto" alt="lesão">
<p id="nome" style="color:#888;font-size:12px"></p>
<p style="color:#aaa;font-size:13px">Marca de caneta: tinta (azul, roxa ou preta) desenhada na pele em torno da lesão. Régua, pelos e sombras não contam.</p>
<script>
const ITENS = __ITENS__;
const CHAVE = "revisao_marcas_caneta_v1";
let resp = JSON.parse(localStorage.getItem(CHAVE) || "{}");
let i = 0;
function mostrar(){
  const it = ITENS[i];
  document.getElementById("foto").src = it.caminho;
  document.getElementById("nome").textContent = it.img_id;
  const feitas = Object.keys(resp).length;
  document.getElementById("pos").textContent = (i+1) + " / " + ITENS.length + "  (respondidas: " + feitas + ")";
  document.getElementById("resp").textContent = resp[it.img_id] || "-";
  document.getElementById("feito").style.width = (100*feitas/ITENS.length) + "%";
}
function marcar(v){ resp[ITENS[i].img_id] = v; localStorage.setItem(CHAVE, JSON.stringify(resp)); ir(1); }
function ir(d){ i = Math.max(0, Math.min(ITENS.length-1, i+d)); mostrar(); }
function pendente(){ const k = ITENS.findIndex(it => !resp[it.img_id]); if(k >= 0){ i = k; } mostrar(); }
function exportar(){
  const linhas = ["img_id,conferencia"].concat(ITENS.filter(it => resp[it.img_id]).map(it => it.img_id + "," + resp[it.img_id]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([linhas.join("\\n") + "\\n"], {type:"text/csv"}));
  a.download = "revisao_marcas_caneta.csv"; a.click();
}
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if(k === "s") marcar("sim"); else if(k === "n") marcar("nao"); else if(k === "d") marcar("duvida");
  else if(e.key === "ArrowLeft") ir(-1); else if(e.key === "ArrowRight") ir(1);
});
pendente();
</script></body></html>
"""


def generate(n_no: int) -> None:
    table = pd.read_csv(MARKS_DIR / "anotacao_marcas_caneta.csv")
    paths = {path.name: path for path in IMAGE_ROOT.rglob("*.png")}
    rng = np.random.default_rng(SEED)
    no = table[table["marca_caneta"] == "nao"]
    sample = pd.concat([table[table["marca_caneta"] != "nao"],
                        no.iloc[rng.choice(len(no), min(n_no, len(no)), replace=False)]])
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    missing = [name for name in sample["img_id"] if name not in paths]
    assert not missing, f"imagens não encontradas: {missing[:5]}"
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    items = [{"img_id": name, "caminho": os.path.relpath(paths[name], REVIEW_DIR).replace(os.sep, "/")}
             for name in sample["img_id"]]
    (REVIEW_DIR / "revisar.html").write_text(PAGE.replace("__ITENS__", json.dumps(items)), encoding="utf-8")
    sample[["img_id", "diagnostic", "marca_caneta"]].to_csv(REVIEW_DIR / "amostra_revisao.csv", index=False)
    print(f"{len(sample)} imagens:", sample["marca_caneta"].value_counts().to_dict())
    print("Abrir no navegador:", REVIEW_DIR / "revisar.html")


def analyse(csv_path: Path) -> None:
    original = pd.read_csv(MARKS_DIR / "anotacao_marcas_caneta.csv")[["img_id", "diagnostic", "marca_caneta"]]
    review = pd.read_csv(csv_path).merge(original, on="img_id", how="left", validate="one_to_one")
    table = pd.crosstab(review["marca_caneta"], review["conferencia"])
    print("Linhas: anotação original; colunas: conferência\n", table, sep="")
    decided = review[(review["marca_caneta"] != "duvida") & (review["conferencia"] != "duvida")]
    a = (decided["marca_caneta"] == "sim").to_numpy()
    b = (decided["conferencia"] == "sim").to_numpy()
    agreement = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    result = {
        "n_conferidas": int(len(review)),
        "tabela": table.to_dict(),
        "sem_duvidas": {
            "n": int(len(decided)),
            "concordancia": agreement,
            "kappa_cohen": (agreement - expected) / (1 - expected) if expected < 1 else None,
            "precisao_do_sim_original": float(b[a].mean()) if a.any() else None,
            "fracao_de_nao_original_com_marca": float(b[~a].mean()) if (~a).any() else None,
        },
        "nota": ("As imagens 'nao' conferidas são uma amostra aleatória; a revocação na base inteira "
                 "deve ser estimada ponderando essa fração pelo total de 'nao' da anotação original."),
    }
    (REVIEW_DIR / "concordancia.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["sem_duvidas"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="modo", required=True)
    gerar = sub.add_parser("gerar")
    gerar.add_argument("--n-nao", type=int, default=200, help="Tamanho da amostra de imagens anotadas como 'nao'.")
    analisar = sub.add_parser("analisar")
    analisar.add_argument("csv", type=Path)
    args = parser.parse_args()
    if args.modo == "gerar":
        generate(args.n_nao)
    else:
        analyse(args.csv)


if __name__ == "__main__":
    main()
