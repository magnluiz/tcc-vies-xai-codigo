"""Conferência manual da anotação de marcas de caneta do PAD-UFES-20 (piloto).

A anotação `marcas_caneta/anotacao_marcas_caneta.csv` foi feita por revisão visual com auxílio de
um assistente de IA e precisa ser conferida pelo autor. Este script tem dois modos:

  gerar     monta `marcas_caneta/revisao/revisar.html`, uma página local (abre no navegador, sem
            internet) que mostra uma imagem por vez, às cegas (sem o rótulo original nem o nome do
            arquivo). Amostra estratificada por rótulo original (padrão: 50 "sim", 50 "duvida",
            50 "nao", semente 42), em ordem embaralhada. Para medir a consistência do próprio
            revisor (efeito de cansaço), --n-repetidas imagens aparecem uma segunda vez, sem aviso,
            pelo menos --distancia posições depois da primeira. A cada --bloco imagens a página
            sugere uma pausa. Teclas: S = tem marca, N = não tem, D = dúvida, seta esquerda =
            voltar. O progresso fica salvo no navegador; "Exportar CSV" baixa
            `revisao_marcas_caneta.csv`.
  analisar  compara o CSV exportado com a anotação original: tabela de concordância, fração com
            marca por estrato (IC de Wilson 95%), estimativa ponderada para a base inteira, kappa
            de Cohen e consistência do revisor nas repetições. Grava
            `marcas_caneta/revisao/concordancia.json`.

Conferência feita em 2026-11-17: o autor usou a versão anterior da página (completa, sem repetições):
todas as 412 "sim" e 105 "duvida" e 200 "nao" sorteadas (717 imagens). As respostas, com os
img_id da amostra, estão em `marcas_caneta/revisao/revisao_marcas_caneta.csv`; o resultado, em
`concordancia.json`. O modo `analisar` aceita os dois formatos de CSV.

Uso (da raiz do projeto):
  .venv\\Scripts\\python.exe codigo\\revisao_marcas_caneta_pad_ufes.py gerar
  .venv\\Scripts\\python.exe codigo\\revisao_marcas_caneta_pad_ufes.py analisar CAMINHO\\revisao_marcas_caneta.csv
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MARKS_DIR = ROOT / "dataset/PAD-UFES-20/derivados_analise_v1/marcas_caneta"
IMAGE_ROOT = ROOT / "dataset/PAD-UFES-20/zr7vgbcyr2-1/images"
REVIEW_DIR = MARKS_DIR / "revisao"
SEED = 42
STRATA = ("sim", "duvida", "nao")

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
 #pausa{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:9;flex-direction:column;justify-content:center;align-items:center;gap:18px;font-size:20px}
</style></head><body>
<div id="pausa"><div id="pausatxt"></div><div style="font-size:15px;color:#aaa">Descanse os olhos por alguns minutos. O progresso está salvo: pode fechar e voltar depois.</div>
 <button class="s" onclick="fecharPausa()">Continuar</button></div>
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
<p style="color:#aaa;font-size:13px">Marca de caneta: tinta (azul, roxa ou preta) desenhada na pele em torno da lesão. Régua, pelos e sombras não contam. Na dúvida, use D.</p>
<script>
const ITENS = __ITENS__;
const BLOCO = __BLOCO__;
const CHAVE = "revisao_marcas_caneta_v2";
let resp = {};
try { resp = JSON.parse(localStorage.getItem(CHAVE) || "{}"); } catch(e) {}
// aproveita respostas da versão anterior (mesma imagem, primeira aparição)
try {
  const antigo = JSON.parse(localStorage.getItem("revisao_marcas_caneta_v1") || "{}");
  for (const it of ITENS) if (!it.chave.includes("#") && antigo[it.img_id] && !resp[it.chave]) resp[it.chave] = antigo[it.img_id];
} catch(e) {}
let i = 0, pausaVista = {};
function salvar(){ try { localStorage.setItem(CHAVE, JSON.stringify(resp)); } catch(e) {} }
function mostrar(){
  const it = ITENS[i];
  document.getElementById("foto").src = it.caminho;
  const feitas = Object.keys(resp).length;
  document.getElementById("pos").textContent = (i+1) + " / " + ITENS.length + "  (respondidas: " + feitas + ")";
  document.getElementById("resp").textContent = resp[it.chave] || "-";
  document.getElementById("feito").style.width = (100*feitas/ITENS.length) + "%";
}
function marcar(v){
  resp[ITENS[i].chave] = v; salvar();
  const fim = (i+1) % BLOCO === 0 && i+1 < ITENS.length;
  if (fim && !pausaVista[i]) {
    pausaVista[i] = true;
    document.getElementById("pausatxt").textContent = "Bloco " + ((i+1)/BLOCO) + " de " + Math.ceil(ITENS.length/BLOCO) + " concluído.";
    document.getElementById("pausa").style.display = "flex";
  }
  ir(1);
}
function fecharPausa(){ document.getElementById("pausa").style.display = "none"; }
function ir(d){ i = Math.max(0, Math.min(ITENS.length-1, i+d)); mostrar(); }
function pendente(){ const k = ITENS.findIndex(it => !resp[it.chave]); if(k >= 0){ i = k; } mostrar(); }
function exportar(){
  const linhas = ["chave,img_id,conferencia"].concat(ITENS.filter(it => resp[it.chave]).map(it => it.chave + "," + it.img_id + "," + resp[it.chave]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([linhas.join("\\n") + "\\n"], {type:"text/csv"}));
  a.download = "revisao_marcas_caneta.csv"; a.click();
}
document.addEventListener("keydown", e => {
  if (document.getElementById("pausa").style.display === "flex") { if (e.key === "Enter") fecharPausa(); return; }
  const k = e.key.toLowerCase();
  if(k === "s") marcar("sim"); else if(k === "n") marcar("nao"); else if(k === "d") marcar("duvida");
  else if(e.key === "ArrowLeft") ir(-1); else if(e.key === "ArrowRight") ir(1);
});
pendente();
</script></body></html>
"""


def generate(n_per: dict, n_rep: int, distance: int, block: int) -> None:
    table = pd.read_csv(MARKS_DIR / "anotacao_marcas_caneta.csv")
    paths = {path.name: path for path in IMAGE_ROOT.rglob("*.png")}
    rng = np.random.default_rng(SEED)
    parts = []
    for label in STRATA:
        stratum = table[table["marca_caneta"] == label]
        parts.append(stratum.iloc[rng.choice(len(stratum), min(n_per[label], len(stratum)), replace=False)])
    sample = pd.concat(parts)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    missing = [name for name in sample["img_id"] if name not in paths]
    assert not missing, f"imagens não encontradas: {missing[:5]}"

    # Repetições: n_rep imagens (equilibradas entre os estratos) entre as que aparecem cedo o
    # bastante para a cópia ficar a pelo menos `distance` posições da primeira aparição.
    order = list(sample["img_id"])
    eligible = sample.index[sample.index <= len(order) - distance]
    rep_ids = []
    for k, label in enumerate(STRATA):
        quota = n_rep // len(STRATA) + (1 if k < n_rep % len(STRATA) else 0)
        pool = [j for j in eligible if sample.at[j, "marca_caneta"] == label]
        rep_ids += [order[j] for j in rng.choice(pool, min(quota, len(pool)), replace=False)]
    final = list(order)
    for img in rng.permutation(rep_ids):
        first = final.index(img)
        final.insert(int(rng.integers(first + distance, len(final) + 1)), str(img) + "#2")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    items = [{"chave": key, "img_id": key.split("#")[0],
              "caminho": os.path.relpath(paths[key.split("#")[0]], REVIEW_DIR).replace(os.sep, "/")}
             for key in final]
    page = PAGE.replace("__ITENS__", json.dumps(items)).replace("__BLOCO__", str(block))
    (REVIEW_DIR / "revisar.html").write_text(page, encoding="utf-8")
    sample["repetida"] = sample["img_id"].isin(rep_ids)
    sample[["img_id", "diagnostic", "marca_caneta", "repetida"]].to_csv(REVIEW_DIR / "amostra_revisao.csv", index=False)
    print(f"{len(sample)} imagens + {len(rep_ids)} repetições = {len(items)} telas:",
          sample["marca_caneta"].value_counts().to_dict())
    print("Abrir no navegador:", REVIEW_DIR / "revisar.html")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def analyse(csv_path: Path) -> None:
    original = pd.read_csv(MARKS_DIR / "anotacao_marcas_caneta.csv")[["img_id", "diagnostic", "marca_caneta"]]
    totals = original["marca_caneta"].value_counts().to_dict()
    answers = pd.read_csv(csv_path)
    if "chave" not in answers.columns:  # CSV da versão anterior da página
        answers["chave"] = answers["img_id"]
    first = answers[~answers["chave"].str.contains("#")].merge(original, on="img_id", how="left", validate="one_to_one")
    second = answers[answers["chave"].str.contains("#")]

    table = pd.crosstab(first["marca_caneta"], first["conferencia"])
    print("Linhas: anotação original; colunas: conferência\n", table, sep="")

    strata = {}
    for label in STRATA:
        part = first[(first["marca_caneta"] == label) & (first["conferencia"] != "duvida")]
        k, n = int((part["conferencia"] == "sim").sum()), len(part)
        strata[label] = {"n_decididas": n, "n_duvida_revisor": int(((first["marca_caneta"] == label) & (first["conferencia"] == "duvida")).sum()),
                         "fracao_com_marca": k / n if n else None, "ic95": wilson(k, n), "total_na_base": int(totals.get(label, 0))}
    est = sum(s["fracao_com_marca"] * s["total_na_base"] for s in strata.values() if s["fracao_com_marca"] is not None)

    decided = first[(first["marca_caneta"] != "duvida") & (first["conferencia"] != "duvida")]
    a = (decided["marca_caneta"] == "sim").to_numpy()
    b = (decided["conferencia"] == "sim").to_numpy()
    agreement = float((a == b).mean()) if len(a) else None
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())) if len(a) else None

    pairs = second.assign(img_id=second["chave"].str.split("#").str[0]).merge(
        first[["img_id", "conferencia"]], on="img_id", suffixes=("_2", "_1"))
    same = int((pairs["conferencia_1"] == pairs["conferencia_2"]).sum())

    result = {
        "n_conferidas": int(len(first)),
        "tabela": table.to_dict(),
        "por_estrato": strata,
        "estimativa_imagens_com_marca_na_base": est,
        "sim_vs_nao_sem_duvidas": {
            "n": int(len(decided)),
            "concordancia": agreement,
            "kappa_cohen": (agreement - expected) / (1 - expected) if expected is not None and expected < 1 else None,
        },
        "consistencia_revisor": {"pares": int(len(pairs)), "respostas_iguais": same,
                                 "discordantes": pairs.loc[pairs["conferencia_1"] != pairs["conferencia_2"],
                                                           ["img_id", "conferencia_1", "conferencia_2"]].to_dict("records")},
        "nota": ("Amostra estratificada pelo rótulo original; a estimativa para a base pondera a fração com "
                 "marca de cada estrato pelo total desse estrato na anotação original."),
    }
    (REVIEW_DIR / "concordancia.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("por_estrato", "estimativa_imagens_com_marca_na_base",
                                              "sim_vs_nao_sem_duvidas", "consistencia_revisor")},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="modo", required=True)
    gerar = sub.add_parser("gerar")
    gerar.add_argument("--n-sim", type=int, default=50)
    gerar.add_argument("--n-duvida", type=int, default=50)
    gerar.add_argument("--n-nao", type=int, default=50)
    gerar.add_argument("--n-repetidas", type=int, default=15, help="Imagens mostradas duas vezes (consistência do revisor).")
    gerar.add_argument("--distancia", type=int, default=40, help="Distância mínima entre as duas aparições.")
    gerar.add_argument("--bloco", type=int, default=55, help="Imagens por bloco antes de sugerir pausa.")
    analisar = sub.add_parser("analisar")
    analisar.add_argument("csv", type=Path)
    args = parser.parse_args()
    if args.modo == "gerar":
        generate({"sim": args.n_sim, "duvida": args.n_duvida, "nao": args.n_nao},
                 args.n_repetidas, args.distancia, args.bloco)
    else:
        analyse(args.csv)


if __name__ == "__main__":
    main()
