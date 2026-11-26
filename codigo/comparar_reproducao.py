"""Compara os resultados de uma reprodução com os de referência (JSON e CSV).

Para cada arquivo .json ou .csv presente nas duas árvores, informa se é idêntico byte a byte; se
não for, compara o conteúdo: no JSON, a maior diferença absoluta entre os valores numéricos e as
chaves que diferem; no CSV, o formato e a maior diferença por coluna numérica. Colunas e chaves
de tempo de execução ("segundos", "duracao", datas) são ignoradas.

Uso (da raiz):
  .venv/bin/python codigo/comparar_reproducao.py REFERENCIA REPRODUCAO [--saida relatorio.md] [--tol 1e-6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

IGNORAR = re.compile(r"segundos|duracao|tempo|data|hora|timestamp|gerado_em|caminho|path|dispositivo|torch|cuda|python",
                     re.IGNORECASE)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def folhas(x, prefixo=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from folhas(v, f"{prefixo}.{k}" if prefixo else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from folhas(v, f"{prefixo}[{i}]")
    else:
        yield prefixo, x


def comparar_json(a: Path, b: Path, tol: float):
    fa = dict(folhas(json.loads(a.read_text(encoding="utf-8"))))
    fb = dict(folhas(json.loads(b.read_text(encoding="utf-8"))))
    so_a, so_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    maior, difs = 0.0, []
    for k in sorted(set(fa) & set(fb)):
        if IGNORAR.search(k):
            continue
        va, vb = fa[k], fb[k]
        if isinstance(va, bool) or isinstance(vb, bool) or not all(isinstance(v, (int, float)) for v in (va, vb)):
            if va != vb:
                difs.append((k, va, vb, None))
            continue
        if va is None or vb is None or (isinstance(va, float) and math.isnan(va)) and (isinstance(vb, float) and math.isnan(vb)):
            continue
        d = abs(va - vb)
        maior = max(maior, d)
        if d > tol:
            difs.append((k, va, vb, d))
    return maior, difs, so_a, so_b


def comparar_csv(a: Path, b: Path, tol: float):
    da, db = pd.read_csv(a, low_memory=False), pd.read_csv(b, low_memory=False)
    notas = []
    if da.shape != db.shape:
        notas.append(f"formato {da.shape} × {db.shape}")
    cols = [c for c in da.columns if c in db.columns and not IGNORAR.search(str(c))]
    if set(da.columns) != set(db.columns):
        notas.append(f"colunas diferentes: só ref {sorted(set(da.columns)-set(db.columns))}, só repr {sorted(set(db.columns)-set(da.columns))}")
    n = min(len(da), len(db))
    maior, por_coluna = 0.0, {}
    for c in cols:
        xa, xb = da[c].iloc[:n], db[c].iloc[:n]
        if pd.api.types.is_numeric_dtype(xa) and pd.api.types.is_numeric_dtype(xb):
            d = (xa.astype(float) - xb.astype(float)).abs()
            m = float(d.max(skipna=True)) if len(d) else 0.0
            if not math.isnan(m):
                maior = max(maior, m)
                if m > tol:
                    por_coluna[c] = m
        else:
            ndif = int((xa.astype(str) != xb.astype(str)).sum())
            if ndif:
                por_coluna[c] = f"{ndif} valores de texto diferentes"
    return maior, por_coluna, notas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("referencia", type=Path)
    ap.add_argument("reproducao", type=Path)
    ap.add_argument("--saida", type=Path)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    ref = {p.relative_to(args.referencia) for p in args.referencia.rglob("*") if p.suffix in {".json", ".csv"}}
    rep = {p.relative_to(args.reproducao) for p in args.reproducao.rglob("*") if p.suffix in {".json", ".csv"}}
    linhas = [f"# Reprodução: {args.reproducao} contra {args.referencia}", "",
              f"Arquivos só na referência: {len(ref - rep)}; só na reprodução: {len(rep - ref)}; em comum: {len(ref & rep)}.", ""]
    contagem = {"idêntico": 0, "dentro da tolerância": 0, "diferente": 0}
    detalhes = []
    for rel in sorted(ref & rep):
        a, b = args.referencia / rel, args.reproducao / rel
        if sha(a) == sha(b):
            contagem["idêntico"] += 1
            continue
        try:
            if rel.suffix == ".json":
                maior, difs, so_a, so_b = comparar_json(a, b, args.tol)
                ok = not difs and not so_a and not so_b
                info = [f"maior diferença numérica: {maior:.3g}"]
                if so_a or so_b:
                    info.append(f"chaves só na ref: {len(so_a)}, só na repr: {len(so_b)}")
                info += [f"  - `{k}`: {va!r} → {vb!r}" + (f" (Δ={d:.3g})" if d is not None else "") for k, va, vb, d in difs[:12]]
                if len(difs) > 12:
                    info.append(f"  - … e mais {len(difs) - 12}")
            else:
                maior, por_coluna, notas = comparar_csv(a, b, args.tol)
                ok = not por_coluna and not notas
                info = [f"maior diferença numérica: {maior:.3g}"] + notas + [
                    f"  - `{c}`: " + (f"Δmáx={v:.3g}" if isinstance(v, float) else v) for c, v in list(por_coluna.items())[:12]]
        except Exception as e:  # arquivo ilegível ou formato inesperado
            ok, info = False, [f"não comparado: {e}"]
        contagem["dentro da tolerância" if ok else "diferente"] += 1
        if not ok:
            detalhes.append((rel, info))
    linhas.append(f"Idênticos byte a byte: {contagem['idêntico']}; iguais dentro de {args.tol:g}: "
                  f"{contagem['dentro da tolerância']}; diferentes: {contagem['diferente']}.")
    linhas.append("")
    for rel, info in detalhes:
        linhas.append(f"## {rel}")
        linhas += info
        linhas.append("")
    if ref - rep:
        linhas.append("## Só na referência")
        linhas += [f"- {p}" for p in sorted(ref - rep)[:60]]
    texto = "\n".join(linhas)
    if args.saida:
        args.saida.write_text(texto, encoding="utf-8")
    print(texto[:6000])


if __name__ == "__main__":
    main()
