"""Números nacionais de derrame pleural nos microdados públicos do DATASUS.

Fontes (FTP público, sem cadastro):
  SIH/SUS, AIH reduzida (RD): ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados/RD<UF><AA><MM>.dbc
  SIM, declarações de óbito:  ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/DOBR<AAAA>.dbc

Códigos CID-10 (categorias):
  J90  derrame pleural não classificado em outra parte
  J91  derrame pleural em afecções classificadas em outra parte (inclui J91.0, derrame maligno)
  J86  piotórax (empiema)

SIH: conta internações (AIH aprovadas, sem as de longa permanência, IDENT = 5, como no TabNet) pelo
ano de processamento (competência), com o diagnóstico principal em cada categoria e, à parte, com
qualquer menção (principal ou secundários). Também óbitos hospitalares, valor total, permanência,
sexo e faixa etária.
SIM: óbitos com a categoria como causa básica e com qualquer menção na declaração (linhas A a D e
parte II).

Uso (da raiz do projeto):
  python codigo/numeros_datasus.py                     SIH 2024 e 2025, SIM 2024
  python codigo/numeros_datasus.py --anos-sih 2025 --anos-sim 2024
Requer: datasus-dbc (descompactação .dbc). Retomável: arquivos baixados e agregados por arquivo
ficam em cache.
"""
from __future__ import annotations

import argparse
import ftplib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "dataset" / "DATASUS"
HOST = "ftp.datasus.gov.br"
SIH_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados"
SIM_DIR = "/dissemin/publicos/SIM/CID10/DORES"
UFS = ("AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO").split()
CATEGORIAS = ("J90", "J91", "J86")
FAIXAS = [(0, 18, "0-17"), (18, 50, "18-49"), (50, 65, "50-64"), (65, 200, "65+")]


# ---------------------------------------------------------------- download

def ftp_size(ftp: ftplib.FTP, path: str) -> int | None:
    try:
        return ftp.size(path)
    except ftplib.error_perm:
        return None


def baixar(remoto: str, destino: Path, tentativas: int = 4) -> str:
    destino.parent.mkdir(parents=True, exist_ok=True)
    for t in range(tentativas):
        try:
            with ftplib.FTP(HOST, timeout=120) as ftp:
                ftp.login()
                ftp.voidcmd("TYPE I")
                tamanho = ftp_size(ftp, remoto)
                if tamanho is None:
                    return "ausente"
                if destino.exists() and destino.stat().st_size == tamanho:
                    return "cache"
                tmp = destino.with_suffix(".part")
                with open(tmp, "wb") as fh:
                    ftp.retrbinary(f"RETR {remoto}", fh.write, blocksize=1 << 16)
                if tmp.stat().st_size != tamanho:
                    raise IOError("tamanho divergente")
                tmp.replace(destino)
                return "baixado"
        except (OSError, EOFError, ftplib.Error) as erro:
            if t == tentativas - 1:
                raise RuntimeError(f"{remoto}: {erro}") from erro
            time.sleep(5 * (t + 1))
    return "falhou"


# ---------------------------------------------------------------- leitura DBF de largura fixa

def ler_dbf(conteudo: bytes, campos: list[str]) -> dict[str, np.ndarray]:
    """Lê só as colunas pedidas de um DBF (registros de largura fixa) como arrays de bytes."""
    n = int.from_bytes(conteudo[4:8], "little")
    cab = int.from_bytes(conteudo[8:10], "little")
    largura = int.from_bytes(conteudo[10:12], "little")
    posicoes, pos, i = {}, 1, 32
    while conteudo[i] != 0x0D:
        nome = conteudo[i:i + 11].split(b"\x00")[0].decode("ascii")
        tam = conteudo[i + 16]
        posicoes[nome] = (pos, tam)
        pos += tam
        i += 32
    registros = np.frombuffer(conteudo, dtype=np.uint8, count=n * largura, offset=cab).reshape(n, largura)
    validos = registros[:, 0] != ord("*")
    saida = {}
    for campo in campos:
        if campo not in posicoes:
            continue
        ini, tam = posicoes[campo]
        bloco = np.ascontiguousarray(registros[validos, ini:ini + tam])
        saida[campo] = np.char.strip(bloco.view(f"S{tam}").ravel())
    return saida


def descompactar(caminho: Path) -> bytes:
    import datasus_dbc
    return datasus_dbc.decompress_bytes(caminho.read_bytes())


def categoria3(codigos: np.ndarray) -> np.ndarray:
    return np.char.upper(np.char.ljust(codigos, 3).astype("S3"))


def para_int(valores: np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(valores.astype(str)), errors="coerce").fillna(-1).to_numpy()


# ---------------------------------------------------------------- SIH

CAMPOS_SIH = ["IDENT", "DIAG_PRINC", "DIAG_SECUN"] + [f"DIAGSEC{k}" for k in range(1, 10)] + [
    "MORTE", "VAL_TOT", "DIAS_PERM", "SEXO", "IDADE", "COD_IDADE"]


def faixa_etaria(idade: np.ndarray, cod: np.ndarray) -> np.ndarray:
    anos = np.where(cod == b"4", idade, np.where(cod == b"5", 100 + idade, 0))
    rotulos = np.full(len(anos), "?", dtype=object)
    for lo, hi, nome in FAIXAS:
        rotulos[(anos >= lo) & (anos < hi)] = nome
    return rotulos


def agregar_sih(caminho: Path) -> dict:
    d = ler_dbf(descompactar(caminho), CAMPOS_SIH)
    manter = d["IDENT"] != b"5"
    d = {k: v[manter] for k, v in d.items()}
    principal = categoria3(d["DIAG_PRINC"])
    secundarios = [categoria3(d[c]) for c in ["DIAG_SECUN"] + [f"DIAGSEC{k}" for k in range(1, 10)] if c in d]
    morte = d["MORTE"] == b"1"
    valor = pd.to_numeric(pd.Series(d["VAL_TOT"].astype(str)), errors="coerce").fillna(0).to_numpy()
    dias = para_int(d["DIAS_PERM"])
    faixa = faixa_etaria(para_int(d["IDADE"]), d["COD_IDADE"])
    sexo = np.where(d["SEXO"] == b"1", "M", np.where(np.isin(d["SEXO"], [b"2", b"3"]), "F", "?"))

    res = {"arquivo": caminho.name, "internacoes_total": int(manter.sum()),
           "obitos_total": int(morte.sum()), "categorias": {}}
    alguma_principal = np.zeros(len(principal), bool)
    alguma_mencao = np.zeros(len(principal), bool)
    for cat in CATEGORIAS + ("J90+J91+J86",):
        alvo = cat.split("+") if "+" in cat else [cat]
        alvo_b = [a.encode() for a in alvo]
        p = np.isin(principal, alvo_b)
        m = p.copy()
        for s in secundarios:
            m |= np.isin(s, alvo_b)
        res["categorias"][cat] = {
            "principal": int(p.sum()),
            "qualquer_mencao": int(m.sum()),
            "obitos_principal": int((p & morte).sum()),
            "obitos_qualquer_mencao": int((m & morte).sum()),
            "valor_total_principal": float(valor[p].sum()),
            "dias_perm_principal": int(dias[p].clip(min=0).sum()),
            "sexo_principal": {s: int((p & (sexo == s)).sum()) for s in ("F", "M", "?")},
            "faixa_principal": {f: int((p & (faixa == f)).sum()) for _, _, f in FAIXAS + [(0, 0, "?")]},
        }
    return res


# ---------------------------------------------------------------- SIM

CAMPOS_SIM = ["CAUSABAS", "LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII", "SEXO", "IDADE"]
PADRAO = {cat: re.compile(rb"\*?" + cat.encode()) for cat in CATEGORIAS}


def agregar_sim(caminho: Path) -> dict:
    d = ler_dbf(descompactar(caminho), CAMPOS_SIM)
    basica = categoria3(d["CAUSABAS"])
    linhas = [d[c] for c in ("LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII") if c in d]
    texto = linhas[0]
    for l in linhas[1:]:
        texto = np.char.add(np.char.add(texto, b"|"), l)
    idade = d["IDADE"]
    unidade = np.array([v[:1] for v in idade])
    valor = para_int(np.array([v[1:] for v in idade]))
    anos = np.where(unidade == b"4", valor, np.where(unidade == b"5", 100 + valor, 0))
    sexo = np.where(d["SEXO"] == b"1", "M", np.where(d["SEXO"] == b"2", "F", "?"))
    res = {"arquivo": caminho.name, "obitos_total": len(basica), "categorias": {}}
    for cat in CATEGORIAS + ("J90+J91+J86",):
        alvo = cat.split("+") if "+" in cat else [cat]
        b = np.isin(basica, [a.encode() for a in alvo])
        m = b.copy()
        for a in alvo:
            m |= np.char.find(texto, a.encode()) >= 0
        res["categorias"][cat] = {
            "causa_basica": int(b.sum()),
            "qualquer_mencao": int(m.sum()),
            "sexo_causa_basica": {s: int((b & (sexo == s)).sum()) for s in ("F", "M", "?")},
            "faixa_causa_basica": {f: int((b & (anos >= lo) & (anos < hi)).sum()) for lo, hi, f in FAIXAS},
        }
    return res


# ---------------------------------------------------------------- orquestração

def processar(tipo: str, caminho: Path) -> dict:
    cache = BASE / "agregados" / f"{caminho.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    res = agregar_sih(caminho) if tipo == "sih" else agregar_sim(caminho)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def somar(lista: list[dict]) -> dict:
    def soma(a, b):
        if isinstance(a, dict):
            return {k: soma(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b)}
        if isinstance(a, str):
            return a
        return a + b
    total = lista[0]
    for r in lista[1:]:
        total = soma(total, r)
    total.pop("arquivo", None)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anos-sih", type=int, nargs="*", default=[2024, 2025])
    ap.add_argument("--anos-sim", type=int, nargs="*", default=[2024])
    ap.add_argument("--downloads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    tarefas = []
    for ano in args.anos_sih:
        for uf in UFS:
            for mes in range(1, 13):
                nome = f"RD{uf}{ano % 100:02d}{mes:02d}.dbc"
                tarefas.append(("sih", ano, f"{SIH_DIR}/{nome}", BASE / "fonte" / "SIH" / nome))
    for ano in args.anos_sim:
        nome = f"DOBR{ano}.dbc"
        tarefas.append(("sim", ano, f"{SIM_DIR}/{nome}", BASE / "fonte" / "SIM" / nome))

    print(f"Baixando {len(tarefas)} arquivos...", flush=True)
    status = {}
    with ThreadPoolExecutor(args.downloads) as pool:
        futuros = {pool.submit(baixar, t[2], t[3]): t for t in tarefas}
        for i, f in enumerate(as_completed(futuros), 1):
            status[futuros[f][3].name] = f.result()
            if i % 50 == 0 or i == len(tarefas):
                print(f"  {i}/{len(tarefas)}", flush=True)
    ausentes = [n for n, s in status.items() if s == "ausente"]
    if ausentes:
        print("Ausentes no FTP:", ausentes)

    print("Agregando...", flush=True)
    resultados = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futuros = {pool.submit(processar, t[0], t[3]): t for t in tarefas if status[t[3].name] != "ausente"}
        for i, f in enumerate(as_completed(futuros), 1):
            tipo, ano = futuros[f][:2]
            resultados.setdefault((tipo, ano), []).append(f.result())
            if i % 50 == 0 or i == len(futuros):
                print(f"  {i}/{len(futuros)}", flush=True)

    resumo = {"data_acesso": date.today().isoformat(),
              "fontes": {"SIH": f"ftp://{HOST}{SIH_DIR}", "SIM": f"ftp://{HOST}{SIM_DIR}"},
              "arquivos_ausentes": ausentes, "SIH": {}, "SIM": {}}
    for (tipo, ano), lista in sorted(resultados.items()):
        total = somar(lista)
        total["n_arquivos"] = len(lista)
        resumo[tipo.upper()][str(ano)] = total
    saida = BASE / "resultados" / "resumo_derrame_pleural.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    for ano, r in resumo["SIH"].items():
        print(f"\nSIH {ano} ({r['n_arquivos']} arquivos): {r['internacoes_total']:,} internações")
        for cat, c in sorted(r["categorias"].items()):
            print(f"  {cat:12s} principal {c['principal']:>7,}  menção {c['qualquer_mencao']:>7,}  "
                  f"óbitos(princ) {c['obitos_principal']:>6,}  R$ {c['valor_total_principal']:>14,.0f}")
    for ano, r in resumo["SIM"].items():
        print(f"\nSIM {ano}: {r['obitos_total']:,} óbitos")
        for cat, c in sorted(r["categorias"].items()):
            print(f"  {cat:12s} causa básica {c['causa_basica']:>6,}  qualquer menção {c['qualquer_mencao']:>7,}")
    print(f"\n-> {saida}")


if __name__ == "__main__":
    main()
