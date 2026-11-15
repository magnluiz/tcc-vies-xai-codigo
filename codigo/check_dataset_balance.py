"""
check_dataset_balance.py

Script generico para checar balanceamento de classes e de subgrupos
demograficos (raca/tom de pele, idade, genero, classe economica etc.)
em qualquer dataset que tenha um CSV de metadados.

USO:
    python check_dataset_balance.py --csv caminho/metadados.csv \
        --class-col diagnostic \
        --group-cols sex,age,fitzpatrick \
        --intersection-cols sex,fitzpatrick \
        --id-col patient_id \
        --out-dir ./relatorio_dataset

O QUE ELE FAZ:
    1. Conta quantas amostras existem por classe (ex: cada tipo de lesao).
     2. Relata dados ausentes e a distribuicao de cada coluna informada.
     3. Cruza classe x grupo e, opcionalmente, grupos interseccionais.
     4. Quando informado, conta identificadores unicos (ex: pacientes) por grupo.
     5. Calcula entropia normalizada e razao maior/menor entre categorias observadas.
     6. Salva tabelas CSV e graficos na pasta de saida.

NAO PRECISA DE PYTORCH/TENSORFLOW - roda so com pandas/matplotlib,
entao funciona antes mesmo de voce treinar qualquer modelo.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalized_entropy(counts: pd.Series) -> float:
    """Entropia normalizada; NaN quando ha menos de duas categorias observadas."""
    counts = counts[counts > 0]
    if len(counts) < 2:
        return float("nan")
    probs = counts / counts.sum()
    entropy = -(probs * np.log2(probs)).sum()
    max_entropy = np.log2(len(counts))
    return float(entropy / max_entropy)


def imbalance_ratio(counts: pd.Series) -> float:
    """Quantas vezes o maior grupo é maior que o menor grupo."""
    counts = counts[counts > 0]
    if len(counts) < 2:
        return float("nan")
    return float(counts.max() / counts.min())


def analyze_column(df: pd.DataFrame, col: str, out_dir: str) -> pd.DataFrame:
    counts = df[col].value_counts(dropna=True)
    pct = (counts / counts.sum() * 100).round(2) if counts.sum() else counts
    summary = pd.DataFrame({"contagem": counts, "percentual (%)": pct})
    missing = int(df[col].isna().sum())

    print(f"\n=== Distribuição de '{col}' ===")
    print(summary)
    print(f"Valores ausentes: {missing} ({missing / len(df) * 100:.2f}%)")
    entropy = normalized_entropy(counts)
    ratio = imbalance_ratio(counts)
    print(f"Entropia normalizada: {entropy:.3f}" if pd.notna(entropy) else
          "Entropia normalizada: nao aplicavel (menos de duas categorias)")
    print(f"Razao maior/menor grupo: {ratio:.2f}x" if pd.notna(ratio) else
          "Razao maior/menor grupo: nao aplicavel (menos de duas categorias)")

    summary.to_csv(os.path.join(
        out_dir, f"distribuicao_{col}.csv"), index_label=col)

    # Salva gráfico de barras
    plt.figure(figsize=(8, 4))
    counts.sort_values(ascending=False).plot(kind="bar")
    plt.title(f"Distribuição de '{col}'")
    plt.ylabel("Nº de amostras")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"grafico_{col}.png"), dpi=150)
    plt.close()

    return summary


def cross_tab(df: pd.DataFrame, class_col: str, group_col: str, out_dir: str,
              id_col: str | None = None):
    valid = df[[class_col, group_col]].dropna()
    ct = pd.crosstab(valid[class_col], valid[group_col])
    pct_by_group = pd.crosstab(
        valid[class_col], valid[group_col], normalize="columns").round(3) * 100
    pct_by_class = pd.crosstab(
        valid[class_col], valid[group_col], normalize="index").round(3) * 100

    print(f"\n=== Cruzamento: {class_col} x {group_col} (contagem) ===")
    print(ct)
    print(
        f"\n=== Classe (%) dentro de cada grupo: {class_col} x {group_col} ===")
    print(pct_by_group)
    print(
        f"\n=== Grupo (%) dentro de cada classe: {class_col} x {group_col} ===")
    print(pct_by_class)

    ct.to_csv(os.path.join(out_dir, f"cruzamento_{class_col}_x_{group_col}_contagem.csv"),
              index_label=class_col)
    pct_by_group.to_csv(
        os.path.join(
            out_dir, f"cruzamento_{class_col}_x_{group_col}_classe_por_grupo.csv"),
        index_label=class_col,
    )
    pct_by_class.to_csv(
        os.path.join(
            out_dir, f"cruzamento_{class_col}_x_{group_col}_grupo_por_classe.csv"),
        index_label=class_col,
    )

    if id_col:
        subject_counts = (
            df[[class_col, group_col, id_col]]
            .dropna(subset=[class_col, group_col])
            .groupby([class_col, group_col], dropna=False)
            .agg(n_imagens=(id_col, "size"), n_pacientes=(id_col, "nunique"))
            .reset_index()
        )
        subject_counts.to_csv(
            os.path.join(out_dir, f"pacientes_{class_col}_x_{group_col}.csv"), index=False
        )

    if not pct_by_group.empty:
        plt.figure(figsize=(8, 5))
        plt.imshow(pct_by_group.to_numpy(), aspect="auto", cmap="Reds")
        plt.colorbar(label="Classe (%) dentro do grupo")
        plt.xticks(range(len(pct_by_group.columns)), list(map(str, pct_by_group.columns)),
                   rotation=45, ha="right")
        plt.yticks(range(len(pct_by_group.index)),
                   list(map(str, pct_by_group.index)))
        plt.title(f"{class_col} x {group_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(
            out_dir, f"heatmap_{class_col}_x_{group_col}.png"), dpi=150)
        plt.close()


def analyze_intersection(df: pd.DataFrame, class_col: str, group_cols: list[str],
                         out_dir: str, id_col: str | None = None):
    valid = df[[class_col, *group_cols]].dropna()
    group_values = [valid[col] for col in group_cols]
    counts = pd.crosstab(valid[class_col], group_values)
    pct_by_class = pd.crosstab(
        valid[class_col], group_values, normalize="index").round(3) * 100
    suffix = "_x_".join(group_cols)

    print(f"\n=== Intersecao: {class_col} x {suffix} (contagem) ===")
    print(counts)
    counts.to_csv(os.path.join(out_dir, f"intersecao_{class_col}_x_{suffix}_contagem.csv"),
                  index_label=class_col)
    pct_by_class.to_csv(
        os.path.join(
            out_dir, f"intersecao_{class_col}_x_{suffix}_percentual_por_classe.csv"),
        index_label=class_col,
    )

    if id_col:
        subject_counts = (
            df[[class_col, *group_cols, id_col]]
            .dropna(subset=[class_col, *group_cols])
            .groupby([class_col, *group_cols], dropna=False)
            .agg(n_imagens=(id_col, "size"), n_pacientes=(id_col, "nunique"))
            .reset_index()
        )
        subject_counts.to_csv(
            os.path.join(
                out_dir, f"intersecao_{class_col}_x_{suffix}_pacientes.csv"),
            index=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Checa balanceamento de classes e subgrupos demográficos.")
    parser.add_argument("--csv", required=True,
                        help="Caminho para o CSV de metadados do dataset.")
    parser.add_argument("--class-col", required=True,
                        help="Nome da coluna com a classe/diagnóstico/label.")
    parser.add_argument("--group-cols", required=True,
                        help="Colunas demográficas separadas por vírgula (ex: sex,age,fitzpatrick).")
    parser.add_argument("--intersection-cols",
                        help="Duas ou mais colunas de grupo para cruzamento interseccional.")
    parser.add_argument(
        "--id-col", help="Coluna de identificador unico (ex: patient_id).")
    parser.add_argument("--out-dir", default="./relatorio_balanceamento",
                        help="Pasta para salvar os resultados.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    group_cols = list(dict.fromkeys(col.strip()
                      for col in args.group_cols.split(",") if col.strip()))
    intersection_cols = (
        list(dict.fromkeys(col.strip()
             for col in args.intersection_cols.split(",") if col.strip()))
        if args.intersection_cols else []
    )
    requested_cols = [args.class_col, *group_cols, *intersection_cols]
    if args.id_col:
        requested_cols.append(args.id_col)
    missing_cols = [col for col in requested_cols if col not in df.columns]
    if missing_cols:
        parser.error(
            f"Colunas nao encontradas no CSV: {', '.join(missing_cols)}")
    if not group_cols:
        parser.error("Informe ao menos uma coluna em --group-cols.")
    if args.class_col in group_cols or args.class_col in intersection_cols:
        parser.error(
            "A coluna de classe nao pode tambem ser uma coluna de grupo.")
    if intersection_cols and (len(intersection_cols) < 2 or
                              not set(intersection_cols).issubset(group_cols)):
        parser.error(
            "--intersection-cols exige ao menos duas colunas presentes em --group-cols.")
    if args.id_col and args.id_col in requested_cols[:-1]:
        parser.error("A coluna de identificador nao pode ser classe ou grupo.")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Dataset carregado: {len(df)} linhas, colunas: {list(df.columns)}")
    missingness = pd.DataFrame({
        "ausentes": df.isna().sum(),
        "ausentes (%)": (df.isna().mean() * 100).round(2),
    })
    print("\n=== Dados ausentes por coluna ===")
    print(missingness)
    missingness.to_csv(os.path.join(
        args.out_dir, "dados_ausentes.csv"), index_label="coluna")

    analyze_column(df, args.class_col, args.out_dir)
    for col in group_cols:
        analyze_column(df, col, args.out_dir)
        cross_tab(df, args.class_col, col, args.out_dir, args.id_col)

    if intersection_cols:
        analyze_intersection(df, args.class_col,
                             intersection_cols, args.out_dir, args.id_col)

    print(f"\nRelatório completo salvo em: {args.out_dir}")


if __name__ == "__main__":
    main()
