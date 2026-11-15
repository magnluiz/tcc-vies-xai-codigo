# XAI como auditoria de viés controlado — código

Código do Trabalho de Conclusão de Curso (UnB) que investiga **se métodos de explicabilidade (XAI) conseguem detectar um viés inserido de forma controlada** em classificadores de imagens médicas.

O trabalho tem duas fases:

| Fase | Dataset | Tarefa | Papel |
|---|---|---|---|
| Piloto | **PAD-UFES-20** (lesões de pele, Brasil) | 6 diagnósticos dermatológicos | linha de base, subgrupos, artefato sintético e marcas de caneta reais |
| Principal | **NIH ChestX-ray14** (radiografias de tórax) | derrame pleural × sem achados | viés com intensidade *q* conhecida: sexo (exp. A) e quadrado xadrez (exp. B) |

Em ambas, o modelo é uma **ResNet-18** pré-treinada no ImageNet. As explicações avaliadas são **Grad-CAM**, **Integrated Gradients (IG)** e **TCAV**, sempre acompanhadas de testes de sanidade (randomização de pesos e rótulos embaralhados).

## Sumário

1. [Conteúdo do repositório](#1-conteúdo-do-repositório)
2. [Ambiente](#2-ambiente)
3. [Estrutura de pastas esperada](#3-estrutura-de-pastas-esperada)
4. [PAD-UFES-20 (piloto)](#4-pad-ufes-20-piloto)
5. [NIH ChestX-ray14 (fase principal)](#5-nih-chestx-ray14-fase-principal)
6. [Referência de cada script](#6-referência-de-cada-script)
7. [Interromper e retomar](#7-interromper-e-retomar)
8. [Travas do protocolo](#8-travas-do-protocolo)
9. [Problemas comuns](#9-problemas-comuns)

---

## 1. Conteúdo do repositório

```text
.
├── README.md
├── requirements.txt
├── codigo/                     25 scripts Python (seção 6). Importam uns aos outros: mantenha todos juntos.
├── protocolo/                  fichas de protocolo pré-registradas (cópias byte a byte, ver seção 8)
│   ├── ficha_protocolo_experimental.md       PAD-UFES-20
│   └── ficha_protocolo_experimental_nih.md   NIH ChestX-ray14
└── anotacoes/
    └── anotacao_marcas_caneta.csv            anotação visual das marcas de caneta (PAD-UFES-20)
```

**Não estão no repositório:** os dados, os caches, os resultados (pasta `dataset/`, dezenas de GB) e os pesos dos modelos. Os dados são públicos e abertos; a seção 4.1 e a seção 5.1 explicam como obtê-los.

**Convenções de todos os scripts:**
- Rodam **a partir da raiz do repositório**: `.venv\Scripts\python.exe codigo\<script>.py`.
- Cada script define `ROOT` como a pasta acima de `codigo/` e lê/grava em `ROOT/dataset/...`. Os caminhos padrão funcionam sem argumentos se a estrutura da seção 3 for seguida.
- Sementes fixas (42 por padrão; réplica com 47–51). Saídas em CSV/JSON (UTF-8) e figuras em PNG.
- Comandos escritos para **PowerShell no Windows**. Em Linux/macOS, troque `.venv\Scripts\python.exe` por `.venv/bin/python` e `\` por `/`.

---

## 2. Ambiente

**Requisitos:**
- Python 3.12.
- GPU NVIDIA com driver compatível com CUDA 13.2 (o trabalho usou uma RTX 3060 Ti de 8 GB).
- Cerca de 60 GB livres em disco, quase todos para o NIH (45 GB de pacotes + 7 GB de cache).

**Os scripts de treino e de XAI exigem GPU** e param com erro se não houver CUDA. Auditoria, preparação de coortes e análises a partir de resultados gravados rodam sem GPU.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
# PyTorch com CUDA (índice do PyTorch, não o PyPI):
.venv\Scripts\python.exe -m pip install "torch==2.14.1+cu132" "torchvision==0.29.1+cu132" --index-url https://download.pytorch.org/whl/cu132
# Demais bibliotecas, nas versões fixadas:
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Conferir a GPU:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# esperado: 2.14.1+cu132 True NVIDIA GeForce RTX 3060 Ti (ou a sua GPU)
```

> Os comandos usam `.venv\Scripts\python.exe` diretamente, então não é preciso ativar o ambiente.

---

## 3. Estrutura de pastas esperada

Depois de baixar os dados e rodar os scripts, a pasta `dataset/` (ignorada pelo Git) fica assim:

```text
dataset/
├── PAD-UFES-20/
│   ├── zr7vgbcyr2-1.zip          pacote original (Mendeley Data)
│   ├── zr7vgbcyr2-1/             pacote extraído: metadata.csv e images/
│   ├── artefatos_protocolo/      auditoria, partições e ficha de protocolo
│   └── derivados_analise_v1/
│       ├── coorte_principal/, coorte_sensibilidade/   metadados e partições de cada coorte
│       ├── cache_imagens_256/
│       ├── smoke_test/
│       ├── experimentos_v1/      treinos, avaliação, Grad-CAM, artefato sintético
│       └── marcas_caneta/        anotação visual, experimento e revisão das marcas
└── NIH-ChestXray14/
    ├── fonte/                    metadados (CSV, listas) e zips/ com os 12 pacotes
    ├── auditoria/
    ├── cache_imagens_256/
    ├── derivados_v1/             coortes e máscaras torácicas
    ├── ficha_protocolo_experimental_nih.md
    └── experimentos_v1/          piloto/, execucoes/, xai/
```

**Arquivos que vêm deste repositório e precisam ser copiados para `dataset/`** (os scripts os procuram lá):

```powershell
New-Item -ItemType Directory -Force dataset\PAD-UFES-20\artefatos_protocolo, dataset\PAD-UFES-20\derivados_analise_v1\marcas_caneta, dataset\NIH-ChestXray14 | Out-Null
Copy-Item protocolo\ficha_protocolo_experimental.md     dataset\PAD-UFES-20\artefatos_protocolo\
Copy-Item protocolo\ficha_protocolo_experimental_nih.md dataset\NIH-ChestXray14\
Copy-Item anotacoes\anotacao_marcas_caneta.csv          dataset\PAD-UFES-20\derivados_analise_v1\marcas_caneta\
```

---

## 4. PAD-UFES-20 (piloto)

### 4.1 Obter os dados

1. Baixar o PAD-UFES-20 no Mendeley Data (DOI `10.17632/zr7vgbcyr2.1`, licença CC BY 4.0) e salvar como `dataset/PAD-UFES-20/zr7vgbcyr2-1.zip`.
2. Extrair em `dataset/PAD-UFES-20/zr7vgbcyr2-1/`. Devem ficar `metadata.csv` e `images/`; dentro de `images/` há partes (`imgs_part_1`, …) que também precisam ser extraídas.

### 4.2 Sequência de execução

Rodar na ordem; cada passo usa o que o anterior gerou. Caminhos de saída relativos a `dataset/PAD-UFES-20/`.

| # | Comando (`.venv\Scripts\python.exe codigo\...`) | O que faz | Saída principal | GPU | Tempo |
|---|---|---|---|:---:|---|
| 1 | `auditar_imagens_pad_ufes.py` | Confere CSV ↔ arquivos, decodifica todas as imagens, mede dimensões e calcula hashes (detecta duplicatas) | `artefatos_protocolo/auditoria_imagens.csv`, `verificacao_imagens.json` | não | ~1 min |
| 2 | `comparar_duplicatas_pad_ufes_zip.py` | Compara os hashes auditados com as imagens dentro do ZIP original | `artefatos_protocolo/comparacao_zip_original.json` | não | ~1 min |
| 3 | `gerar_artefatos_experimentos.py` | Auditoria por paciente e partições: teste separado por paciente e 5 folds com `StratifiedGroupKFold` (semente 42) | `artefatos_protocolo/particoes_pacientes.csv`, `pacientes_validacao_cruzada.csv`, `verificacao_particoes.json` | não | segundos |
| 4 | `preparar_coortes_pad_ufes.py` | Cria as coortes principal e de sensibilidade, aplicando as decisões sobre duplicatas e registros inválidos sem alterar a fonte | `derivados_analise_v1/coorte_*/metadata_analise.csv`, `decisoes_coortes.csv`, `manifesto_coortes.json` | não | segundos |
| 5 | ver abaixo | Partições de cada coorte | `coorte_*/particoes/` | não | segundos |
| 6 | `smoke_test_pipeline_pad_ufes.py` | Teste técnico: um lote passa pela ResNet-18, perda e passo do otimizador funcionam (não mede desempenho) | `derivados_analise_v1/smoke_test/model_pipeline_smoke.json` | opcional | segundos |
| 7 | `preparar_cache_imagens_pad_ufes.py` | Pré-processa as imagens das duas coortes uma vez (RGB, menor lado 256, recorte central 256 × 256) | `derivados_analise_v1/cache_imagens_256/` (2.274 imagens, ~450 MB) | não | ~20 s |
| 8 | `treinar_pad_ufes.py cv` | Validação cruzada agrupada (5 folds) comparando entropia cruzada padrão × ponderada; escolhe a configuração e o número de épocas | `experimentos_v1/cv_principal/decisao_selecao.json` | **sim** | minutos |
| 9 | `treinar_pad_ufes.py final` | Treina no conjunto de desenvolvimento inteiro e prediz o teste **uma única vez** | `experimentos_v1/final_principal/predicoes_teste.csv`, `modelo_final.pt` | **sim** | minutos |
| 10 | `treinar_pad_ufes.py final --coorte sensibilidade` | O mesmo para a coorte de sensibilidade | `experimentos_v1/final_sensibilidade/` | **sim** | minutos |
| 11 | `avaliar_pad_ufes.py` (e `--coorte sensibilidade`) | Métricas do teste a partir das predições gravadas: macro-F1, por classe, subgrupos (sexo, fototipo, idade), bootstrap por paciente | `final_*/avaliacao/` | não | segundos |
| 12 | `gradcam_pad_ufes.py` | Grad-CAM no modelo final + sanidade (randomização em cascata e rótulos embaralhados) | `final_principal/gradcam/` | **sim** | minutos |
| 13 | `artefato_sintetico_pad_ufes.py` | Quadrado verde correlacionado com NEV, 4 níveis de *q* × 5 sementes; dependência real × sinal do Grad-CAM e do IG | `experimentos_v1/artefato_sintetico/` | **sim** | ~1 h |
| 14 | `marcas_caneta_pad_ufes.py --raio-central 56` | Marcas de caneta reais: apaga a tinta por *inpainting* e mede a mudança na predição, com controle placebo | `marcas_caneta/experimento_fora_do_centro_r56/` | **sim** | minutos |
| 15 | `revisao_marcas_caneta_pad_ufes.py gerar` → `... analisar <csv>` | Conferência manual da anotação: página HTML local com 717 imagens às cegas (teclas S/N/D, exporta CSV); depois calcula concordância, kappa e precisão da anotação original | `marcas_caneta/revisao/revisar.html`, `amostra_revisao.csv`, `concordancia.json` | não | — |

**Passo 5, partições das coortes.** A coorte de sensibilidade herda a divisão da principal, para que cada paciente caia na mesma partição nas duas:

```powershell
$d = "dataset\PAD-UFES-20\derivados_analise_v1"
.venv\Scripts\python.exe codigo\gerar_artefatos_experimentos.py --csv "$d\coorte_principal\metadata_analise.csv" --out-dir "$d\coorte_principal\particoes"
.venv\Scripts\python.exe codigo\gerar_artefatos_experimentos.py --csv "$d\coorte_sensibilidade\metadata_analise.csv" --out-dir "$d\coorte_sensibilidade\particoes" --reference-splits "$d\coorte_principal\particoes"
```

**Observações:**
- **Passo 12:** `--epocas-rotulos-aleatorios N` muda o número de épocas do modelo com rótulos embaralhados (padrão: o mesmo do treino final).
- **Passo 14:** `--raio-central 56` exclui o disco central da máscara de tinta, porque a máscara completa capturava pixels violáceos da própria lesão. Sem a opção, roda a versão com máscara completa (`marcas_caneta/experimento/`). `--so-analise` refaz só o resumo e as figuras a partir do `por_imagem.csv` gravado. Na primeira execução, o script treina e guarda os modelos por fold em `experimentos_v1/modelos_fold_receita_final/`; nas seguintes, reaproveita-os.
- **Anotação das marcas de caneta:** foi feita visualmente, imagem por imagem, e **não é gerada por script**. Vem em `anotacoes/anotacao_marcas_caneta.csv` (copiar conforme a seção 3). Os passos 14 e 15 dependem dela.

---

## 5. NIH ChestX-ray14 (fase principal)

### 5.1 Obter os dados

Fonte: espelho público no Hugging Face `alkzar90/NIH-Chest-X-ray-dataset` (não precisa de login). Dados do NIH Clinical Center, em domínio público (CC0); citar Wang et al. (CVPR 2017) e reconhecer o NIH Clinical Center.

```powershell
New-Item -ItemType Directory -Force dataset\NIH-ChestXray14\fonte\zips | Out-Null
$base = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"
foreach ($f in "Data_Entry_2017_v2020.csv", "train_val_list.txt", "test_list.txt", "BBox_List_2017.csv") {
    curl.exe -L -o "dataset\NIH-ChestXray14\fonte\$f" "$base/$f"
}
# 12 pacotes de ~2 GB cada (~45 GB). O -C - retoma um download interrompido.
foreach ($i in 1..12) {
    $z = "images_{0:D3}.zip" -f $i
    curl.exe -L -C - -o "dataset\NIH-ChestXray14\fonte\zips\$z" "$base/images/$z"
}
```

- Se um pacote ficar lento, interrompa e rode o laço de novo: o `-C -` continua de onde parou.
- SHA-256 esperado do CSV de metadados: `dc1d2df67fdc1c5a7601d48699cda2b13dc2c4841488b4183dcf04884dbaca11`.
- O SHA-256 dos 12 pacotes é conferido automaticamente no passo 2.

### 5.2 Desenho experimental (resumo da ficha)

- Coorte: só incidência PA, **derrame pleural × sem achados**, uma imagem por paciente, divisão oficial treino/teste do NIH (sem paciente compartilhado) e validação de 15% estratificada por rótulo × sexo.
- Cada treino sorteia 1.200 positivos e 2.400 negativos; E = 9 épocas (fixado pelo piloto).
- 2 experimentos × 4 níveis de *q* × 5 sementes = 40 treinos, mais um modelo com rótulos embaralhados para sanidade. A réplica confirmatória repete os 40 com as sementes 47–51.

| | Experimento A: viés não localizado | Experimento B: viés localizado |
|---|---|---|
| Atalho no treino | sexo: *q* = P(homem \| derrame) = P(mulher \| sem achados) | quadrado xadrez 20 × 20 num canto, em *q* dos positivos e 1 − *q* dos negativos |
| Viés real medido | diferença de sensibilidade H − M; acurácia do pior grupo | fração dos negativos do teste que viram "derrame" quando o quadrado é inserido |
| Sinal por pixel | concentração do Grad-CAM e do IG na região torácica; diferença entre os mapas médios de mulheres e homens (sinal H/M) | concentração no quadrado |
| Sinal por conceito | TCAV do conceito "masculino" (layer3 e layer4) contra 20 CAVs aleatórios | o mesmo TCAV, como controle negativo |

### 5.3 Sequência de execução

Caminhos de saída relativos a `dataset/NIH-ChestXray14/`.

| # | Comando (`.venv\Scripts\python.exe codigo\...`) | O que faz | Saída principal | GPU | Tempo |
|---|---|---|---|:---:|---|
| 1 | `auditar_nih_chestxray14.py` | Auditoria só dos metadados: partições oficiais, sexo, idade, incidência e achados por sexo | `auditoria/` | não | segundos |
| 2 | `preparar_cache_nih_chestxray14.py` | **Confere o SHA-256 dos 12 pacotes** com os hashes publicados no HF (precisa de internet), lê os PNG direto dos ZIPs, converte para tons de cinza e grava o cache 256 × 256 | `cache_imagens_256/imagens_256.npy` (112.120 imagens, ~7 GB), `indice.csv`, `manifesto_cache.json` | não | dezenas de min |
| 3 | `preparar_coortes_nih_chestxray14.py` | Coorte principal (regras da seção 5.2) | `derivados_v1/coorte_principal/coorte.csv` | não | segundos |
| 4 | `preparar_coortes_nih_chestxray14.py --derrame-isolado` | Coorte de sensibilidade: positivos só com "Effusion", sem outros achados | `derivados_v1/coorte_derrame_isolado/coorte.csv` | não | segundos |
| 5 | `mascaras_toracicas_nih.py` | Região torácica das imagens de teste: PSPNet do `torchxrayvision`, fecho convexo de pulmões + coração + mediastino, dilatado 10 px | `derivados_v1/mascaras_toracicas/` | **sim** | ~1 min |
| 6 | `treinar_nih_chestxray14.py piloto` | Treino piloto (*q* = 0,5, semente 42, até 20 épocas); fixa E pela maior AUC na validação, sem usar o teste | `experimentos_v1/piloto/decisao_epocas.json` | **sim** | ~3 min |
| 7 | `treinar_nih_chestxray14.py experimentos` | 40 treinos (A e B, 4 *q* × 5 sementes) + modelo de rótulos embaralhados | `experimentos_v1/execucoes/<execução>/` | **sim** | ~30 min |
| 8 | `predizer_validacao_nih.py` | Prediz a validação com cada modelo: teste alternativo para as medidas de viés, porque o teste oficial mistura o rótulo com a época de aquisição | `execucoes/<execução>/predicoes_validacao.csv` | **sim** | ~2 min |
| 9 | `xai_nih_chestxray14.py` | Grad-CAM, IG e TCAV em cada execução; sanidade; medidas de viés; Spearman com IC por bootstrap; figuras | `experimentos_v1/xai/` | **sim** | ~2 h |
| 10 | `xai_nih_chestxray14.py --so-analise` | Refaz só a análise e as figuras a partir dos resultados gravados | `xai/analise.json`, `xai/execucoes.csv`, figuras | não | ~1 min |
| 11 | `controle_dissimilaridade_nih.py` | Controles do sinal H/M: metades sorteadas (piso de ruído), idade (atributo não manipulado) e experimento B (controle negativo) | `xai/controle_dissimilaridade.csv` e `.json` | **sim** | ~40 min |
| 12 | `pranchas_nih.py` | Pranchas de mapas com imagens sorteadas (semente 42): exemplos do exp. A, mapas médios por sexo, exemplos do exp. B. Pula pranchas existentes (apague o PNG para refazer) | `xai/prancha_*.png` | **sim** | ~5 min |
| 13 | `treinar_nih_chestxray14.py replica` | Réplica confirmatória pré-registrada: os mesmos 40 treinos com sementes 47–51 | `execucoes/<execução>_s47` … `_s51` | **sim** | ~27 min |
| 14 | `xai_nih_chestxray14.py --conjunto replica` | Passo 9 para a réplica. Depois, rodar de novo os passos 8 e 11 (já incluem as sementes da réplica) e `xai_nih_chestxray14.py --conjunto combinado --so-analise` para as 10 sementes | `xai/analise_replica.json`, `xai/controle_dissimilaridade_replica.json` (decisão da hipótese confirmatória), idem `_combinado` | **sim** | ~2 h |
| 15 | `sensibilidade_nih.py` | Sensibilidades pré-registradas: S1, TCAV com 100 CAVs aleatórios; S2, IG com linha de base borrada (σ = 10 px). `--so-analise` refaz só o resumo | `xai/sensibilidade/resumo.json` | **sim** | ~3 h |
| 16 | `controle_condicionado_nih.py` | Sinal H/M com mulheres e homens **pareados pela probabilidade prevista** (e só nos verdadeiros positivos), nas 80 execuções. Exploratório, não pré-registrado | `xai/controle_condicionado.csv`, `controle_condicionado{,_replica,_combinado}.json` | **sim** | ~35 min |

**Observações:**
- **Passo 2:** `--workers N` controla quantos processos leem os ZIPs (padrão 12). Entradas `__MACOSX/._*.png` são ignoradas; 519 PNG em RGBA são convertidos para tons de cinza.
- **Passo 5:** o PSPNet baixa os pesos do GitHub na primeira execução e os guarda em `~/.torchxrayvision/`.
- **Passo 6:** o piloto **se recusa a rodar de novo** se `decisao_epocas.json` existir (seção 8).
- **Nomes das execuções:** `A_q0.75_s43` = experimento A, *q* = 0,75, semente 43; `B_q1_s42` = experimento B, *q* = 1,0, semente 42; `sanidade_rotulos_embaralhados` = modelo do teste de sanidade.
- **Conteúdo de cada `execucoes/<execução>/`:**

  | Arquivo | Conteúdo |
  |---|---|
  | `modelo.pt` | pesos do modelo |
  | `treino.csv` | as 3.600 imagens sorteadas, com sexo e presença do quadrado |
  | `historico.csv` | perda por época |
  | `predicoes_teste.csv` | P(derrame) no teste; no exp. B, também com o quadrado inserido e sua posição |
  | `predicoes_teste_derrame_isolado.csv` | predições na coorte de derrame isolado |
  | `predicoes_validacao.csv` | predições na validação (passo 8) |
  | `execucao.json` | resumo, gravado por último (marca a execução como concluída) |

- **Principais saídas de `xai/`:**

  | Arquivo | Conteúdo |
  |---|---|
  | `por_execucao/<execução>.json` e `_por_imagem.csv` | resultados de cada execução (servem de cache) |
  | `sanidade.json` | randomização em cascata e rótulos embaralhados, para Grad-CAM e IG |
  | `execucoes.csv` | uma linha por execução: viés real e sinais da XAI |
  | `analise.json` | verificação da manipulação, Spearman de cada sinal com *q* e com o viés real, TCAV × pixel, AUC por imagem, controle negativo |
  | `experimento_A.png`, `experimento_B.png` | figuras-resumo |

---

## 6. Referência de cada script

Todos ficam em `codigo/`. "Biblioteca" indica arquivos importados por outros scripts.

### Utilitários e bibliotecas compartilhadas

| Script | Função | Argumentos |
|---|---|---|
| `xai_atribuicoes.py` | **Biblioteca** (não roda sozinho). Implementa `GradCAM` (com *hooks* na camada-alvo), `integrated_gradients` (linha de base configurável, *n* passos) e `mass_fraction` (fração da atribuição positiva dentro de uma máscara). Usado pelos scripts de XAI do PAD e do NIH. | — |
| `treinar_pad_ufes.py` | Além do treino do PAD, é **biblioteca** de funções comuns (construção da ResNet-18, normalização, laço de treino com AMP, carregamento do cache), importada pelos scripts do NIH. | ver seção PAD |
| `check_dataset_balance.py` | Utilitário genérico, usado na escolha do dataset: balanceamento de classes e subgrupos (e interseções) em qualquer CSV de metadados, com contagens por imagem e por paciente e gráficos. | `--csv`, `--class-col`, `--group-cols`, `--intersection-cols`, `--id-col`, `--out-dir` |

Exemplo do utilitário:

```powershell
.venv\Scripts\python.exe codigo\check_dataset_balance.py --csv dataset\PAD-UFES-20\zr7vgbcyr2-1\metadata.csv --class-col diagnostic --group-cols gender,fitspatrick,age --intersection-cols gender,fitspatrick --id-col patient_id --out-dir dataset\PAD-UFES-20\relatorio_balanceamento
```

### PAD-UFES-20

| Script | Função | Argumentos |
|---|---|---|
| `auditar_imagens_pad_ufes.py` | Verifica correspondência entre `metadata.csv` e arquivos, decodifica cada imagem, registra dimensões e modo e calcula o SHA-256 de cada arquivo para achar duplicatas exatas. | `--csv`, `--image-root`, `--out-dir` |
| `comparar_duplicatas_pad_ufes_zip.py` | Confirma, lendo o ZIP original, que as duplicatas encontradas já existem na fonte (e não surgiram na extração). | `--archive`, `--audit`, `--metadata`, `--out-dir` |
| `gerar_artefatos_experimentos.py` | Partições por paciente: separa o teste e cria 5 folds estratificados e agrupados. Com `--reference-splits`, herda as partições de outra coorte. | `--csv`, `--out-dir`, `--seed` (42), `--reference-splits` |
| `preparar_coortes_pad_ufes.py` | Gera as coortes principal e de sensibilidade (decisões sobre duplicatas e registros inválidos), com manifesto e registro de cada decisão. A fonte nunca é alterada. | `--metadata`, `--audit`, `--zip-check`, `--out-dir` |
| `smoke_test_pipeline_pad_ufes.py` | Teste de fumaça do pipeline: carrega um lote, faz *forward*, perda e um passo do otimizador. | `--metadata`, `--cv`, `--image-root`, `--out` |
| `preparar_cache_imagens_pad_ufes.py` | Cache 256 × 256 das imagens das duas coortes (RGB, redimensiona o menor lado e recorta o centro). | `--primary`, `--sensitivity`, `--image-root`, `--out-dir`, `--workers` (8) |
| `treinar_pad_ufes.py` | Fase `cv`: validação cruzada agrupada comparando entropia cruzada padrão e ponderada, escolhe receita e épocas. Fase `final`: treina no desenvolvimento e prediz o teste uma vez (trava). | `fase` (`cv`/`final`), `--coorte` (`principal`/`sensibilidade`), `--forcar` |
| `avaliar_pad_ufes.py` | Avaliação do teste a partir das predições gravadas: macro-F1, métricas por classe, subgrupos (sexo, fototipo, idade) e IC por bootstrap por paciente. Não treina nem prediz. | `--coorte` |
| `gradcam_pad_ufes.py` | Grad-CAM do modelo final, figuras por classe e testes de sanidade (randomização em cascata das camadas e modelo treinado com rótulos embaralhados; Spearman entre mapas). | `--epocas-rotulos-aleatorios` |
| `artefato_sintetico_pad_ufes.py` | Viés com verdade conhecida: quadrado verde 20 × 20 num canto, correlacionado com NEV em níveis de *q*; mede a dependência real (inserir/remover o quadrado) e a atribuição do Grad-CAM e do IG no quadrado, agregada e por imagem (AUC). | — |
| `marcas_caneta_pad_ufes.py` | Artefato real: estima a máscara de tinta (cor), apaga por *inpainting* (OpenCV) e mede a mudança da predição em modelos por fold, contra um placebo de mesma área; relaciona com a atribuição na tinta. | `--raio-central` (0), `--so-analise` |
| `revisao_marcas_caneta_pad_ufes.py` | Conferência às cegas da anotação visual: `gerar` monta uma página HTML local com amostra embaralhada; `analisar <csv>` compara a revisão com a anotação (concordância, kappa, precisão). | `gerar [--n-nao 200]`, `analisar <csv>` |

### NIH ChestX-ray14

| Script | Função | Argumentos |
|---|---|---|
| `auditar_nih_chestxray14.py` | Auditoria dos metadados: partições oficiais, sexo, idade, incidência PA/AP, prevalência de achados por sexo. Também é biblioteca de leitura dos metadados. | `--fonte`, `--out-dir` |
| `preparar_cache_nih_chestxray14.py` | Confere SHA-256 dos ZIPs contra o HF, lê os PNG dos ZIPs em paralelo (validando CRC) e grava um `.npy` 112.120 × 256 × 256 (`uint8`) com índice e manifesto. | `--out-dir`, `--workers` (12) |
| `preparar_coortes_nih_chestxray14.py` | Monta a coorte (PA, derrame × sem achados, uma imagem por paciente, teste oficial, validação estratificada por rótulo × sexo). | `--out-dir`, `--derrame-isolado` |
| `mascaras_toracicas_nih.py` | Segmenta as imagens de teste com o PSPNet do `torchxrayvision` e grava a região torácica (fecho convexo de pulmões, coração e mediastino, dilatado), usada para medir a concentração das atribuições. | — |
| `treinar_nih_chestxray14.py` | Fase `piloto` (fixa E), `experimentos` (40 treinos + sanidade) e `replica` (sementes 47–51). Sorteia o treino conforme *q*, insere o quadrado no exp. B, treina e grava predições. Retomável. | `fase` (`piloto`/`experimentos`/`replica`) |
| `predizer_validacao_nih.py` | Predições na validação para todos os modelos gravados (teste alternativo das medidas de viés). Pula execuções já feitas. | — |
| `xai_nih_chestxray14.py` | Núcleo da análise: Grad-CAM, IG e TCAV (CAV do conceito "masculino" vs. 20 aleatórios, layer3 e layer4) por execução; sanidade; medidas de viés; correlações de Spearman com IC bootstrap; AUC por imagem; figuras. Também é biblioteca para os scripts abaixo. | `--conjunto` (`original`/`replica`/`combinado`), `--so-analise` |
| `controle_dissimilaridade_nih.py` | Controles do sinal H/M: piso de ruído por metades sorteadas, contraste por idade (não manipulada) e exp. B como controle negativo; inclui o teste confirmatório da réplica. | — |
| `pranchas_nih.py` | Pranchas ilustrativas: exemplos de Grad-CAM/IG por *q*, mapas médios por sexo e exemplos do exp. B com o quadrado. | — |
| `sensibilidade_nih.py` | Análises de sensibilidade pré-registradas: TCAV com 100 CAVs aleatórios e IG com linha de base borrada. | `--so-analise` |
| `controle_condicionado_nih.py` | Controle exploratório: recalcula o sinal H/M com mulheres e homens pareados pela probabilidade prevista, para separar o efeito do atalho do efeito da confiança do modelo. | — |

---

## 7. Interromper e retomar

Qualquer script pode ser interrompido com **Ctrl+C**; a GPU é liberada na hora.

| Script | Ao rodar de novo | Perda máxima |
|---|---|---|
| `treinar_nih_chestxray14.py experimentos` / `replica` | pula as execuções com `execucao.json` | a execução em andamento (~40 s) |
| `xai_nih_chestxray14.py` | pula as execuções com `xai/por_execucao/<execução>.json`; sanidade fica em `xai/sanidade.json` | a execução em andamento (~3 min) |
| `predizer_validacao_nih.py` | pula execuções com `predicoes_validacao.csv` | segundos |
| `controle_dissimilaridade_nih.py` | pula as execuções já em `xai/controle_dissimilaridade.csv` | ~1 min |
| `controle_condicionado_nih.py` | pula as execuções já em `xai/controle_condicionado.csv` | ~30 s |
| `sensibilidade_nih.py` | pula as execuções com resultado em `xai/sensibilidade/tcav100/` e `ig_borrada/` | ~2 min |
| `pranchas_nih.py` | pula pranchas cujo PNG existe | uma prancha |
| `marcas_caneta_pad_ufes.py` | reaproveita os modelos por fold já treinados | o fold em andamento |
| `preparar_cache_nih_chestxray14.py` | recomeça do início | o cache inteiro |
| demais | recomeçam do início | segundos a minutos |

---

## 8. Travas do protocolo

As proteções abaixo impedem que uma decisão seja tomada depois de vistos os resultados. Não as contorne sem registrar o motivo.

- **`treinar_pad_ufes.py final`** recusa refazer a predição do teste se `predicoes_teste.csv` existir. `--forcar` permite, e o JSON da execução registra `"forcado": true`.
- **`treinar_nih_chestxray14.py piloto`** recusa rodar de novo. Reescolher E exigiria apagar `piloto/decisao_epocas.json` à mão, o que altera o protocolo.
- **Fichas de protocolo** (`protocolo/`): congeladas antes dos treinos. Mudanças posteriores ficam numa seção "Ajustes após o congelamento", com data e justificativa, sem editar o texto acima dela. Conferência: `Get-FileHash <arquivo> -Algorithm SHA256`.

| Ficha | SHA-256 do arquivo neste repositório | Corresponde a |
|---|---|---|
| `ficha_protocolo_experimental.md` (PAD) | `50028fccf93a52083021cbea86c1c636dd757b24aac3aa1757146dbcebad9367` | congelamento original |
| `ficha_protocolo_experimental_nih.md` (NIH) | `df39aff70d1dfc0b5137731579ca8079cfd6b761c62a37b6381751f614b98e8f` | versão com o pré-registro da réplica |

Hashes anteriores da ficha do NIH: congelamento original `50cdaf2604260dce9500c95044332d2df0f44893ba69929a68b2030fb599bab5`; com o ajuste da validação `55046734ee5fbd258125bd1dbdef9f5673f9f6ef6c029f091aaf135a96c11ee1`.

---

## 9. Problemas comuns

| Sintoma | Causa provável e solução |
|---|---|
| `CUDA indisponível` / `torch.cuda.is_available()` = `False` | PyTorch instalado sem CUDA (pelo PyPI). Reinstale com o comando `--index-url .../cu132` da seção 2. |
| `CUDA out of memory` | Outro processo usa a GPU (p. ex., treino e XAI ao mesmo tempo). Confira com `nvidia-smi`. |
| `FileNotFoundError` em `dataset\...` | Comando rodado fora da raiz do repositório, passo anterior não executado ou arquivo da seção 3 não copiado. |
| Acentos aparecem como `�` no terminal | Só exibição do console; os arquivos são UTF-8. Corrija com `$env:PYTHONIOENCODING = "utf-8"`. |
| `preparar_cache_nih_chestxray14.py` acusa hash divergente | Pacote incompleto ou corrompido. Apague o ZIP indicado e baixe de novo (seção 5.1). |
| `Execute a fase cv antes da fase final` / `Execute a fase piloto antes dos experimentos` | A ordem das tabelas não foi seguida. |
| `ModuleNotFoundError: No module named 'treinar_pad_ufes'` (ou outro script) | Um script saiu de `codigo/`. Eles importam uns aos outros e precisam ficar na mesma pasta. |
| `O piloto já foi executado` | Trava da seção 8. Os experimentos usam o E já gravado. |
