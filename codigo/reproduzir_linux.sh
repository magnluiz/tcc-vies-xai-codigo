#!/usr/bin/env bash
# Reprodução completa do TCC no Linux, na ordem do README (seções 4 e 5; ver também a seção 10).
#
# Uso (da raiz do repositório):
#   bash codigo/reproduzir_linux.sh            # roda tudo, retomando do último passo concluído
#   bash codigo/reproduzir_linux.sh --lista    # só mostra os passos e quais já terminaram
#
# Cada passo grava um log em dataset/_reproducao/logs/ e, ao terminar sem erro, uma marca em
# dataset/_reproducao/concluidos.txt. Ao rodar de novo, os passos marcados são pulados. O
# roteiro para no primeiro erro. Entradas manuais (não geradas por script) que precisam existir;
# as fichas e as anotações vêm de protocolo/ e anotacoes/ (cópia na seção 3 do README):
#   - dataset/PAD-UFES-20/artefatos_protocolo/ficha_protocolo_experimental.md (ficha congelada)
#   - dataset/PAD-UFES-20/derivados_analise_v1/marcas_caneta/anotacao_marcas_caneta.csv
#   - dataset/PAD-UFES-20/derivados_analise_v1/marcas_caneta/revisao_entrada/revisao_marcas_caneta.csv
#     (é o anotacoes/conferencia_marcas_caneta.csv)
#   - dataset/NIH-ChestXray14/ficha_protocolo_experimental_nih.md e dataset/NIH-ChestXray14/fonte/
#   - dataset/DATASUS/fonte/ (os .dbc já baixados; sem eles, o script baixa do FTP)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# Memória da GPU: com outros programas abertos (navegador, editor), a reserva fragmentada do
# PyTorch impediu blocos de 1 GB no XAI do Experimento B. Segmentos expansíveis evitam isso sem
# mudar nenhum cálculo. PYTHONUNBUFFERED faz o log de cada passo sair linha a linha.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1
# CPU: o numpy e o scipy trazem cada um seu OpenBLAS, e o PyTorch traz o OpenMP; com 16 threads
# em cada um, os CAVs do TCAV (regressões logísticas em 400 × 25–50 mil atributos) ficavam 40 a 100
# vezes mais lentos que com 4 threads (medido em 25/11/2026: 4–10 s contra 0,1 s por ajuste).
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
OUT=dataset/_reproducao
LOGS=$OUT/logs
DONE=$OUT/concluidos.txt
mkdir -p "$LOGS"; touch "$DONE"
PADD=dataset/PAD-UFES-20/derivados_analise_v1

PASSOS=(
  "pad01_auditoria|$PY codigo/auditar_imagens_pad_ufes.py"
  "pad02_duplicatas_zip|$PY codigo/comparar_duplicatas_pad_ufes_zip.py"
  "pad03_particoes_originais|$PY codigo/gerar_artefatos_experimentos.py"
  "pad04_coortes|$PY codigo/preparar_coortes_pad_ufes.py"
  "pad05a_particoes_principal|$PY codigo/gerar_artefatos_experimentos.py --csv $PADD/coorte_principal/metadata_analise.csv --out-dir $PADD/coorte_principal/particoes"
  "pad05b_particoes_sensibilidade|$PY codigo/gerar_artefatos_experimentos.py --csv $PADD/coorte_sensibilidade/metadata_analise.csv --out-dir $PADD/coorte_sensibilidade/particoes --reference-splits $PADD/coorte_principal/particoes"
  "pad06_smoke_test|$PY codigo/smoke_test_pipeline_pad_ufes.py"
  "pad07_cache|$PY codigo/preparar_cache_imagens_pad_ufes.py"
  "pad08_cv|$PY codigo/treinar_pad_ufes.py cv"
  "pad09_final_principal|$PY codigo/treinar_pad_ufes.py final"
  "pad10_final_sensibilidade|$PY codigo/treinar_pad_ufes.py final --coorte sensibilidade"
  "pad11a_avaliar_principal|$PY codigo/avaliar_pad_ufes.py"
  "pad11b_avaliar_sensibilidade|$PY codigo/avaliar_pad_ufes.py --coorte sensibilidade"
  "pad12_gradcam|$PY codigo/gradcam_pad_ufes.py"
  "pad13_artefato_sintetico|$PY codigo/artefato_sintetico_pad_ufes.py"
  "pad14a_marcas_r56|$PY codigo/marcas_caneta_pad_ufes.py --raio-central 56"
  "pad14b_marcas_mascara_completa|$PY codigo/marcas_caneta_pad_ufes.py"
  "pad15_revisao_marcas|$PY codigo/revisao_marcas_caneta_pad_ufes.py analisar $PADD/marcas_caneta/revisao_entrada/revisao_marcas_caneta.csv"
  "nih01_auditoria|$PY codigo/auditar_nih_chestxray14.py"
  "nih02_cache|$PY codigo/preparar_cache_nih_chestxray14.py"
  "nih03_coorte_principal|$PY codigo/preparar_coortes_nih_chestxray14.py"
  "nih04_coorte_derrame_isolado|$PY codigo/preparar_coortes_nih_chestxray14.py --derrame-isolado"
  "nih05_mascaras|$PY codigo/mascaras_toracicas_nih.py"
  "nih06_piloto|$PY codigo/treinar_nih_chestxray14.py piloto"
  "nih07_experimentos|$PY codigo/treinar_nih_chestxray14.py experimentos"
  "nih07b_validacao|$PY codigo/predizer_validacao_nih.py"
  "nih08_xai_original|$PY codigo/xai_nih_chestxray14.py"
  "nih10_controle_dissimilaridade|$PY codigo/controle_dissimilaridade_nih.py"
  "nih11_pranchas|$PY codigo/pranchas_nih.py"
  "nih12_replica|$PY codigo/treinar_nih_chestxray14.py replica"
  "nih13a_xai_replica|$PY codigo/xai_nih_chestxray14.py --conjunto replica"
  "nih13b_validacao_replica|$PY codigo/predizer_validacao_nih.py"
  "nih13c_controle_dissimilaridade_replica|$PY codigo/controle_dissimilaridade_nih.py"
  "nih13d_analise_combinada|$PY codigo/xai_nih_chestxray14.py --conjunto combinado --so-analise"
  "nih14_sensibilidade|$PY codigo/sensibilidade_nih.py"
  "nih15_controle_condicionado|$PY codigo/controle_condicionado_nih.py"
  "datasus01_numeros|$PY codigo/numeros_datasus.py"
)

if [[ "${1:-}" == "--lista" ]]; then
  for p in "${PASSOS[@]}"; do n=${p%%|*}; grep -qx "$n" "$DONE" && echo "[ok] $n" || echo "[  ] $n"; done
  exit 0
fi

echo "== início: $(date '+%F %T')" | tee -a "$OUT/resumo.txt"
for p in "${PASSOS[@]}"; do
  nome=${p%%|*}; cmd=${p#*|}
  if grep -qx "$nome" "$DONE"; then continue; fi
  ini=$(date +%s)
  echo "-> $nome: $cmd" | tee -a "$OUT/resumo.txt"
  if ! bash -c "$cmd" > "$LOGS/$nome.log" 2>&1; then
    echo "!! ERRO em $nome após $(( $(date +%s) - ini )) s; ver $LOGS/$nome.log" | tee -a "$OUT/resumo.txt"
    tail -20 "$LOGS/$nome.log"
    exit 1
  fi
  echo "   ok em $(( $(date +%s) - ini )) s" | tee -a "$OUT/resumo.txt"
  echo "$nome" >> "$DONE"
done
echo "== fim: $(date '+%F %T')" | tee -a "$OUT/resumo.txt"
