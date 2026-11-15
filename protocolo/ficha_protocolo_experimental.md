# Ficha de protocolo experimental — PAD-UFES-20

**Estado:** congelado em 27/10/2026, antes do primeiro treinamento, sem validação da orientação (revisão pendente). Os ajustes feitos no congelamento estão na última seção; o hash SHA-256 desta ficha está registrado em `Organização/Day7.md`.

## Dataset e proveniência

- **Dataset:** PAD-UFES-20, versão 1 do Mendeley Data, DOI [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1), conforme a referência primária do dataset.
- **Arquivo analisado:** `dataset/PAD-UFES-20/zr7vgbcyr2-1/metadata.csv`.
- **Identificação do snapshot local:** SHA-256 `14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527`.
- **Unidade:** cada linha representa um registro/imagem; `patient_id` identifica o grupo que não pode ser separado entre treino e avaliação.
- **Escopo auditado:** 2.298 registros, 1.373 pacientes e seis diagnósticos (ACK, BCC, MEL, NEV, SCC e SEK). Há 179 pacientes com mais de um diagnóstico; contagens de pacientes por classe não são aditivas.
- **Qualidade observada:** não foram encontradas linhas idênticas, `img_id` repetidos nem valores ausentes em `patient_id`, `img_id` ou `diagnostic`. Os campos `gender` e `fitspatrick` estão ausentes nas mesmas 804 linhas, pertencentes a 579 pacientes; não há pacientes com preenchimento demográfico misto entre registros.
- **Artefato detalhado:** [qualidade_dados_por_paciente.csv](qualidade_dados_por_paciente.csv). A auditoria e os arquivos de divisão podem ser regenerados por `python gerar_artefatos_experimentos.py` (requisitos em `requirements.txt`).

## Preparação das imagens

- Carregar a imagem indicada por `img_id`, converter para RGB e redimensionar mantendo a proporção, de modo que o menor lado tenha 256 pixels; aplicar recorte central de 224 × 224 pixels.
- Normalizar com média `[0.485, 0.456, 0.406]` e desvio padrão `[0.229, 0.224, 0.225]` (referência ImageNet).
- No treino, usar espelhamentos horizontal e vertical com probabilidade 0,5 e rotação aleatória de até 15 graus. Não aplicar variação artificial de cor (color jitter), para preservar sinais de cor da pele e da lesão.
- Validação e teste usam somente redimensionamento, recorte e normalização determinísticos. Aumentos artificiais não podem ser aplicados fora do treino.
- Auditoria técnica realizada em 2.298 imagens: todas decodificam e correspondem a um `img_id`; não foram encontrados arquivos ausentes ou ilegíveis. A largura varia de 147 a 3.474 pixels e a altura, de 147 a 3.476 pixels; 1.440 imagens estão em RGBA e 858 em RGB, que devem ser convertidas consistentemente para RGB no pré-processamento.
- A comparação com o ZIP original confirmou os 15 grupos de cópias exatas: os 30 arquivos comparados têm hashes idênticos no ZIP e no diretório extraído, e o metadata local é byte a byte igual ao do ZIP. As fontes não definem como resolver os 12 grupos em que os mesmos bytes têm diagnósticos diferentes.
- **Regra de elegibilidade desta análise:** excluir os 24 registros dos grupos com conflito de diagnóstico, sem eleger um rótulo. Nos três grupos restantes, com diagnóstico coincidente, manter o menor `img_id` lexicográfico na coorte principal e manter os dois registros na coorte de sensibilidade. Trata-se de uma regra analítica conservadora, não de correção dos rótulos originais.
- Os metadados derivados, as decisões por registro, os hashes e os manifestos estão em `dataset/PAD-UFES-20/derivados_analise_v1/`. A fonte original permanece intacta. PAT_311 e PAT_570 são removidos integralmente das coortes por não terem outros registros após a exclusão dos conflitos.
- Na coorte principal, há 415 pacientes FEMALE (744 imagens) e 377 MALE (723 imagens); faltam dados demográficos em 579 pacientes (804 imagens). Fitzpatrick I: 74 pacientes/145 imagens; II: 463/859; III: 212/390; IV: 34/62; V: 8/10; VI: 1/1. As contagens devem ser recalculadas se a elegibilidade mudar.
- Consultar `auditoria_imagens.csv` e `verificacao_imagens.json` para os registros, caminhos, dimensões, hashes e diagnósticos associados.

## Divisão e seleção

- Usar `StratifiedGroupKFold`, agrupando por `patient_id`, com semente 42. O primeiro fold externo fica reservado como teste final; os demais formam o conjunto de desenvolvimento. A estratificação é aproximada por diagnóstico, pois pacientes podem ter imagens de mais de uma classe.
- No desenvolvimento, usar cinco folds agrupados: cada paciente aparece na validação exatamente uma vez e fica no treino dos outros quatro folds. O arquivo de teste não participa da seleção de modelo, dos pesos, do número de épocas ou das configurações.
- **Coorte principal:** 1.371 pacientes, 2.271 imagens; desenvolvimento com 1.096 pacientes/1.816 imagens e teste com 275 pacientes/455 imagens.
- **Sensibilidade:** 1.371 pacientes, 2.274 imagens; desenvolvimento com 1.096 pacientes/1.819 imagens e teste com 275 pacientes/455 imagens.
- A sensibilidade herda exatamente os mesmos pacientes por partição e folds da coorte principal, isolando o efeito de manter os três pares de mesma classe. As interseções entre desenvolvimento-teste e treino-validação são vazias nas duas coortes.
- Cada coorte tem seus próprios arquivos de metadados, listas de pacientes/folds, distribuição por classe e manifesto em `dataset/PAD-UFES-20/derivados_analise_v1/coorte_principal/particoes/` e `dataset/PAD-UFES-20/derivados_analise_v1/coorte_sensibilidade/particoes/`. Os arquivos de partição diretamente em `artefatos_protocolo/` descrevem a fonte não filtrada e não devem ser usados no treinamento.
- Selecionar configurações pela média de macro-F1 dos cinco folds. Para a avaliação final, treinar no conjunto de desenvolvimento inteiro pelo número mediano de épocas ótimo observado nos folds e avaliar o teste uma única vez.

## Validação técnica do pipeline

- Smoke test concluído com seis imagens, uma por classe, selecionadas exclusivamente do treino do fold 1 da coorte principal.
- O lote foi transformado para `6 × 3 × 224 × 224`; a ResNet-18 pré-treinada produziu logits `6 × 6`; entropia cruzada resultou em perda finita, e uma etapa AdamW alterou os pesos da camada final.
- Evidência em `dataset/PAD-UFES-20/derivados_analise_v1/smoke_test/model_pipeline_smoke.json`. Este teste confirma somente carregamento, formas, perda e atualização; não é métrica de desempenho e não foram salvos pesos.

## Modelos e hiperparâmetros propostos

- **Base:** ResNet-18 com pesos ImageNet `IMAGENET1K_V1`, camada final substituída por seis logits e entropia cruzada sem pesos.
- **Comparação controlada:** mesma ResNet-18, pesos, divisão e pipeline, mas com entropia cruzada ponderada por classe. Calcular os pesos somente com as frequências do fold de treino (frequência inversa, normalizada para média 1).
- **Otimização inicial:** AdamW, taxa de aprendizado `1e-4`, weight decay `1e-4`, batch size 32 e máximo de 30 épocas.
- **Seleção/regularização:** acompanhar macro-F1 na validação; reduzir a taxa de aprendizado por fator 0,1 após duas épocas sem melhora e interromper após cinco épocas sem melhora. Em caso de empate de macro-F1, preferir a menor perda de validação.
- **Reprodutibilidade:** usar a semente 42 para divisão, inicialização e operações estocásticas compatíveis. Registrar versões de bibliotecas, dispositivo e configuração efetivamente usados no relatório dos experimentos.
- As configurações acima são uma especificação inicial, não resultados medidos. Qualquer mudança deve ser anotada antes de consultar o conjunto de teste.

## Métricas e subgrupos

- **Primária global:** macro-F1.
- **Complementares:** acurácia balanceada, acurácia simples, precisão, sensibilidade e F1 por diagnóstico, matriz de confusão e intervalos de confiança de 95%.
- Reamostrar pacientes, não imagens, em 2.000 replicações bootstrap; em cada replicação, manter juntos todos os registros/imagens de cada paciente selecionado.
- Para análises demográficas, informar sempre o número de imagens e de pacientes distintos, a cobertura, as ausências e os denominadores. Não imputar `gender` ou `fitspatrick` ausentes como categorias conhecidas.
- **Regra operacional pré-especificada:** com menos de 30 pacientes distintos em um grupo, apresentar somente contagens descritivas, sem estimativa interpretada como estável nem comparação inferencial. Para sensibilidade por diagnóstico, não apresentar estimativa inferencial quando a célula grupo-diagnóstico tiver menos de 10 pacientes distintos. Não apresentar macro-F1 de seis classes para um grupo se qualquer classe tiver menos de 10 pacientes; não comparar macro-F1 calculados sobre conjuntos diferentes de classes.
- Os limiares são salvaguardas operacionais deste estudo, não garantias universais de poder estatístico. Mesmo acima dos limiares, interpretar as estimativas como descritivas e acompanhadas de incerteza.
- Na coorte principal, FEMALE tem 415 pacientes/744 imagens e MALE 377/723; os dados demográficos faltam em 579 pacientes/804 imagens. Fitzpatrick I: 74 pacientes/145 imagens; II: 463/859; III: 212/390; IV: 34/62; V: 8/10; VI: 1/1. V e VI ficam abaixo do limiar. Recalcular estes suportes se a elegibilidade mudar.
- Fitzpatrick V tem 8 pacientes e VI tem 1; ambos ficam abaixo do limiar. Há 16 de 33 células fototipo-diagnóstico com menos de 10 pacientes distintos; as métricas dessas células devem ser omitidas ou apresentadas apenas descritivamente. Recalcular estes suportes após qualquer alteração da regra de elegibilidade.
- Diferenças serão descritas como disparidades observadas nesta base, sem atribuir causalidade ou concluir, isoladamente, que o sistema é justo ou injusto.

## Grad-CAM

- Inspecionar uma amostra documentada de previsões corretas e incorretas, registrando imagem, classe real e prevista, confiança e grupo demográfico conhecido.
- Tratar os mapas como evidência qualitativa de localização, não como prova de causalidade, fidelidade ou relevância clínica. Adebayo et al. mostram por que a inspeção visual isolada é insuficiente; testes de sanidade adicionais devem ser identificados como tal, caso sejam realizados.

## Artefatos gerados

- Metadados de análise: `dataset/PAD-UFES-20/derivados_analise_v1/coorte_principal/metadata_analise.csv` e `dataset/PAD-UFES-20/derivados_analise_v1/coorte_sensibilidade/metadata_analise.csv`.
- Auditorias de pacientes, IDs por partição/fold e distribuições de classes: subpastas `particoes/` de cada coorte derivada.
- Política aplicada por registro e totais das coortes: `dataset/PAD-UFES-20/derivados_analise_v1/decisoes_coortes.csv` e `manifesto_coortes.json`.
- Confirmação das imagens e do metadata contra o ZIP original: `dataset/PAD-UFES-20/artefatos_protocolo/comparacao_zip_original.json` e `comparacao_duplicatas_zip_original.csv`.
- Teste técnico de carregamento/modelo: `dataset/PAD-UFES-20/derivados_analise_v1/smoke_test/model_pipeline_smoke.json`.

## Ajustes registrados no congelamento (27/10/2026, antes de qualquer treino)

Decisões tomadas sem a orientação, para revisão posterior. Nenhuma métrica de validação ou de teste havia sido calculada quando foram fixadas.

- **Regra de elegibilidade das duplicatas:** mantida como descrita acima (excluir os 24 registros conflitantes; principal com um registro por hash; sensibilidade com os dois).
- **Limiares de subgrupo:** mantidos (30 pacientes por grupo; 10 por célula grupo-diagnóstico).
- **Subgrupos adicionais, pré-especificados agora:** (a) faixa etária por registro (até 50, 51–65 e acima de 65 anos), com os mesmos limiares; como a idade se confunde com o diagnóstico, a sensibilidade por diagnóstico dentro de cada faixa é a leitura principal, e o macro-F1 por faixa é secundário; (b) presença de dados demográficos (registros com e sem sexo/fototipo, que coincidem com biopsiados e não biopsiados), reportada como estrato de seleção, não como grupo demográfico.
- **Pré-processamento em cache:** as imagens são convertidas para RGB, redimensionadas (menor lado 256, bilinear) e recortadas no centro em 256 × 256 uma única vez. No treino, aplicam-se espelhamentos e rotação de até 15 graus sobre o recorte de 256 e, em seguida, o recorte central de 224. Na avaliação, usa-se só o recorte central de 224. O resultado equivale ao redimensionamento seguido de recorte de 224 descrito acima, com diferença de no máximo 1 pixel no deslocamento do recorte.
- **Execução:** GPU (RTX 3060 Ti) com precisão mista (float16 automática) e cuDNN em modo determinístico; os aumentos são aplicados na GPU, com sorteio por imagem.
- **Esquema de taxa de aprendizado no treino final:** como não há validação no treino final, usa-se, em cada época, a mediana das taxas de aprendizado registradas nos cinco folds para aquela época (folds encerrados antes contribuem com a última taxa usada).
- **Época ótima:** a época de maior macro-F1 na validação; em caso de empate, a de menor perda. O número de épocas do treino final é a mediana das épocas ótimas dos cinco folds da configuração escolhida, arredondada para cima.
- **Escolha entre configurações:** maior média de macro-F1 nos cinco folds. Se a diferença entre as médias for menor que um desvio padrão entre folds, registra-se que a escolha não é robusta, mas mantém-se a regra.
- **Coorte de sensibilidade:** a configuração escolhida na principal é treinada na sensibilidade com o mesmo número de épocas e o mesmo esquema de taxa, sem nova seleção.
- **Teste:** avaliado uma única vez por coorte, depois que todas as decisões acima estiverem tomadas e a validação cruzada concluída.
