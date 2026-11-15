# Ficha de protocolo experimental — NIH ChestX-ray14

**Estado:** CONGELADA em 27/10/2026, antes de qualquer treino no NIH. O SHA-256 deste arquivo está registrado em `Organização/Day10.md`. Qualquer mudança posterior vai para uma seção "Ajustes após o congelamento", com data e justificativa, sem editar o texto acima dela.

**Pergunta (Dia 5):** em que medida métodos de explicabilidade por atribuição de pixels (Grad-CAM, IG) e por conceitos (TCAV) revelam a dependência de um classificador de imagens em um atributo sensível ou espúrio, quando a intensidade desse viés nos dados de treino é controlada?

## 1. Dados

- **Fonte:** NIH ChestX-ray14 (CC0), obtido do espelho público `alkzar90/NIH-Chest-X-ray-dataset`, com o SHA-256 dos 12 pacotes conferido. Metadados: `Data_Entry_2017_v2020.csv` (SHA-256 `dc1d2df6…baca11`).
- **Cache:** `cache_imagens_256/imagens_256.npy`, com 112.120 × 256 × 256 em tons de cinza (SHA-256 no manifesto).
- **Coorte:** `derivados_v1/coorte_principal/coorte.csv`.
  - Incidência PA e uma imagem por paciente.
  - Positivo = "Effusion" com ou sem outros achados. Negativo = "No Finding" de paciente sem nenhuma imagem PA com derrame.
- **Partições:**
  - **Teste:** a lista oficial do NIH.
  - **Validação:** 15% dos pacientes de treino/validação, estratificada por rótulo × sexo, semente 42.
  - Nenhum paciente aparece em mais de uma partição.

| Partição | Derrame, mulheres / homens | Sem achados, mulheres / homens |
| --- | ---: | ---: |
| Treino (reservatório) | 1.086 / 1.214 | 7.283 / 8.495 |
| Validação | 192 / 214 | 1.285 / 1.499 |
| Teste | 251 / 322 | 820 / 1.010 |

- **Análise de sensibilidade:** `coorte_derrame_isolado` (só "Effusion", sem outros achados), aplicada à configuração final de cada nível.

## 2. Tarefa e grupos

- **Classificação binária:** derrame pleural (1) contra sem achados (0). Saída com duas classes e softmax.
- **Atributo de grupo:** sexo registrado (M/F), que define quatro grupos: sexo × rótulo.
- O sexo **não** é entrada do modelo. Ele só é usado para montar os treinos e avaliar.

## 3. Experimento A — viés não localizado (sexo)

**Treinos com correlação controlada:**
- Cada treino é sorteado do reservatório "treino" com **tamanho fixo: 1.200 positivos e 2.400 negativos** (3.600 imagens).
- *q* = P(homem | derrame) = P(mulher | sem achados). Assim, *q* = 0,5 significa nenhuma correlação entre sexo e rótulo.
- **Níveis:** *q* ∈ {0,5; 0,75; 0,9; 1,0}. São os mesmos do artefato sintético do PAD-UFES-20, para permitir comparação.
- O tamanho de 1.200 positivos é limitado pelo nível *q* = 1,0, que exige 1.200 homens com derrame (o reservatório tem 1.214).
  - **Decisão:** manter *q* = 1,0 e esse tamanho. O nível extremo é a referência de "atalho perfeito", que também existe no PAD-UFES-20.
  - **Consequência a registrar:** em *q* = 1,0, as 5 sementes usam quase os mesmos positivos. A variação entre as sementes nesse nível é menor que nos outros.
- **Sorteio com contagens exatas:** em cada célula (rótulo × sexo), sorteia-se sem reposição o número de imagens arredondado (homens com derrame = round(1.200 *q*); mulheres sem achados = round(2.400 *q*); o restante do outro sexo). O sorteio usa `numpy.random.default_rng(semente)`.
- **5 repetições por nível**, sementes 42 a 46. A semente controla o sorteio das imagens, a inicialização da camada final e a ordem dos lotes. São 20 treinos.

## 4. Experimento B — viés localizado (contraste)

- **Artefato:** um quadrado xadrez de 20 × 20 pixels (casas de 4 pixels, valores 0 e 255), num canto sorteado. Fica a uma distância de 8 a 40 pixels da borda do recorte de 224 e é inserido depois do aumento de dados.
- **Correlação:** o quadrado aparece em exatamente round(*q* · n) dos positivos e round((1 − *q*) · n) dos negativos, com *q* nos mesmos quatro níveis. As imagens com quadrado são fixas por treino. O canto e a posição são sorteados a cada lote.
- **Sexo:** balanceado (600/600 positivos e 1.200/1.200 negativos), para isolar o artefato.
- **Tamanho e repetições:** o mesmo tamanho do experimento A, com 5 sementes. São 20 treinos.

## 5. Modelo e treino

- **Modelo:** ResNet-18 com pesos ImageNet `IMAGENET1K_V1` e camada final de 2 saídas. O canal de cinza é replicado nos três canais de entrada, com a normalização do ImageNet.
- **Aumento de dados (só no treino):** rotação aleatória de até 10 graus e recorte aleatório de 224 a partir do 256. **Sem espelhamento horizontal**: inverteria a anatomia (lado do coração) e os marcadores de lateralidade.
- **Avaliação:** recorte central de 224.
- **Otimização:** AdamW, taxa 1e-4, decaimento de pesos 1e-4, lote 32, entropia cruzada sem pesos, precisão mista e cuDNN determinístico.
- **Número de épocas E, fixado antes dos experimentos:**
  - Um treino piloto com *q* = 0,5 e semente 42, por até 20 épocas, avaliando a AUC na **validação** a cada época.
  - E = época de maior AUC. Com E fixo e taxa constante, todos os 40 treinos usam a mesma receita.
  - O piloto não usa o teste.
- **Sem seleção de modelo no teste.** O teste é o conjunto de avaliação de todas as execuções. Nenhuma escolha (épocas, limiar, métrica) é feita com ele.

## 6. Medida do viés (verdade de referência)

- **Experimento A**, no teste oficial (composição natural):
  - Acurácia por grupo (sexo × rótulo), com limiar de probabilidade 0,5.
    - **Decisão sobre o limiar:** todos os treinos têm a mesma proporção (1 positivo para 2 negativos). Por isso, o limiar fixo é comparável entre os níveis, embora tenda a favorecer "sem achados". As medidas usadas para testar as hipóteses são **diferenças** entre os sexos, menos sensíveis a esse deslocamento. A AUC por sexo, que não depende do limiar, acompanha essas medidas.
  - **Acurácia do pior grupo.**
  - **Diferença de sensibilidade** (homens − mulheres) e diferença de taxa de falsos positivos.
  - AUC global e AUC por sexo.
  - IC de 95% por bootstrap (2.000 replicações; uma imagem por paciente, então bootstrap de imagens = de pacientes).
- **Experimento B:** dependência medida por intervenção, como no PAD-UFES-20. Insere-se o quadrado em todas as imagens negativas do teste e mede-se a fração que passa a ser prevista como derrame e o aumento médio da probabilidade de derrame.
- **Verificação da manipulação (H1):** a diferença de sensibilidade e a perda na acurácia do pior grupo devem crescer com *q* (Spearman > 0 entre as 20 execuções).

## 7. Explicações por pixel (Grad-CAM e IG)

- **Métodos:** Grad-CAM (saída da layer4) e IG (32 passos, imagem preta como linha de base, soma do valor absoluto nos canais). Implementação em `xai_atribuicoes.py`.
  - Explica-se a **classe derrame** em todas as imagens explicadas, por ser a classe cujo viés se testa (e a mesma do TCAV).
  - **Experimento A:** todas as imagens do teste.
  - **Experimento B:** as imagens negativas do teste com o quadrado inserido, as mesmas da medida de intervenção.
  - Um mapa com soma zero (possível no Grad-CAM, por causa da ReLU) é excluído, e as exclusões são contadas.
- **Região torácica ampliada:**
  - É o **fecho convexo** da união das máscaras de pulmão esquerdo, pulmão direito, coração e mediastino (`chestx_det.PSPNet` do `torchxrayvision`), dilatado em 10 pixels no recorte de 224.
  - Ela inclui as bases pulmonares e os seios costofrênicos, onde o derrame aparece. A máscara pulmonar sozinha as exclui.
  - Ficam fora o abdome, os ombros, o pescoço e os cantos, onde estão os marcadores e textos.
  - **Verificado antes do congelamento:** a união com a estrutura "Facies Diaphragmatica" se estendia pelo abdome até a borda e cobria cerca de 82% do recorte, o que deixaria a métrica quase sem amplitude. O fecho convexo cobre cerca de 60 a 70%.
  - As máscaras são calculadas uma vez para as imagens do teste. Usa-se a imagem do cache ampliada para 512 (a entrada do PSPNet), com limiar de 0,5 por estrutura. O resultado é reduzido para 256 e recortado como a entrada (224). Tudo é salvo com hash.
  - Uma imagem cuja máscara cubra menos de 15% do recorte é tratada como falha de segmentação e excluída da métrica de região, com contagem registrada.
- **Métricas:**
  - **Experimento A:** concentração da atribuição na região torácica (fração da atribuição ÷ fração da área).
  - **Experimento B:** concentração no quadrado, como no PAD-UFES-20.
  - **Experimento A, sinal exploratório sem região:** diferença entre o mapa médio das mulheres e o dos homens com derrame, medida como 1 − correlação de Pearson entre os dois mapas médios (normalizados para soma 1). Ele existe porque a medida de região favorece H2 por construção: as pistas de sexo (mamas, porte físico) também ficam dentro do tórax. Esse sinal não entra na leitura principal de H2, mas é relatado ao lado dela.
- **Direção esperada, definida antes:**
  - **Experimento B:** a concentração no quadrado cresce com *q*. No PAD-UFES-20, isso só ficou nítido em *q* = 1 no agregado; imagem por imagem, o mapa separou as imagens que mudaram de predição.
  - **Experimento A (H2):** a concentração na região torácica **não** acompanha *q* de forma distinguível (IC de Spearman incluindo 0), mesmo com a disparidade crescendo. Isso porque o sexo não ocupa uma região própria.
  - Se a concentração na região torácica **cair** com *q* (atribuição migrando para fora do tórax), isso conta como um sinal detectável e contraria H2.

## 8. Explicações por conceito (TCAV)

- **Conceitos:**
  - "Masculino" e "feminino", definidos por imagens **sem achados** da partição de validação: 250 de cada sexo, sorteadas com semente 42.
  - Usar só imagens sem achados evita que o conceito se confunda com a doença. A validação não entra em nenhum treino.
- **Camadas:** saídas da layer3 e da layer4 (ativações achatadas).
- **CAV:** regressão logística (scikit-learn, L2, C = 1) masculino × feminino, ajustada com 80% das imagens de conceito (estratificado por sexo). A acurácia nos 20% restantes é registrada como medida de quanto o sexo é linearmente decodificável. O CAV é o vetor normal da fronteira, normalizado e orientado para "masculino".
- **Escore TCAV:** fração das imagens de teste com derrame em que a derivada direcional de (logit de derrame − logit de sem achados) na direção do CAV "masculino" é positiva.
  - Como secundária, a média dessa derivada direcional, porque a fração satura em 0 e 1.
- **Teste contra o acaso:**
  - 20 CAVs aleatórios, cada um treinado com duas metades sorteadas das mesmas 500 imagens, com sexo misturado. A partição é sorteada com sementes 0 a 19, iguais em todas as execuções.
  - O escore é considerado significativo se ficar fora do intervalo de 2,5% a 97,5% dos escores aleatórios.
- **Direção esperada (H3):**
  - O escore TCAV do conceito "masculino" para a classe "derrame" cresce com *q*.
  - Ele deve ser significativo a partir de *q* = 0,75 e acompanhar *q* melhor que as métricas por pixel do experimento A (Spearman maior, com IC por bootstrap das execuções).

## 9. Testes de sanidade

- **Randomização em cascata e rótulos embaralhados** (Adebayo et al., 2018), no modelo com *q* = 0,5 e semente 42, para Grad-CAM e IG. O procedimento e a métrica (Spearman por imagem) são os mesmos do Dia 7.
- **TCAV:** os CAVs aleatórios da seção 8 funcionam como controle.

## 10. Análise da detecção

- **Para cada sinal de XAI**, Spearman entre o sinal médio por execução e (a) *q* e (b) a medida real do viés (diferença de sensibilidade no experimento A; taxa de mudança no experimento B), entre as 20 execuções de cada experimento. IC de 95% por bootstrap das execuções.
- **No experimento B**, também a AUC por imagem dentro de cada nível, para identificar as imagens cuja predição muda com o quadrado.
- **Leitura das hipóteses:**
  - **H2** é apoiada se os sinais por pixel acompanharem o experimento B e não o A.
  - **H3** é apoiada se o TCAV acompanhar o experimento A melhor que os sinais por pixel.
  - Um IC que inclui 0 indica ausência de evidência, não prova de ausência. A leitura de H2 é **comparativa**: o sinal por pixel precisa responder ao experimento B muito mais que ao A.
  - Com 5 execuções por nível, o estudo só detecta efeitos grandes. Isso é registrado como limitação.

## 11. Cautelas e limitações (registrar no texto)

- Os rótulos foram extraídos de laudos por PLN (cerca de 90% de acerto, segundo os autores) e contêm erros.
- O sexo é o registrado no exame, binário. Não representa identidade de gênero.
- A população é de um único hospital dos EUA, sem dados brasileiros. O vínculo brasileiro fica no piloto com o PAD-UFES-20.
- O modelo de segmentação foi treinado no ChestX-Det, um subconjunto do NIH. Ele afeta só as máscaras, não o classificador.
- A restrição à incidência PA reduz o confundimento por gravidade, mas não o elimina.
- Treinos pequenos (3.600 imagens) limitam o desempenho absoluto. O objetivo é comparar níveis de *q*, não superar o estado da arte.

## Ajustes após o congelamento

### 1. Teste alternativo na validação (27/10/2026)

**Motivo:** o modelo com rótulos embaralhados ficou com AUC de 0,48 na validação e de 0,66 no teste oficial. A investigação mostrou que o teste oficial mistura o rótulo com a época de aquisição:
- 79% dos negativos do teste são de pacientes com ID acima de 28.000, os mais recentes; entre os positivos, são 9%.
- A divisão oficial pôs no teste 58% desses pacientes, contra 4% dos demais.
- Essas imagens têm outro contraste. No teste, o desvio-padrão da intensidade separa os rótulos com AUC de 0,81; na validação, com 0,59.
- No treino e na validação, o ID do paciente não prevê o rótulo (AUC de 0,50 e 0,49).
- O ID também não prevê o sexo no teste (AUC de 0,48 a 0,51).

**Ajuste:** as medidas de viés da seção 6 (diferença de sensibilidade, acurácia do pior grupo, diferença de FPR e AUC por sexo) também são calculadas na **partição de validação, sem as 500 imagens de conceito do TCAV** (`predizer_validacao_nih.py`). A validação tem a mesma distribuição do treino e não entra em nenhum treino; o único uso dela foi a escolha de E no piloto.
- **A análise primária continua no teste oficial**, como pré-registrado. A validação é uma análise de sensibilidade.
- As explicações (seções 7 e 8) continuam calculadas no teste oficial.

### 2. Réplica confirmatória pré-registrada (27/10/2026, antes de treinar as sementes novas)

**Motivo:** a diferença entre os mapas médios de mulheres e homens (sinal exploratório da seção 7) acompanhou *q* e resistiu a controles de ruído, de atributo não manipulado e ao controle negativo do experimento B (Dia 11). Porém, ela foi definida e avaliada nos mesmos 40 treinos. Para confirmá-la, usa-se uma amostra nova e independente de treinos, com a hipótese e a regra de decisão fixadas aqui, antes de qualquer treino com as sementes novas.

**Desenho:**
- Os mesmos 40 treinos (experimentos A e B, 4 níveis de *q*) com as **sementes 47 a 51**.
- Receita, coorte, código e E = 9 inalterados. Não há novo modelo de rótulos embaralhados.
- Explicações (seções 7 e 8), medidas de viés (seção 6) e controles do sinal (`controle_dissimilaridade_nih.py`: 20 sorteios de metades com sementes 0 a 19 e divisão por idade na mediana) idênticos aos já usados.

**Hipótese confirmatória (HC):**
- **Medida:** excesso = dissimilaridade por sexo − média das dissimilaridades das 20 metades sorteadas com os mesmos tamanhos, nas 573 imagens de teste com derrame, por execução.
- **HC-A:** nas 20 execuções do experimento A com sementes 47 a 51, o Spearman entre *q* e o excesso é positivo, com o limite inferior do IC de 95% acima de 0. O IC é obtido por bootstrap das execuções, com 2.000 reamostragens e gerador com semente 42.
- **HC-B:** nas execuções do experimento B com sementes 47 a 51, o IC de 95% do mesmo Spearman inclui 0 ou fica inteiro abaixo de 0. No Grad-CAM, as execuções B com *q* = 1 são excluídas: sem o quadrado, quase todos os mapas ficam vazios (Dia 11).
- **Regra de decisão:** a HC é confirmada para um método se HC-A e HC-B forem atendidas. Grad-CAM e IG são avaliados separadamente, e os dois resultados são relatados. Se a HC não for confirmada, o sinal é relatado como exploratório não replicado.

**Análises secundárias:**
- H1, H2 e H3 pelas regras das seções 6 a 10, só nas sementes 47 a 51 (replicação).
- Análise combinada das 10 sementes, relatada como descritiva, com IC.

**Análises de sensibilidade, declaradas antes e sem alterar as leituras primárias:**
- **S1, TCAV com 100 CAVs aleatórios** (sementes 0 a 99, mesmo procedimento da seção 8), em todas as execuções. Significância pela faixa de 2,5% a 97,5% dos 100 escores aleatórios.
- **S2, IG com linha de base borrada:** para cada imagem, a linha de base é a própria imagem com filtro gaussiano de σ = 10 pixels no recorte de 224 (Sturmfels; Lundberg; Lee, 2020), com 32 passos. Aplicada a:
  - (a) concentração no quadrado do experimento B, em todas as execuções;
  - (b) testes de sanidade do IG (cascata e rótulos embaralhados, nas mesmas 300 imagens).
- **Expectativa registrada para S2:** concentração no quadrado maior que a da linha de base preta para *q* < 1, porque as casas pretas deixam de coincidir com a linha de base.
