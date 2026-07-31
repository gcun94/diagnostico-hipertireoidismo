# Diagnostico de Hipertireoidismo - Modelo Preditivo

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)](https://xgboost.readthedocs.io/)

---

## Problema de Negocio

Hospital e medicos enfrentam dilemas diarios no diagnostico do hipertireoidismo. O diagnostico tardio traz inumeras complicacoes, alem de sintomas similares a outras condicoes de saude.

Exemplo: o hipertireoidismo tardio (ou nao tratado) apresenta sintomas como taquicardia, que pode ser tambem visualizada em quadros de ansiedade, desidratacao e infeccoes. O medico decidir pelo tratamento incorreto por duvida entre patologias diferentes como tratar taquicardia de hipertireoidismo com ansioliticos nao mostra resultado nenhum na condicao, e a tireoide continua em negligencia.

O resultado e agravamento da condicao do paciente, desperdicio de recursos medicos e tempo. Um modelo preditivo preciso e fundamental para garantir o tratamento correto, reduzir custos com procedimentos desnecessarios e melhorar a qualidade de vida dos pacientes.

---

## Dataset

- **Fonte:** Base_M43_Pratique_Hypothyroid.csv
- **Linhas:** 3772
- **Colunas:** 30
- **Target:** `binaryClass` (P = positivo / N = negativo)
- **Desbalanceamento:** 92.29% positivo

> **Nota sobre o dataset:** apesar do nome do arquivo sugerir hipotireoidismo, a analise dos valores de TSH confirma que o dataset representa hipertireoidismo. A classe positiva apresenta mediana de TSH = 1.2, compativel com excesso de hormonio tireoidiano, enquanto a classe negativa apresenta mediana de TSH = 12.0.

### Descricao das Variaveis

| Variavel                  | Tipo       | Descricao                          |
| ------------------------- | ---------- | ---------------------------------- |
| age                       | Numerica   | Idade do paciente                  |
| sex                       | Categorica | Sexo do paciente                   |
| on thyroxine              | Categorica | Paciente em uso de tiroxina        |
| on antithyroid medication | Categorica | Uso de medicamento antitireoidiano |
| sick                      | Categorica | Paciente doente                    |
| pregnant                  | Categorica | Paciente gravida                   |
| thyroid surgery           | Categorica | Cirurgia de tireoide previa        |
| query hypothyroid         | Categorica | Consulta de hipotireoidismo        |
| query hyperthyroid        | Categorica | Consulta de hipertireoidismo       |
| goitre                    | Categorica | Presenca de bocio                  |
| tumor                     | Categorica | Presenca de tumor                  |
| psych                     | Categorica | Condicao psiquiatrica              |
| TSH                       | Numerica   | Valor do TSH                       |
| T3                        | Numerica   | Valor do T3                        |
| T4U                       | Numerica   | Valor do T4U                       |
| FTI                       | Numerica   | Valor do FTI                       |
| referral source           | Categorica | Origem do encaminhamento           |
| binaryClass               | Categorica | Target - Diagnostico (P/N)         |

---

## Estrutura do Projeto

```
diagnostico-hipertireoidismo/
│
├── data/
│   ├── raw/                  # Dados originais sem alteracao
│   └── processed/            # Dados apos pre-processamento
│
├── notebooks/
│   ├── 01_eda.ipynb          # Analise exploratoria
│   ├── 02_preprocessing.ipynb # Preparacao dos dados
│   ├── 03_training.ipynb     # Treinamento e selecao do modelo
│   └── 04_evaluation.ipynb   # Avaliacao final
│
├── src/
│   └── utils/                # Funcoes auxiliares reutilizaveis
│
├── outputs/
│   ├── figures/              # Graficos e visualizacoes
│   └── models/               # Modelos treinados serializados
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Metodologia

### 01 - Analise Exploratoria (EDA)

Quatro camadas de analise:

1. **Reconhecimento inicial:** dataset com 30 colunas, todas carregadas como categoricas. Dados faltantes representados como `?`. TBG e TBG measured 100% nulos.
2. **Distribuicao univariada:** identificacao de outlier em `age` (valor 455), assimetria positiva em TSH, T3, T4U e FTI.
3. **Relacao com o target:** TSH identificado como principal preditor (correlacao -0.43). Mediana de TSH na classe positiva = 1.2 vs 12.0 na negativa, confirmando hipertireoidismo.
4. **Analise multivariada:** investigacao de multicolinearidade (TT4 vs FTI = 0.79), artefato de coleta em `num_exames_coletados` ligado a `referral_source=other`, e confounding de `sick` e `thyroid surgery` sobre o TSH.

### 02 - Pre-processamento

Divisao explicita entre operacoes deterministicas (antes do split, no notebook 02) e estatisticas (depois do split, dentro do Pipeline por fold no notebook 03).

**Operacoes deterministicas (notebook 02):**

| Etapa | Operacao                                                  |
| ----- | --------------------------------------------------------- |
| 1     | Substituicao de `?` por NaN e conversao de tipos          |
| 2     | Encoding do target (P=1, N=0)                             |
| 3     | Encoding das binarias (t/f, M/F)                          |
| 4     | Feature engineering (risk_factors, psych_lithium_group)   |
| 5     | Remocao de 15 colunas                                     |
| 6     | Transformacao log1p em TSH, T3, T4U, FTI                  |
| 7     | Tratamento de outlier de age (>115 anos marcado como NaN) |
| 8     | Split estratificado (75/25, random_state=42, stratify)    |

**Operacoes estatisticas (notebook 03, dentro do Pipeline, por fold):**

| Operacao      | Detalhe               |
| ------------- | --------------------- |
| OneHotEncoder | referral source       |
| RobustScaler  | variaveis continuas   |
| KNNImputer    | k=5, weights=distance |

As tres operacoes estatisticas foram deslocadas para o Pipeline do notebook 03 para que o fit ocorra apenas no treino de cada fold do cross-validation, eliminando vazamento. O artefato exportado pelo notebook 02 sai com referral source em texto, NaN preservados em age e nos labs, e sem escala.

**Features engineered:**

- `risk_factors`: contagem de 7 variaveis de burden clinico (sick, thyroid surgery, pregnant, tumor, goitre, on antithyroid medication, query hyperthyroid). Reformulada da versao inicial de 5 variaveis, que mantinha leakage num subgrupo agregado.
- `psych_lithium_group`: 3 categorias com logica OR para capturar pacientes com lithium sem psych documentado.

**Shape do artefato exportado:** X_train (2829, 17) e X_test (943, 17), com NaN preservados em age e nos labs, imputados no Pipeline do notebook 03.

### 03 - Treinamento e Selecao

Nove configuracoes avaliadas (3 modelos x 3 estrategias de balanceamento) por validacao cruzada estratificada (StratifiedKFold, 5 folds). Pipeline com ColumnTransformer + KNNImputer + modelo, com SMOTE exclusivamente dentro do fold de treino.

**Investigacao de leakage:** recall inicial proximo de 0.99 levantou suspeita. Inspecao dos coeficientes da regressao logistica identificou `on thyroxine` com coeficiente 4.68, confirmando leakage clinico (tiroxina trata hipotireoidismo, condicao oposta). Feature removida na origem e artefatos regenerados.

**Criterio de selecao (fixado antes dos resultados):**

- Primario: maior recall da classe 1 (doente)
- Desempate: maior recall da classe 0 dentro de 1 ponto de diferenca
- Filtro: modelos com treino saturado em 1.0 descartados

**Resultado:** XGBoost com class_weight selecionado para tuning por ser o unico modelo sem overfit e equilibrado (recall 0.97 em ambas as classes).

### 04 - Avaliacao Final

Modelo reconstruido do zero no notebook 04 e avaliado uma unica vez no conjunto de teste.

---

## Resultados

### Modelo Final

**XGBoost tunado** (combinacao 58 de 60 no RandomizedSearchCV, selecionada pelo criterio clinico do projeto sobre o `cv_results_`, e nao pela metrica interna da busca)

| Hiperparametro     | Valor |
| ------------------ | ----- |
| max\_depth         | 4     |
| min\_child\_weight | 1     |
| learning\_rate     | 0.05  |
| n\_estimators      | 50    |
| subsample          | 0.9   |
| colsample\_bytree  | 1.0   |
| scale\_pos\_weight | 0.5   |
| random\_state      | 42    |

### Metricas

| Metrica                  | Validacao Cruzada | Teste |
| ------------------------ | ----------------- | ----- |
| Recall classe 1 (doente) | 0.98              | 0.98  |
| Recall classe 0 (sadio)  | 0.95              | 0.95  |
| ROC-AUC                  | ~0.98             | 0.984 |

### Leitura Clinica (943 pacientes de teste)

- 870 doentes reais: 853 identificados corretamente, 17 nao detectados (2% falsos negativos)
- 73 sadios reais: 69 corretos, 4 falsos alarmes
- Baseline produziria ~22 falsos alarmes; modelo final reduziu para 4 mantendo o mesmo recall de doentes

---

## Limitacoes e Trabalho Futuro

- Limiar de decisao (0.5) nao calibrado; ajuste permite mover o equilibrio entre falsos negativos e positivos sem retreinar
- Unidades de medida dos exames nao confirmadas (TSH chega a 530); nao afeta o modelo mas limita interpretacao clinica absoluta
- Alta separabilidade do problema: o desempenho reflete caracteristica do dataset, nao apenas qualidade do modelo
- Dupla contagem de `sick` e `thyroid surgery` (individuais e em risk_factors), mantida por decisao consciente
- Interpretabilidade via SHAP: aplicar SHAP values ao modelo final permitiria uma leitura mais aprofundada das metricas, mostrando a contribuicao individual de cada feature nas predicoes e validando clinicamente o raciocinio do modelo alem dos coeficientes ja analisados

---

## Como Reproduzir

```
# Clone o repositorio
git clone https://github.com/gcun94/diagnostico-hipertireoidismo.git

# Entre na pasta
cd diagnostico-hipertireoidismo

# Copie o dataset manualmente para:
# data/raw/Base_M43_Pratique_Hypothyroid.csv

# Instale as dependencias
pip install -r requirements.txt

# Execute os notebooks na ordem numerica
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_training.ipynb
notebooks/04_evaluation.ipynb
```

---

## Tecnologias

- **Manipulacao de dados:** Pandas, NumPy
- **Visualizacao:** Matplotlib, Seaborn, Plotly
- **Pre-processamento:** Scikit-learn, Imbalanced-learn
- **Machine Learning:** XGBoost, Scikit-learn
- **Otimizacao:** RandomizedSearchCV
- **Ambiente:** Jupyter Notebook

---

## Requisitos

Ver arquivo `requirements.txt`

---

## Autor

**Gustavo Simoes da Cunha**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/gustavo-s-cunha-516102181/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/gcun94)

---

## Licenca

MIT License
