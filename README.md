# Diagnostico de Hipertireoidismo - Modelo Preditivo

> 🚧 Projeto em desenvolvimento

## Problema de Negocio

Hospital e médicos enfrentam dilemas diários no diagnostico do hipertireoidismo. O diagnostico tardio traz inúmeras complicações, além de sintomas similares a outras condições de saúde.

Exemplo: o hipertireoidismo tardio (ou não tratado) apresenta sintomas (não somente) como taquicardia que pode ser também visualizada em quadros de ansiedades, desidratações, infecções, entre outros. O medico decidir o tratamento incorreto por duvida de possibilidade de patologias diferentes, como tratar taquicardia de hipertireoidismo com ansiolíticos, não mostra resultado nenhum na condições e a tireoide continua em negligencia.

Assim resulta em agravamento da condição do paciente, desperdício de recursos médicos e tempo. Desenvolver um modelo preditivo para o diagnostico preciso e fundamental, para o tratamento correto. A precisão e velocidade do diagnostico e fundamental para evitar complicações e tratamento assertivo, reduzindo custos com procedimentos e tratamentos desnecessários e melhorando a qualidade de vida dos pacientes.

---

## Dataset

* **Fonte:** Base_M43_Pratique_Hypothyroid.csv
* **Linhas:** 3772
* **Colunas:** 30
* **Target:** `binaryClass`

### Descrição das Variáveis

| Variável | Tipo | Descricao |
|---|---|---|
| age | Categórica | Idade do paciente |
| sex | Categórica | Sexo do paciente |
| on thyroxine | Categórica | Paciente em uso de tiroxina |
| query on thyroxine | Categórica | Consulta sobre tiroxina |
| on antithyroid medication | Categórica | Uso de medicamento antitireoidiano |
| sick | Categórica | Paciente doente |
| pregnant | Categórica | Paciente gravida |
| thyroid surgery | Categórica | Cirurgia de tireoide previa |
| I131 treatment | Categórica | Tratamento com Iodo-131 |
| query hypothyroid | Categórica | Consulta de hipotireoidismo |
| query hyperthyroid | Categórica | Consulta de hipertireoidismo |
| lithium | Categórica | Uso de litio |
| goitre | Categórica | Presenca de bocio |
| tumor | Categórica | Presenca de tumor |
| hypopituitary | Categórica | Hipopituitarismo |
| psych | Categórica | Condicao psiquiatrica |
| TSH measured | Categórica | TSH foi medido |
| TSH | Numérica | Valor do TSH |
| T3 measured | Categórica | T3 foi medido |
| T3 | Numérica | Valor do T3 |
| TT4 measured | Categórica | TT4 foi medido |
| TT4 | Numérica | Valor do TT4 |
| T4U measured | Categórica | T4U foi medido |
| T4U | Numérica | Valor do T4U |
| FTI measured | Categórica | FTI foi medido |
| FTI | Numérica | Valor do FTI |
| TBG measured | Categórica | TBG foi medido |
| TBG | Numérica | Valor do TBG |
| referral source | Categórica | Origem do encaminhamento |
| binaryClass | Categórica | Target - Diagnostico (P/N) |

---

## Estrutura do Projeto

```
diagnostico-hipertireoidismo/
│
├── data/
│   ├── raw/                  # Dados originais sem alteração
│   └── processed/            # Dados após pré-processamento
│
├── notebooks/
│   ├── 01_eda.ipynb          # Analise exploratória
│   ├── 02_preprocessing.ipynb # Preparação dos dados
│   ├── 03_training.ipynb     # Treinamento do modelo
│   └── 04_evaluation.ipynb   # Avaliação e métricas
│
├── src/
│   └── utils/                # Funções auxiliares reutilizáveis
│
├── outputs/
│   ├── figures/              # Gráficos e visualizações
│   └── models/               # Modelos treinados serializados
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Metodologia

1. Analise Exploratória de Dados (EDA)
2. Pré-processamento e tratamento de dados
3. Treinamento do modelo de classificação
4. Avaliação de métricas e desempenho

---

## Resultados

> A ser preenchido após conclusão do modelo.

---

## Como Reproduzir

```bash
# Clone o repositório
git clone <url-do-repositorio>

# Instale as dependências
pip install -r requirements.txt

# Execute os notebooks na ordem numérica
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_training.ipynb
notebooks/04_evaluation.ipynb
```

---

## Tecnologias

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-purple)](https://shap.readthedocs.io/)

* **Manipulação de dados:** Pandas · NumPy
* **Visualização:** Matplotlib · Seaborn
* **Machine Learning:** Scikit-learn
* **Interpretabilidade:** SHAP
* **Ambiente:** Jupyter Notebook

---

## Requisitos

Ver arquivo `requirements.txt`

---

## Autor

**Gustavo Simões da Cunha**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/gustavo-s-cunha-516102181/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/gcun94)

---

## Licenca

MIT License
