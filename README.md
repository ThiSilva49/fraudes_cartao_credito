# Detectando Fraude em Cartões de Crédito

*SO TEM O CODIGO POIS O CSV O GITHUB NÃO SUPORTA*

Este projeto utiliza técnicas de Machine Learning para detectar fraudes em transações de cartão de crédito. Ele usa um **modelo de Random Forest** para classificar transações como legítimas ou fraudulentas com base em dados históricos de transações.

## Objetivo

O objetivo deste projeto é prever se uma transação de cartão de crédito é fraudulenta ou não. Para isso, utilizamos um dataset que contém informações sobre transações de cartões de crédito, incluindo variáveis como valores das transações e a classe da transação (fraudulenta ou não).

## Como Funciona

### 1. **Carregamento dos Dados**
O código começa carregando um arquivo CSV chamado `creditcard.csv` que contém os dados das transações. Esse dataset contém informações sobre transações, incluindo a variável target "Class", onde `1` representa uma fraude e `0` representa uma transação legítima.

### 2. **Pré-processamento**
O código realiza algumas etapas de pré-processamento, incluindo:
- **Normalização** da variável "Amount" para que tenha média 0 e desvio padrão 1.
- **Separação** dos dados em features (`X`) e target (`y`), e então divide os dados em conjuntos de treino e teste (80% treino, 20% teste).

### 3. **Treinamento do Modelo**
O modelo de **Random Forest** é treinado utilizando os dados de treino. O modelo é configurado com 100 árvores de decisão.

### 4. **Avaliação**
O modelo é avaliado usando:
- **Acurácia**, que indica a proporção de previsões corretas.
- **Relatório de Classificação**, que inclui métricas como precisão, recall e F1-score.
- **Matriz de Confusão**, que mostra a quantidade de falsos positivos, falsos negativos, verdadeiros positivos e verdadeiros negativos.
- **Curva ROC e AUC**, que são usadas para avaliar a capacidade do modelo em distinguir entre transações legítimas e fraudulentas.

### 5. **Importância das Features**
O código também exibe a importância relativa das variáveis para o modelo, ajudando a entender quais características influenciam mais as previsões de fraude.

