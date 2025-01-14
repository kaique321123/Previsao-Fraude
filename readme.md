## Descrição do Projeto
Este projeto é um modelo de classificação binária desenvolvido para detectar fraudes financeiras em transações. Utiliza algoritmos de aprendizado de máquina e visualizações para avaliar o desempenho do modelo e compreender os padrões dos dados.

### Estrutura do Repositório
- `modelo.py`: Código principal contendo o modelo de classificação.
- `curva_ROC.png`: Gráfico da Curva ROC (Receiver Operating Characteristic).
- `distribuicao_das_previsoes.png`: Gráfico de barras representando a distribuição das previsões do modelo.
- `fraude_vs_nao_fraude.png`: Boxplot que compara os valores de transação entre classes fraudulentas e não fraudulentas.
- **Espaço reservado**: Adicione o nome do arquivo do dataset aqui.

## Dataset
Insira aqui o nome do arquivo ZIP contendo o dataset e informações relevantes sobre o mesmo.

## Requisitos
- Python 3.x
- Bibliotecas: `numpy`, `pandas`, `sklearn`, `matplotlib`

## Execução do Código
1. Certifique-se de que todas as dependências estão instaladas.
2. Extraia o dataset e forneça o caminho correto no código.
3. Execute o script `modelo.py`.

## Conceitos Matemáticos

### Curva ROC e AUC
A Curva ROC é utilizada para avaliar o desempenho de modelos de classificação binária. Ela relaciona:
- **Taxa de Verdadeiros Positivos (TPR)**: 
\[ TPR = \frac{TP}{TP + FN} \]
- **Taxa de Falsos Positivos (FPR)**: 
\[ FPR = \frac{FP}{FP + TN} \]

O valor da área sob a curva (AUC - Area Under the Curve) fornece uma métrica de desempenho:
- **AUC = 1**: Classificação perfeita.
- **AUC = 0.5**: Modelo equivalente ao acaso.

### Boxplot
O boxplot permite visualizar a dispersão dos valores de transações entre as classes "Fraude" e "Não Fraude". Ele exibe:
- Mediana.
- Quartis (Q1 e Q3).
- Outliers.

## Visualizações

### 1. Curva ROC
![Curva ROC](curva_ROC.png)

Representa o equilíbrio entre a taxa de verdadeiros positivos e a taxa de falsos positivos. O AUC obtido foi **0.98**, indicando excelente desempenho.

### 2. Distribuição das Previsões
![Distribuição das Previsões](distribuicao_das_previsoes.png)

Gráfico de barras mostrando a quantidade de transações classificadas como "Fraude" e "Não Fraude". Observa-se um desbalanceamento entre as classes.

### 3. Boxplot: Fraude vs Não Fraude
![Fraude vs Não Fraude](fraude_vs_nao_fraude.png)

Boxplot que compara os valores das transações entre as classes. Transações fraudulentas tendem a ter valores distintos dos não fraudulentos.
