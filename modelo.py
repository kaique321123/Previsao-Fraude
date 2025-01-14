import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import joblib

# lendo o dataset
dados = pd.read_csv('creditcard.csv')

# Informações Importantes 
# Time: O tempo em que a transação foi feita (em segundos desde o início do conjunto de dados).
# V1 a V28: Variáveis que representam características extraídas (features), mas não possuem interpretação explícita, pois foram ofuscadas por motivos de privacidade.
# Amount: O valor da transação.
# Class: A coluna alvo (target), onde 1 significa fraude e 0 significa não fraude.

# limpando dados
dados = dados.dropna()

# Amostragem aleatória de 10 mil registros para conseguir trabalhar de forma mais rápida com o processamento disponível
# dados = dados.sample(n=10000, random_state=1)

# Ver os dados
#print(dados.info())
#print(dados.describe())
print("Quantidade de Fraudes = 1 e Não Fraudes = 0")
print(dados['Class'].value_counts())

# Criar o boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='Class', y='Amount', data=dados)

# Adicionar título e rótulos
# plt.title('Boxplot da coluna "Amount" por "Class" (Fraude vs Não Fraude)')
# plt.xlabel('Class (0 = Não Fraude, 1 = Fraude)')
# plt.ylabel('Valor da Transação')
# plt.show()

# Escalonando as Variáveis para deixar elas padronizadas
scaler = StandardScaler()
dados['Amount'] = scaler.fit_transform(dados[['Amount']])
dados['Time'] = scaler.fit_transform(dados[['Time']])

# Separando Variáveis Independentes e Dependentes 
X = dados.drop(columns=['Class'])   #dependente
y = dados['Class']  # independente

# Dividindo os dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Aumentar a quantidade de dados de Fraude para balancear com Não Fraude
# smote = SMOTE(random_state=1)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# print("Quantidade de Fraudes = 1 e Não Fraudes = 0 após o SMOTE")
# print(y_train_resampled.value_counts())

# Diminuir a quantidade de dados de Fraude para balancear com Não Fraude, o SMOTE estava muito pesado
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=1)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
print("Quantidade de Fraudes = 1 e Não Fraudes = 0 após o Under-Sampling")
print(y_train_resampled.value_counts())


# Mudei de Redes Neurais para Random Forest por ser mais simples de entender e exigir menos poder computacional
# modelo Base
rf = RandomForestClassifier(random_state=1)

# Definir a grade de parâmetros
# Diminui os valores, pois estava muito pesado
param_grid = {
    'n_estimators': [50],
    'max_depth': [10, 20],
    'min_samples_leaf': [4],
    'bootstrap': [True]
}

# Troquei o Grid Search pelo Random Search para diminuir o tempo de processamento 
random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_grid, 
    n_iter=2,  # padrão é 5, mas só tenho dois valores para testar
    cv=3, 
    n_jobs=-1, 
    verbose=2
)
random_search.fit(X_train, y_train)

# # Exibir os melhores parâmetros
# print("Melhores Parâmetros:", random_search.best_params_)
# param_grid = {
#     'n_estimators': [50, 100],
#     'max_depth': [10, 20, 30],
#     'min_samples_leaf': [4, 5],
#     'bootstrap': [True, False]
# }
# grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
# print("Melhores Parâmetros:", grid_search.best_params_)

# Treinar o modelo final com os melhores parâmetros
best_model = random_search.best_estimator_

# Usando o Modelo Ajustado
y_pred = best_model.predict(X_test)

# # Comparando Previsão com Real
comparacao = pd.DataFrame({'Real': y_test.values, 'Previsto': y_pred})
print(comparacao.head())

# Boxplot da coluna "Amount" por "Class"
plt.figure(figsize=(8, 6))
plt.boxplot([X_train[y_train == 0]['Amount'], X_train[y_train == 1]['Amount']], labels=['Não Fraude', 'Fraude'])
plt.title('Boxplot da coluna "Amount" por "Class" (Fraude vs Não Fraude)')
plt.xlabel('Class (0 = Não Fraude, 1 = Fraude)')
plt.ylabel('Valor da Transação')
plt.show()

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Fraude', 'Fraude']).plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# Curva ROC
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

# Distribuição de Previsões
plt.figure(figsize=(8, 6))
sns.countplot(x=y_pred)
plt.title('Distribuição de Previsões')
plt.xlabel('Classe Prevista')
plt.ylabel('Quantidade')
plt.xticks([0, 1], ['Não Fraude', 'Fraude'])
plt.show()

# Colocar tudo em um Subgráfico apenas
# fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # 2 linhas e 2 colunas de subgráficos

# # Ajustando o layout para evitar sobreposição
# plt.tight_layout(pad=4.0)

# # Boxplot da coluna "Amount" por "Class"
# axs[0, 0].boxplot([X_train[y_train == 0]['Amount'], X_train[y_train == 1]['Amount']], labels=['Não Fraude', 'Fraude'])
# axs[0, 0].set_title('Boxplot da coluna "Amount" por "Class" (Fraude vs Não Fraude)')
# axs[0, 0].set_xlabel('Class (0 = Não Fraude, 1 = Fraude)')
# axs[0, 0].set_ylabel('Valor da Transação')

# # Matriz de Confusão
# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Fraude', 'Fraude']).plot(cmap='Blues', ax=axs[0, 1])
# axs[0, 1].set_title('Matriz de Confusão')

# # Curva ROC
# y_pred_proba = best_model.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)

# axs[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
# axs[1, 0].plot([0, 1], [0, 1], 'k--', label='Aleatório')
# axs[1, 0].set_xlabel('Taxa de Falsos Positivos')
# axs[1, 0].set_ylabel('Taxa de Verdadeiros Positivos')
# axs[1, 0].set_title('Curva ROC')
# axs[1, 0].legend()

# # Distribuição de Previsões
# sns.countplot(x=y_pred, ax=axs[1, 1])
# axs[1, 1].set_title('Distribuição de Previsões')
# axs[1, 1].set_xlabel('Classe Prevista')
# axs[1, 1].set_ylabel('Quantidade')
# axs[1, 1].set_xticklabels(['Não Fraude', 'Fraude'])

# # Exibindo todos os gráficos
# plt.show()