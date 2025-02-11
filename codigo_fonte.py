import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("creditcard.csv")

# Função para mostrar informações gerais do dataset
def dataset_info(df):
    print("\nResumo do Dataset:")
    print(df.info())
    print("\nValores Nulos:")
    print(df.isnull().sum())
    print("\nDuplicatas:", df.duplicated().sum())
    print("\nDistribuição da Classe:")
    print(df['Class'].value_counts(normalize=True))

dataset_info(df)

# Normalização da variável "Amount"
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Separação dos dados em treino e teste
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Treinamento do modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Validação cruzada para avaliação mais robusta
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Acurácia média na validação cruzada: {cv_scores.mean():.4f}")

# Treinando o modelo no conjunto de treino
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
print("\nAcurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.title("Matriz de Confusão")
plt.xlabel('Predição')
plt.ylabel('Real')
plt.show()

# Curva ROC e AUC
y_probs = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print(f"\nAUC: {roc_auc:.2f}")

# Plot da Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Exibindo a importância das features
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nImportância das Features:")
print(importance_df)

print("\nDistribuição da Classe:")
print(df['Class'].value_counts(normalize=True))
