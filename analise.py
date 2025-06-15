import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o arquivo CSV
df = pd.read_csv("train.csv")

# -------------------------------
# ANÁLISE EXPLORATÓRIA
# -------------------------------

print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nValores nulos por coluna:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Estilo dos gráficos
sns.set(style="whitegrid")

# Gráfico 1: Distribuição do preço dos imóveis
plt.figure(figsize=(10, 6))
sns.histplot(df["SalePrice"], kde=True, color="green", bins=40)
plt.title("Distribuição dos Preços dos Imóveis")
plt.xlabel("Preço de Venda (SalePrice)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# Gráfico 2: Correlação entre variáveis numéricas
plt.figure(figsize=(15, 12))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True, mask=correlation.isnull())
plt.title("Mapa de Correlação entre Variáveis Numéricas")
plt.tight_layout()
plt.show()

# -------------------------------
# PRÉ-PROCESSAMENTO E MODELO
# -------------------------------

# Selecionar colunas numéricas relevantes para o modelo
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "YearBuilt"]
target = "SalePrice"

# Criar novo DataFrame apenas com essas colunas (e remover linhas com valores nulos)
df_model = df[features + [target]].dropna()

# Separar variáveis independentes (X) e dependente (y)
X = df_model[features]
y = df_model[target]

# Separar os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões com os dados de teste
y_pred = modelo.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nAvaliação do Modelo:")
print(f"Erro quadrático médio (MSE): {mse:.2f}")
print(f"Coeficiente de determinação (R²): {r2:.2f}")
