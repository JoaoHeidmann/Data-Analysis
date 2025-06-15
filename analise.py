import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Carregamento dos Dados
# -------------------------------

df = pd.read_csv("train.csv")

print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nValores nulos por coluna:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# -------------------------------
# 2. Análise Exploratória
# -------------------------------

sns.set(style="whitegrid")

# Gráfico 1: Distribuição do Preço dos Imóveis
plt.figure(figsize=(10, 6))
sns.histplot(df["SalePrice"], kde=True, color="green", bins=40)
plt.title("Distribuição dos Preços dos Imóveis")
plt.xlabel("Preço de Venda (SalePrice)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# Gráfico 2: Mapa de Correlação
plt.figure(figsize=(15, 12))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True, mask=correlation.isnull())
plt.title("Mapa de Correlação entre Variáveis Numéricas")
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Modelagem com Random Forest
# -------------------------------

# Seleção de variáveis (features) mais relevantes
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "YearBuilt"]
target = "SalePrice"

# Subconjunto dos dados
df_model = df[features + [target]]

# Separar variáveis independentes (X) e alvo (y)
X = df_model[features]
y = df_model[target]

# Imputação de valores nulos com mediana
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Criar e treinar modelo Random Forest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nAvaliação do Modelo Random Forest:")
print(f"Erro quadrático médio (MSE): {mse:.2f}")
print(f"Coeficiente de determinação (R²): {r2:.2f}")

# -------------------------------
# 4. Validação Cruzada
# -------------------------------

cv_scores = cross_val_score(modelo, X_imputed, y, cv=5, scoring="r2")
print("\nValidação Cruzada (R²) - 5 Folds:")
print("Scores:", cv_scores)
print(f"Média dos R²: {cv_scores.mean():.2f}")
