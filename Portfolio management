# Calcul des rendements quotidiens
returns = df.pct_change().dropna()

# Définition des poids pour chaque actif (20% pour chaque actif)
num_assets = len(df.columns)
weights = np.array([0.2] * num_assets)

# Calcul des rendements cumulés ajustés pour chaque actif
cumulative_returns = (weights * returns).cumsum()

# Plot des rendements cumulés de chaque actif
plt.figure(figsize=(18, 9))

for column in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)

plt.title('Rendements cumulés des actifs du portefeuille en fonction du temps')
plt.xlabel('Date (en année)')
plt.ylabel('Rendements cumulés')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Calcul de la Value-at-Risk (VaR) à 95% pour 1 jour
var_95_1D = np.percentile(returns.sum(axis=1), 5)
print(f"Value-at-Risk (VaR) au niveau de confiance de 95% à une horizon de 1 jour : {var_95_1D:.4f}")

# Calcul de la Conditional Value-at-Risk (CVaR) à 99% pour 1 mois
cvar_99_1M = np.mean(returns[returns.sum(axis=1) <= np.percentile(returns.sum(axis=1), 1)].sum(axis=1))
print(f"Conditional Value-at-Risk (CVaR) au niveau de confiance de 99% à une horizon de 1 mois : {cvar_99_1M:.4f}")
