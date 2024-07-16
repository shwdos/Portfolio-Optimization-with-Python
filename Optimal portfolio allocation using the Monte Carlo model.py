#Nombre d'actifs dans le portefeuille
number_assets = len(returns.columns)

#Nombre de simulations Monte Carlo
number_simulations = 1000

#Initialisation des résultats : 
#Pour stocker rendement, volatilité, ratio de Sharpe et poids des actifs
results = np.zeros((number_simulations, 3 + number_assets)) 

# Simulation Monte Carlo
for i in range(number_simulations):
    # Génération de poids aléatoires pour les actifs
    weights = np.random.rand(num_assets)
    weights /= np.sum(weights) 
    
    # Calcul du rendement du portefeuille simulé (252 jours de trading)
    portfolio_output = np.sum(returns.mean() * weights) * 252  
    
    # Calcul de la volatilité du portefeuille simulé
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Calcul du ratio de Sharpe du portefeuille simulé (avec rf=0)
    sharpe_ratio = portfolio_output / portfolio_volatility
    
    # Stockage des résultats
    results[i, 0] = portfolio_output
    results[i, 1] = portfolio_volatility
    results[i, 2] = sharpe_ratio
    results[i, 3:] = weights

# Trouver l'indice du portefeuille avec le meilleur ratio de Sharpe
best_sharpe_ratio_ = np.argmax(results[:, 2])

# Extraire les informations du portefeuille optimal
best_portfolio_output = results[best_sharpe_ratio_, 0]
best_portfolio_volatility = results[best_sharpe_ratio_, 1]
best_sharpe_ratio = results[best_sharpe_ratio_, 2]
best_portfolio_weights = results[best_sharpe_ratio_, 3:]

# Affichage des résultats du portefeuille optimal
print(f"Portefeuille avec le ratio de Sharpe le plus élevé :")
print(f"Rendement attendu : {best_portfolio_output:.4f}")
print(f"Volatilité attendue : {best_portfolio_volatility:.4f}")
print(f"Ratio de Sharpe : {best_sharpe_ratio:.4f}")
print("Répartition des poids des actifs :")
for i in range(num_assets):
    print(f"{df.columns[i]} : {best_portfolio_weights[i]:.4f}")

# Visualisation des résultats
plt.figure(figsize=(12, 8))
plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis', marker='o', alpha=0.7)
plt.title('Simulation Monte Carlo pour une allocation optimale du portefeuille')
plt.xlabel('Volatilité attendue')
plt.ylabel('Rendement attendu')
plt.colorbar(label='Ratio de Sharpe')
plt.scatter(best_portfolio_output, best_portfolio_volatility, marker='*', color='r', s=300, label='Meilleur Ratio de Sharpe')
plt.legend()
plt.grid(True)
plt.show()
