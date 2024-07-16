# Load the CSV file
nasdaq = pd.read_csv('Nasdaq.csv', index_col=0, parse_dates=True, dayfirst=True)  
nasdaq

nasdaq_evolution = nasdaq.loc['01/01/2020':'31/12/2020', 'Adj Close']

plt.figure(figsize=(12,8))
plt.plot(nasdaq_evolution.index, nasdaq_evolution, marker='o', linestyle='-', color='black', label='Nasdaq Adj Close')
plt.title('Évolution du Nasdaq entre 2020-2021')
plt.xlabel('Date')
plt.ylabel('Prix de fermeture ajustés')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show

nasdaq['Daily_Output'] = nasdaq['Adj Close'].pct_change()

#Calcul pour l'ensemble de la période disponible : 1971-2022

mean_all = nasdaq['Daily_Output'].mean()
volatility_all = nasdaq['Daily_Output'].std()

print("Rendement journalier moyen sur toute la période de 1971 à 2022 est de : {:.6f}".format(mean_all))
print("La volatilité sur toute la période est de : {:.6f}".format(mean_all))

#Calcul pour la période : 2020-2021

nasdaq_2020_21 = nasdaq.loc['01/01/2020':'31/12/2020']
mean_2020_21 = nasdaq_2020_21['Daily_Output'].mean()
volatility_2020_21 = nasdaq_2020_21['Daily_Output'].std()

print("Rendement journalier moyen sur la période de 2020 à 2021 est de : {:.6f}".format(mean_2020_21))
print("La volatilité sur la période 2020 à 2021 est de : {:.6f}".format(mean_2020_21))

trade_days_per_year = 252 

#Annualiser toute la période 
annualized_mean_all = (1 + mean_all) ** trade_days_per_year - 1
annualized_volatility_all = volatility_all * np.sqrt(trade_days_per_year)

print("Rendement journalier moyen sur toute la période de 1971 à 2022 est de : {:.2f}".format(annualized_mean_all))
print("La volatilité sur toute la période est de : {:.2f}".format(annualized_volatility_all))

#Annualiser la période de 2020-2021
annualized_mean_2020_21 = (1 + mean_2020_21) ** trade_days_per_year - 1
annualized_volatility_2020_21 = volatility_2020_21 * np.sqrt(trade_days_per_year)

print("Rendement journalier moyen sur la période de 2020 à 2021 est de : {:.2f}".format(annualized_mean_2020_21))
print("La volatilité sur la période 2020 à 2021 est de : {:.2f}".format(annualized_volatility_2020_21))

#Nettoyage des données manquantes

nasdaq['Daily_Output'] = nasdaq['Adj Close'].pct_change().dropna()
nasdaq_update = nasdaq[np.isfinite(nasdaq['Daily_Output'])]

# Histogramme

plt.figure(figsize=(16,6))
plt.hist(nasdaq_update['Daily_Output'], bins=80, density=True, alpha=0.5, color='black', label='Rendements historiques de 1971 à 2022')

# Distribution normale

mu, sigma = norm.fit(nasdaq_update['Daily_Output'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin,xmax, 100)
p = norm.pdf(x, mu, sigma)

plt.plot(x, p, 'k', linewidth=2, label=f'Distribution normale\n$\mu$ = {mu:.6f}, $\sigma$ = {sigma:.6f}')

plt.title('Distrubution des rendements historiques de 1971 à 2022 du Nasdaq')
plt.xlabel('Rendements journaliers')
plt.ylabel('Densité de probabilité')
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()

