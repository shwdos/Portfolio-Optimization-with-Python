d = 0
while adfuller(us_cpi['US CPI'])[1] > 0.05:
    us_cpi['US CPI'] = us_cpi['US CPI'].diff().dropna()
    d += 1

print(f"Nombre de différenciations nécessaires (d) : {d}")

# Tracer les ACF et PACF pour déterminer p et q
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(us_cpi.dropna(), ax=plt.gca(), lags=20)
plt.title('ACF')
plt.subplot(212)
plot_pacf(us_cpi.dropna(), ax=plt.gca(), lags=20)
plt.title('PACF')
plt.show()

p = 1  
q = 1  

# Construire et ajuster le modèle ARIMA
model = ARIMA(us_cpi['US CPI'], order=(p, d, q))
model_fit = model.fit()

# Résumé des résultats du modèle ARIMA
print('Résumé des résultats du modèle ARIMA :')
print(model_fit.summary())

# Tracé des résidus du modèle ARIMA
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Résidus du modèle ARIMA')
plt.xlabel('Date (en année)')
plt.ylabel('Résidus')
plt.grid(True)
plt.show()

# Test de stationnarité des résidus avec le test de Dickey-Fuller augmenté (ADF)
print('Résultats du test de Dickey-Fuller augmenté sur les résidus :')
adf_test = adfuller(residuals.dropna(), autolag='AIC')
adf_output = pd.Series(adf_test[0:4], index=['Test Statistique', 'p-value', 'Lags Utilisés', 'Nombre d\'observations utilisées'])
for key, value in adf_test[4].items():
    adf_output[f'Valeur critique ({key})'] = value
print(adf_output)

# Prévision des 12 prochains mois
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)

# Dates des prédictions
last_date = us_cpi.index[-1]
forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_steps + 1)]

# Création d'un dataframe pour stocker les prédictions
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Prévision de l\'inflation US': forecast
})
forecast_df.set_index('Date', inplace=True)

# Visualisation des données historiques et prévisions
plt.figure(figsize=(12, 6))
plt.plot(us_cpi.index, us_cpi['US CPI'], label='Inflation historique US')
plt.plot(forecast_df.index, forecast_df['Prévision de l\'inflation US'], label='Prévision de l\'inflation US', linestyle='--', color='orange')
plt.title('Inflation historique vs Prévisions de l\'inflation US')
plt.xlabel('Date')
plt.ylabel('Taux d\'inflation (%)')
plt.legend()
plt.grid(True)
plt.show()

# Calcul du niveau d'inflation attendu au 28 février 2025
target_date = pd.Timestamp('2025-02-28')
expected_inflation = forecast_df.loc[target_date]['Prévision de l\'inflation US'] 
print(f"Niveau d'inflation attendu au 28 février 2025 : {expected_inflation:.4f}%")
