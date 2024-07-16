nasdaq['Daily_Output'] = nasdaq['Adj Close'].pct_change().dropna()

# Sélection des données jusqu'au 19 janvier 2022 inclus
nasdaq_bf_jan19 = nasdaq[nasdaq['Date'] <= '19/01/2022']

# Extraction du prix de clôture ajusté pour le NASDAQ le 19 janvier 2022
price_190122 = nasdaq[nasdaq['Date'] == '19/01/2022']['Adj Close'].values[0]

n = 10

def features(data, price_190122, n):
    x = []
    y = []
    for i in range(len(data) - n):
        #Calcul des rendements log sur n jours glissants
        response = np.log(data['Adj Close'].iloc[i:i+n] / price_190122)
        x.append(response)
        y.append(data['Adj Close'].iloc[i + n])
    
    return np.array(x), np.array(y)

# Obtention des caractéristiques (X) et de la cible (Y)
x, y = features(nasdaq_bf_jan19, price_190122, n)

print("Dimensions de x et y après création des caractéristiques :", x.shape, y.shape)

# Séparation des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Initialisation du modèle SVM
svr = SVR()

# Entraînement du modèle SVM sur les données d'entraînement
svr.fit(x_train, y_train)

# Prédictions sur l'ensemble de test
predictions = svr.predict(x_test)

# Définition de la grille des hyperparamètres à tester
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear']
}

# Recherche des meilleurs hyperparamètres avec GridSearchCV
grid = GridSearchCV(svr, param_grid, refit=True, verbose=2)
grid.fit(x_train, y_train)

print("Meilleurs paramètres trouvés :", grid.best_params_)

# Meilleur modèle SVM trouvé
best_svr_model = grid.best_estimator_

# Prédiction sur l'ensemble de test
y_pred = best_svr_model.predict(x_test)

# Évaluation du modèle en calculant la racine carrée de l'erreur quadratique moyenne (RMSE) entre les prédictions et les valeurs réelles
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Prédiction du prix du Nasdaq pour le 2 février 2022
latest_data = nasdaq['Daily_Output'].iloc[-n:].values.reshape(1, -1)
predicted_price = best_svr_model.predict(latest_data)
print(f"Prix du Nasdaq le 2 février 2022 : {predicted_price[0]} USD")

# Visualisation des résultats (optionnel)
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Prix réel')
plt.plot(y_pred, label='Prix prédit')
plt.title('Prix réels vs Prix prédits du Nasdaq')
plt.xlabel('Rendements log')
plt.ylabel('Prix de clôture ajusté')
plt.legend()
plt.show()
