import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

age = np.random.randint(18, 70, size=n)
monthly_spending = np.random.uniform(20, 150, size=n).round(2)
login_frequency = np.random.randint(1, 30, size=n)
support_tickets = np.random.randint(0, 10, size=n)
subscription = np.random.choice(['Basic', 'Standard', 'Premium'], size=n)

# Calcul d'un score de risque de churn logique
churn_score = (
    0.3 * support_tickets 
    - 0.15 * login_frequency 
    - 0.01 * monthly_spending 
    + np.random.normal(0, 1, size=n)
)

# Convertir le score en probabilité (Sigmoïde) puis en 0 ou 1
prob = 1 / (1 + np.exp(-churn_score))
churn = (prob > 0.4).astype(int)

df = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'age': age,
    'monthly_spending': monthly_spending,
    'login_frequency': login_frequency,
    'support_tickets': support_tickets,
    'subscription': subscription,
    'churn': churn
})

df.to_csv('churn_data.csv', index=False)
print("Fichier churn_data.csv régénéré avec des corrélations réalistes !")