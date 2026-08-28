import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'age': np.random.randint(18, 70, size=n),
    'monthly_spending': np.random.uniform(20, 150, size=n).round(2),
    'login_frequency': np.random.randint(1, 30, size=n),
    'support_tickets': np.random.randint(0, 10, size=n),
    'subscription': np.random.choice(['Basic', 'Standard', 'Premium'], size=n),
    'churn': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
})

df.to_csv('churn_data.csv', index=False)
print("Fichier churn_data.csv créé avec succès !")