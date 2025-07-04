import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.DataFrame({
    'serialno': np.random.normal(0, 1, 100),
    'price': np.random.normal(5, 2, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.randint(0, 2, 100)
})

for col in ['serialno', 'price']:
    plt.figure()
    df[col].hist()
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()