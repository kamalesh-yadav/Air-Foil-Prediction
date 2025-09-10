import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Example: Dummy dataset with 5 features
X = np.random.rand(100, 5)   # 100 rows, 5 features
y = np.random.randint(0, 2, 100)  # Binary target (0 or 1)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save to pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl retrained with 5 features!")
