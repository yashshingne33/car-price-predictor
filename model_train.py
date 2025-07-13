import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Step 1: Load CSV
df = pd.read_csv('quikr_car.csv')

# Step 2: Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
print("Cleaned column names:", df.columns.tolist())

# Step 3: Clean and preprocess individual columns
df['fuel_type'] = df['fuel_type'].astype(str).str.strip()
df['price'] = df['price'].astype(str).str.strip()
df['kms_driven'] = df['kms_driven'].astype(str).str.strip()
df['year'] = df['year'].astype(str).str.strip()

# Remove rows with 'Ask For Price'
df = df[df['price'].str.lower() != 'ask for price']

# Convert price to float
df['price'] = df['price'].str.replace(',', '', regex=False)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Clean kms_driven and convert to numeric
print("🔍 Before cleaning kms_driven:", df['kms_driven'].unique()[:10])
print("")
df['kms_driven'] = df['kms_driven'].str.replace('kms', '', regex=False)
df['kms_driven'] = df['kms_driven'].str.replace(',', '', regex=False)
df['kms_driven'] = df['kms_driven'].str.extract(r'(\d+)', expand=False)
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

# Set print options for better display
np.set_printoptions(suppress=True)
print("After cleaning kms_driven values :", df['kms_driven'].head(10).tolist())
print("")

# Convert year to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Drop rows with missing key values
df.dropna(subset=['price', 'kms_driven', 'year', 'fuel_type'], inplace=True)
df.drop('name', axis=1, inplace=True) #cloumn "name" has been dropped as it is complex and noisy
df = df[df['year'].notnull()]  # Remove rows with missing year
df['year'] = df['year'].astype(int)
df['kms_driven'] = df['kms_driven'].astype(int)



# Debug: shape and structure
print("🟢 Final shape after cleaning:", df.shape)
print(df.head())

# Step 4: Feature engineering
df['car_age'] = 2025 - df['year']
df.drop(['year', 'name'], axis=1, inplace=True, errors='ignore')

# Drop duplicates
df.drop_duplicates(inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Step 5: Train/test prep
X = df.drop('price', axis=1)
y = df['price']

# Debug info
print("🟡 Final dataset shape before split:", df.shape)
print("✅ Nulls in price:", df['price'].isnull().sum())
print("✅ Nulls in kms_driven:", df['kms_driven'].isnull().sum())
print("🧠 Features used for training:", X.columns.tolist())

# Step 6: Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Step 8: Save model
with open('linearregressionmodel.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'linearregressionmodel.pkl'")
