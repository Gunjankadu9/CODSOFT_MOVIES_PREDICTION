import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('IMDb Movies India.csv', encoding='latin-1')

print(df.head())
df.info()

# Target cleanup
if 'Rating' not in df.columns:
    raise KeyError("'Rating' column not found in dataset")
df.dropna(subset=['Rating'], inplace=True)
print(f"DataFrame shape after dropping rows with NaN in 'Rating': {df.shape}")

# Extract numeric values from year and duration
if 'Year' in df.columns:
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0]
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

if 'Duration' in df.columns:
    df['Duration'] = df['Duration'].astype(str).str.replace(' min', '', regex=False)
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

if 'Votes' in df.columns:
    df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=False)
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Fill missing values for text columns
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    if col in df.columns:
        df[col].fillna('Unknown', inplace=True)

print('Missing values after filling NaNs:')
print(df.isnull().sum())

if 'Duration' in df.columns:
    df['Duration'].fillna(df['Duration'].median(), inplace=True)
if 'Year' in df.columns:
    df['Year'].fillna(df['Year'].median(), inplace=True)
if 'Votes' in df.columns:
    df['Votes'].fillna(df['Votes'].median(), inplace=True)

print(f"Missing values in 'Duration' after filling: {df['Duration'].isnull().sum() if 'Duration' in df.columns else 0}")
print(df.info())

all_genres = set()
if 'Genre' in df.columns:
    for genres in df['Genre']:
        if isinstance(genres, str):
            for genre in genres.split(', '):
                all_genres.add(genre.strip())
        else:
            all_genres.add('Unknown')

print(f"Total unique genres: {len(all_genres)}")
print(f"Unique genres: {list(all_genres)[:10]}...")

for genre in sorted(all_genres):
    df[genre] = df['Genre'].apply(lambda x: 1 if isinstance(x, str) and genre in x else 0)

print(df.head())


def get_top_n_dummies(df, column, n=50):
    top_values = df[column].value_counts().nlargest(n).index.tolist()
    df_dummies = pd.get_dummies(df[column].apply(lambda x: x if x in top_values else 'Other'), prefix=column)
    return df_dummies

for column, n in [('Director', 100), ('Actor 1', 100), ('Actor 2', 100), ('Actor 3', 100)]:
    if column in df.columns:
        dummies = get_top_n_dummies(df, column, n=n)
        df = pd.concat([df, dummies], axis=1)

columns_to_drop = [col for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Name'] if col in df.columns]
df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

print('DataFrame shape after feature engineering:', df.shape)
print(df.head())

X = df.drop('Rating', axis=1)
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual', y='Predicted', data=results_df, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Movie Ratings')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.grid(True)
plt.show()

errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
