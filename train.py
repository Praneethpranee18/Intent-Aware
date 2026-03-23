import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ── 1. Generate Synthetic Dataset ─────────────────────────────────────────────

np.random.seed(42)
n = 20000

user_types = np.random.choice(['new', 'returning', 'loyal'], n, p=[0.3, 0.45, 0.25])
categories = np.random.choice(
    ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty', 'Sports'], n,
    p=[0.2, 0.25, 0.2, 0.1, 0.15, 0.1]
)

category_price_map = {
    'Electronics':    (80,  1200),
    'Clothing':       (15,  300),
    'Home & Kitchen': (20,  500),
    'Books':          (5,   60),
    'Beauty':         (10,  200),
    'Sports':         (25,  400),
}

purchase_frequency = np.where(
    user_types == 'loyal',      np.random.randint(8, 20, n),
    np.where(user_types == 'returning', np.random.randint(3, 10, n),
             np.random.randint(1, 4, n))
)

avg_order_value = np.where(
    user_types == 'loyal',      np.round(np.random.uniform(80, 300, n), 2),
    np.where(user_types == 'returning', np.round(np.random.uniform(40, 150, n), 2),
             np.round(np.random.uniform(20, 80, n), 2))
)

product_price = np.array([
    round(np.random.uniform(*category_price_map[c]), 2) for c in categories
])

rating       = np.round(np.random.uniform(1.0, 5.0, n), 1)
discount_pct = np.random.choice(
    [0, 5, 10, 15, 20, 25, 30, 40, 50], n,
    p=[0.25, 0.1, 0.15, 0.1, 0.15, 0.1, 0.08, 0.05, 0.02]
)

score = (
    (user_types == 'loyal').astype(float)     * 0.35 +
    (user_types == 'returning').astype(float) * 0.20 +
    (purchase_frequency / 20)                 * 0.15 +
    (rating / 5.0)                            * 0.20 +
    (discount_pct / 50)                       * 0.15 +
    np.random.uniform(0, 0.15, n)
)
purchase = (score > 0.45).astype(int)

df = pd.DataFrame({
    'user_type':           user_types,
    'purchase_frequency':  purchase_frequency,
    'avg_order_value':     avg_order_value,
    'product_category':    categories,
    'product_price':       product_price,
    'rating':              rating,
    'discount_percentage': discount_pct,
    'purchase':            purchase,
})

df.to_csv('data.csv', index=False)
print(f"✔ Dataset saved → data.csv  ({n:,} rows, {df.shape[1]} columns)")
print(f"  Purchase rate : {df['purchase'].mean():.2%}")

# ── 2. Preprocessing ───────────────────────────────────────────────────────────

df_enc = df.copy()

le_user     = LabelEncoder()
le_category = LabelEncoder()

df_enc['user_type']        = le_user.fit_transform(df_enc['user_type'])
df_enc['product_category'] = le_category.fit_transform(df_enc['product_category'])

print("\n  Label encoding mappings")
print(f"  user_type        : {dict(zip(le_user.classes_, le_user.transform(le_user.classes_)))}")
print(f"  product_category : {dict(zip(le_category.classes_, le_category.transform(le_category.classes_)))}")

X = df_enc.drop(columns='purchase')
y = df_enc['purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")

# ── 3. Train Random Forest ─────────────────────────────────────────────────────

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

print("\n  Training Random Forest …")
model.fit(X_train, y_train)
print("✔ Training complete")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n── Evaluation ────────────────────────────────")
print(f"  ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase']))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
print(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

print("\n  Feature Importances:")
importances = sorted(
    zip(X.columns, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
for feat, imp in importances:
    bar = '█' * int(imp * 60)
    print(f"  {feat:<22} {imp:.4f}  {bar}")

# ── 5. Save Model ──────────────────────────────────────────────────────────────

payload = {
    'model':            model,
    'le_user':          le_user,
    'le_category':      le_category,
    'feature_columns':  list(X.columns),
}

with open('model.pkl', 'wb') as f:
    pickle.dump(payload, f)

print("\n✔ Model saved → model.pkl")
print("\n  To load and predict:")
print("    import pickle")
print("    with open('model.pkl', 'rb') as f:")
print("        payload = pickle.load(f)")
print("    model = payload['model']")
print("    preds = model.predict(X_new)")