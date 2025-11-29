# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Load in the data
data = pd.read_csv('German_Credit_Data.txt', header=None, sep=',')

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1].values

# Encode features
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col].astype(str))

# Convert
X = X.values
y = y - 1

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Baseline with class weights
m1 = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced'
)
m1.fit(X_train_scaled, y_train)
pred1 = m1.predict(X_test_scaled)
f1_macro1 = f1_score(y_test, pred1, average='macro')
f1_weighted1 = f1_score(y_test, pred1, average='weighted')

# Print baseline results
print(f"Baseline: F1-Macro = {f1_macro1:.4f}, F1-Weighted = {f1_weighted1:.4f}")

# Method 2: Random Under-sampling
random = RandomUnderSampler(random_state=42)
X_train_random, y_train_random = random.fit_resample(X_train_scaled, y_train)

m2 = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced'
)
m2.fit(X_train_random, y_train_random)
pred2 = m2.predict(X_test_scaled)
f1_macro2 = f1_score(y_test, pred2, average='macro')
f1_weighted2 = f1_score(y_test, pred2, average='weighted')

# Print under-sampling results
print(f"Under-Sampling: F1-Macro = {f1_macro2:.4f}, F1-Weighted = {f1_weighted2:.4f}")

# Method 3: SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

m3 = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced'
)
m3.fit(X_train_smote, y_train_smote)
pred3 = m3.predict(X_test_scaled)
f1_macro3 = f1_score(y_test, pred3, average='macro')
f1_weighted3 = f1_score(y_test, pred3, average='weighted')

# Print SMOTE results
print(f"SMOTE: F1-Macro = {f1_macro3:.4f}, F1-Weighted = {f1_weighted3:.4f}")

# Method 4: SMOTE-ENN
enn = SMOTEENN(random_state=42)
X_train_combined, y_train_combined = enn.fit_resample(X_train_scaled, y_train)

m4 = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced'
)
m4.fit(X_train_combined, y_train_combined)
pred4 = m4.predict(X_test_scaled)
f1_macro4 = f1_score(y_test, pred4, average='macro')
f1_weighted4 = f1_score(y_test, pred4, average='weighted')

# Print SMOTE-ENN results
print(f"SMOTE-ENN: F1-Macro = {f1_macro4:.4f}, F1-Weighted = {f1_weighted4:.4f}")