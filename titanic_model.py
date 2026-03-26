import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─── GENERATE REALISTIC TITANIC DATASET ───────────────────────────────────────
np.random.seed(42)
n = 891

# Passenger class distribution
pclass = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])

# Sex
sex = np.random.choice(['male', 'female'], n, p=[0.647, 0.353])

# Age — with realistic missing values
age_base = np.where(pclass == 1, 
                    np.random.normal(39, 14, n),
                    np.where(pclass == 2, np.random.normal(29, 14, n),
                             np.random.normal(25, 14, n)))
age = np.clip(age_base, 1, 80).astype(float)
age[np.random.choice(n, 177, replace=False)] = np.nan  # ~20% missing

# SibSp, Parch
sibsp = np.random.choice([0,1,2,3,4,5,8], n, p=[0.68,0.23,0.05,0.02,0.01,0.005,0.005])
parch = np.random.choice([0,1,2,3,4,5,6], n, p=[0.76,0.13,0.08,0.01,0.01,0.005,0.005])

# Fare
fare = np.where(pclass == 1,
                np.random.lognormal(4.8, 0.7, n),
                np.where(pclass == 2, np.random.lognormal(3.1, 0.4, n),
                         np.random.lognormal(2.0, 0.5, n)))
fare = np.clip(fare, 0, 512)

# Embarked
embarked = np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09])

# ─── SURVIVAL LOGIC (historically realistic) ──────────────────────────────────
survival_prob = np.full(n, 0.38)  # base rate
survival_prob[sex == 'female'] += 0.35
survival_prob[sex == 'male'] -= 0.15
survival_prob[pclass == 1] += 0.22
survival_prob[pclass == 2] += 0.06
survival_prob[pclass == 3] -= 0.14

age_filled = np.where(np.isnan(age), 28, age)
survival_prob[age_filled < 10] += 0.18  # children first
survival_prob[age_filled > 60] -= 0.10
survival_prob[(sibsp > 0) & (sibsp <= 2)] += 0.04
survival_prob[(parch > 0) & (parch <= 2)] += 0.03
survival_prob[fare > 100] += 0.10
survival_prob = np.clip(survival_prob, 0.02, 0.97)
survived = (np.random.rand(n) < survival_prob).astype(int)

df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived': survived,
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
})

print("=== DATASET OVERVIEW ===")
print(df.describe())
print(f"\nSurvival Rate: {df.Survived.mean():.1%}")

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Sex_enc'] = (df['Sex'] == 'female').astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3], duplicates='drop').astype(int)
    df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Child'] = ((df['Age'] < 12) & (df['Sex'] == 'female')).astype(int) * 0 + (df['Age'] < 12).astype(int)
    df['Mother'] = ((df['Sex'] == 'female') & (df['Age'] > 18) & (df['Parch'] > 0)).astype(int)
    return df

df_eng = engineer_features(df)

FEATURES = ['Pclass','Sex_enc','Age','SibSp','Parch','Fare',
            'FamilySize','IsAlone','AgeBin','FareBin','Embarked_C','Embarked_Q','Child','Mother']

X = df_eng[FEATURES]
y = df_eng['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ─── MODELS ───────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
}

print("\n=== MODEL COMPARISON ===")
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'test_acc': test_acc}
    print(f"{name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}  Test={test_acc:.3f}")

# Best model
best_model = models['Random Forest']
y_pred = best_model.predict(X_test)
print("\n=== RANDOM FOREST CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n=== FEATURE IMPORTANCES ===")
print(feat_imp.round(4))

# Survival stats by group
print("\n=== SURVIVAL RATES BY GENDER ===")
print(df.groupby('Sex')['Survived'].mean().round(3))
print("\n=== SURVIVAL RATES BY CLASS ===")
print(df.groupby('Pclass')['Survived'].mean().round(3))
print("\n=== SURVIVAL RATES BY CLASS & GENDER ===")
print(df.groupby(['Pclass','Sex'])['Survived'].mean().unstack().round(3))

age_bins = pd.cut(df['Age'].fillna(df['Age'].median()), bins=[0,12,18,35,60,100], labels=['Child','Teen','Adult','Middle','Senior'])
print("\n=== SURVIVAL RATES BY AGE GROUP ===")
print(df.groupby(age_bins)['Survived'].mean().round(3))

# Save feature importances to JSON for the frontend
import json
fi_data = feat_imp.to_dict()
stats = {
    'overall_survival': float(df.Survived.mean()),
    'female_survival': float(df[df.Sex=='female']['Survived'].mean()),
    'male_survival': float(df[df.Sex=='male']['Survived'].mean()),
    'class1_survival': float(df[df.Pclass==1]['Survived'].mean()),
    'class2_survival': float(df[df.Pclass==2]['Survived'].mean()),
    'class3_survival': float(df[df.Pclass==3]['Survived'].mean()),
    'feature_importances': fi_data,
    'model_accuracy': results['Random Forest']['test_acc'],
    'model_cv': results['Random Forest']['cv_mean']
}
print("\n=== STATS JSON ===")
print(json.dumps(stats, indent=2))
