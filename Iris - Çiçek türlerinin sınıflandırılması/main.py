import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Iris.csv")
print(df.head())

if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

print(df['species'].value_counts())

print(df.describe())

def cicek_tur_dagilim(df):
    sns.countplot(x='species', data=df, color='pink')
    plt.title('Çiçek Türlerinin Dağılımı')
    plt.show()

cicek_tur_dagilim(df)

def ozellik_dagilimi(df):
    sns.pairplot(df, hue='species', palette='pastel')
    plt.show()

ozellik_dagilimi(df)

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)