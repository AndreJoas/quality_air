# ===================== IMPORTAÇÕES =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
import optuna

# ===================== PARÂMETROS GLOBAIS =====================
limites_por_unidade = {
    'µg/m³': {
        'PM10': 500, 'NO': 500, 'PM2.5': 300, 'O3': 1000,
        'CO': 3200, 'NO2': 3000, 'SO2': 2100, 'NOX': 800,
        'BC': 20, 'PM1': 80,
    },
    'ppm': {
        'PM10': 0.5, 'NO': 0.6, 'PM2.5': 0.2, 'O3': 0.2,
        'CO': 40, 'NO2': 0.3, 'SO2': 0.2, 'NOX': 0.4,
        'BC': 0.01, 'PM1': 0.15,
    }
}

# ===================== FUNÇÕES =====================
def carregar_e_tratar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    df[['Latitude', 'Longitude']] = df['Coordinates'].str.strip().str.split(',', expand=True)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    for col in ['Source Name', 'Location']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df_validos = df.dropna(subset=['City', 'Latitude', 'Longitude']).copy()
    df_nulos = df[df['City'].isnull() & df['Latitude'].notnull() & df['Longitude'].notnull()].copy()

    if not df_nulos.empty:
        le = LabelEncoder()
        df_validos['City_Code'] = le.fit_transform(df_validos['City'])
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(df_validos[['Latitude', 'Longitude']], df_validos['City_Code'])
        y_pred = knn.predict(df_nulos[['Latitude', 'Longitude']])
        df.loc[df_nulos.index, 'City'] = le.inverse_transform(y_pred)

    df = df[df['Value'] >= 0].drop_duplicates()
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

    combined_mask = pd.Series(False, index=df.index)
    for unidade, limites in limites_por_unidade.items():
        for poluente, limite in limites.items():
            mask = (df['Unit'] == unidade) & (df['Pollutant'] == poluente) & (df['Value'] > limite)
            combined_mask |= mask

    return df.loc[~combined_mask].copy()


def classificar_qualidade(poluente, valor):
    if pd.isnull(valor):
        return 'Desconhecido'
    if poluente == 'PM10':
        if valor <= 80: return 'Boa'
        elif valor <= 240: return 'Regular'
        elif valor <= 375: return 'Inadequada'
        elif valor <= 625: return 'Má'
        elif valor <= 875: return 'Péssima'
        else: return 'Crítica'
    elif poluente == 'PM2.5':
        if valor <= 60: return 'Boa'
        elif valor <= 150: return 'Regular'
        elif valor <= 250: return 'Inadequada'
        elif valor <= 350: return 'Má'
        elif valor <= 420: return 'Péssima'
        else: return 'Crítica'
    elif poluente == 'NO2':
        if valor <= 80: return 'Boa'
        elif valor <= 365: return 'Regular'
        elif valor <= 586: return 'Inadequada'
        elif valor <= 800: return 'Má'
        elif valor <= 1000: return 'Péssima'
        else: return 'Crítica'
    elif poluente == 'SO2':
        if valor <= 50: return 'Boa'
        elif valor <= 150: return 'Regular'
        elif valor <= 250: return 'Inadequada'
        elif valor <= 350: return 'Má'
        elif valor <= 420: return 'Péssima'
        else: return 'Crítica'
    elif poluente == 'O3':
        if valor <= 100: return 'Boa'
        elif valor <= 320: return 'Regular'
        elif valor <= 1130: return 'Inadequada'
        elif valor <= 2260: return 'Má'
        elif valor <= 3000: return 'Péssima'
        else: return 'Crítica'
    elif poluente == 'CO':
        if valor <= 4.5: return 'Boa'
        elif valor <= 9.0: return 'Regular'
        elif valor <= 12.4: return 'Inadequada'
        elif valor <= 30: return 'Má'
        elif valor <= 40: return 'Péssima'
        else: return 'Crítica'
    elif poluente == 'BC':
        if valor <= 20: return 'Boa'
        else: return 'Crítica'
    else:
        return 'Desconhecido'


# ===================== SCRIPT PRINCIPAL =====================
if __name__ == "__main__":
    caminho_arquivo = 'data/world_air_quality.csv'
    df = carregar_e_tratar_dados(caminho_arquivo)

    df = df[df['Unit'] == 'µg/m³'].copy()
    df['qualidade_ar'] = df.apply(lambda row: classificar_qualidade(row['Pollutant'], row['Value']), axis=1)
    df = df[df['qualidade_ar'] != 'Desconhecido']

    treino, teste = train_test_split(df, test_size=0.15, stratify=df['qualidade_ar'], random_state=42)

    base_balanceada = []
    for _, g in treino.groupby('qualidade_ar'):
        classe = g['qualidade_ar'].iloc[0]
        n_desejado = 4500 if classe in ['Péssima', 'Má', 'Inadequada'] else 1000
        g_resample = resample(g, replace=True, n_samples=n_desejado, random_state=42)
        base_balanceada.append(g_resample)

    base_balanceada = pd.concat(base_balanceada, ignore_index=True)

    X_train = base_balanceada[['Pollutant', 'City', 'Latitude', 'Longitude', 'Value']].dropna()
    y_train = base_balanceada.loc[X_train.index, 'qualidade_ar']

    X_test = teste[['Pollutant', 'City', 'Latitude', 'Longitude', 'Value']].dropna()
    y_test_real = teste.loc[X_test.index, 'qualidade_ar']

    cat_cols = ['Pollutant', 'City']
    num_cols = ['Latitude', 'Longitude', 'Value']
    preprocessador = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    LIMITE_POR_CLASSE = 8000
    k_neighbors_value = 1

    def objective(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 350),
            max_depth=trial.suggest_int("max_depth", 6, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 4, 20),
            max_features=trial.suggest_categorical("max_features", ['sqrt', 'log2']),
            class_weight='balanced',
            random_state=42, n_jobs=-1
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            X_tr_transf = preprocessador.fit_transform(X_tr)
            X_val_transf = preprocessador.transform(X_val)

            rus = RandomUnderSampler(sampling_strategy={cls: min(LIMITE_POR_CLASSE, cnt) for cls, cnt in Counter(y_tr).items()}, random_state=42)
            X_rus, y_rus = rus.fit_resample(X_tr_transf, y_tr)

            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_value,
                          sampling_strategy={cls: LIMITE_POR_CLASSE for cls in set(y_rus)})
            X_res, y_res = smote.fit_resample(X_rus, y_rus)

            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_val_transf)
            scores.append(f1_score(y_val, y_pred, average='macro'))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    with open('melhores_parametros.json', 'w') as f:
        json.dump(study.best_params, f)

    modelos = {
        "Random Forest": RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    }

    resultados = []
    from sklearn.preprocessing import LabelEncoder

    # Encoder para XGBoost (apenas)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test_real)

    resultados = []

    for nome_modelo, modelo in modelos.items():
        print(f"\n🧪 Treinando e avaliando {nome_modelo}...")

        if nome_modelo == "XGBoost":
            y_treino_pipeline = y_train_encoded
            y_teste_pipeline = y_test_encoded
        else:
            y_treino_pipeline = y_train
            y_teste_pipeline = y_test_real

        pipeline = ImbPipeline([
            ('preprocessador', preprocessador),
            ('undersampler', RandomUnderSampler(
                sampling_strategy={cls: min(LIMITE_POR_CLASSE, count) for cls, count in pd.Series(y_treino_pipeline).value_counts().items()},
                random_state=42)),
            ('smote', SMOTE(
                random_state=42,
                k_neighbors=k_neighbors_value,
                sampling_strategy={cls: LIMITE_POR_CLASSE for cls in pd.Series(y_treino_pipeline).value_counts().index})),
            ('modelo', modelo)
        ])

        pipeline.fit(X_train, y_treino_pipeline)
        y_pred = pipeline.predict(X_test)

        # Decodifica os rótulos se for o XGBoost
        if nome_modelo == "XGBoost":
            y_pred = label_encoder.inverse_transform(y_pred)
            y_teste_pipeline = label_encoder.inverse_transform(y_teste_pipeline)

        acc = accuracy_score(y_teste_pipeline, y_pred)
        prec = precision_score(y_teste_pipeline, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_teste_pipeline, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_teste_pipeline, y_pred, average='macro', zero_division=0)

        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Macro: {f1:.4f}")

        resultados.append({
            'Modelo': nome_modelo,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Macro': f1,
            'Confusion_Matrix': confusion_matrix(y_teste_pipeline, y_pred)
        })


    df_resultados = pd.DataFrame(resultados).sort_values(by='F1_Macro', ascending=False)
    print("\n🏆 Ranking dos Modelos (ordenado por F1 Macro):\n")
    print(df_resultados[['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1_Macro']])

    for res in resultados:
        plt.figure(figsize=(8, 6))
        sns.heatmap(res['Confusion_Matrix'], annot=True, fmt='d',
                    xticklabels=sorted(y_test_real.unique()),
                    yticklabels=sorted(y_test_real.unique()),
                    cmap='Blues')
        plt.title(f"Matriz de Confusão - {res['Modelo']}")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.show()
