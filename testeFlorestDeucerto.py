import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import json
from collections import Counter
from sklearn.utils import resample

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
    print("\n📦 Carregando a base de dados...")
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
        print(f"🏙️ Preenchendo {len(df_nulos)} cidades ausentes com KNN...")
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
    df_limpo = df.loc[~combined_mask].copy()

    print(f"✅ Base carregada e limpa: {df_limpo.shape[0]} registros restantes após remoção de outliers.\n")
    return df_limpo

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

    base_ugm3_original = df[df['Unit'] == 'µg/m³'].copy()
    base_ugm3_original['qualidade_ar'] = base_ugm3_original.apply(
        lambda row: classificar_qualidade(row['Pollutant'], row['Value']), axis=1)
    base_ugm3_original = base_ugm3_original[base_ugm3_original['qualidade_ar'] != 'Desconhecido']

    # 1. Fazer split treino/teste na base original (sem balanceamento)
    treino, teste = train_test_split(
        base_ugm3_original, 
        test_size=0.15, 
        random_state=7, 
        stratify=base_ugm3_original['qualidade_ar']
    )

    # 2. Balancear manualmente só o treino (com resample)
    grupos = [g for _, g in treino.groupby('qualidade_ar')]
    base_treino_balanceada = []
    for grupo in grupos:
        classe = grupo['qualidade_ar'].iloc[0]
        tamanho_original = len(grupo)

        if classe in ['Péssima', 'Má', 'Inadequada']:
            tamanho_desejado = 4500
        elif classe in ['Boa']:
            tamanho_desejado = 1000
        elif classe in ['Crítica', 'Regular']:
            tamanho_desejado = 1000
        else:
            tamanho_desejado = tamanho_original

        grupo_ampliado = resample(
            grupo, replace=True, n_samples=tamanho_desejado, random_state=42
        )
        base_treino_balanceada.append(grupo_ampliado)

    base_balanceada = pd.concat(base_treino_balanceada, ignore_index=True)
    print(f"📊 Distribuição das classes no treino balanceado:")
    print(Counter(base_balanceada['qualidade_ar']))

    # Preparar dados para treino/teste
    X_train = base_balanceada[['Pollutant', 'City', 'Latitude', 'Longitude', 'Value']].dropna()
    y_train = base_balanceada.loc[X_train.index, 'qualidade_ar']

    X_test = teste[['Pollutant', 'City', 'Latitude', 'Longitude', 'Value']].dropna()
    y_test_real = teste.loc[X_test.index, 'qualidade_ar']

    print(f"\n📊 Distribuição das classes no treino balanceado:")
    print(y_train.value_counts())
    print(f"\n📊 Distribuição das classes no teste (original):")
    print(y_test_real.value_counts())

    cat_cols = ['Pollutant', 'City']
    num_cols = ['Latitude', 'Longitude', 'Value']

    preprocessador = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    k_neighbors_value = 1
    print(f"Fixando k_neighbors={k_neighbors_value} para SMOTE.")

    LIMITE_POR_CLASSE = 8000  # limite fixo para todas as classes após undersampling e oversampling
    smote_printed = False 
    def objective(trial):
        global smote_printed
        rf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 350),  # Evite poucos estimadores
            max_depth=trial.suggest_int("max_depth", 6, 24),          # Reduza profundidade máxima
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),  # Mais folhas mínimas
            min_samples_split=trial.suggest_int("min_samples_split", 4, 20),  # Mais robusto
            max_features=trial.suggest_categorical("max_features", ['sqrt', 'log2']),  # Menos features por árvore
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )


        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            X_tr_transf = preprocessador.fit_transform(X_tr)

            contagens = Counter(y_tr)
            # undersampling para reduzir classes maiores a LIMITE_POR_CLASSE
            sampling_under_strategy = {cls: min(cnt, LIMITE_POR_CLASSE) for cls, cnt in contagens.items()}

            rus = RandomUnderSampler(sampling_strategy=sampling_under_strategy, random_state=42)
            X_rus, y_rus = rus.fit_resample(X_tr_transf, y_tr)

            contagens_rus = Counter(y_rus)

            # oversampling SMOTE para aumentar classes menores até LIMITE_POR_CLASSE
            sampling_smote_strategy = {cls: LIMITE_POR_CLASSE for cls in contagens_rus.keys()
                                       if contagens_rus[cls] < LIMITE_POR_CLASSE}

            if sampling_smote_strategy:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors_value, sampling_strategy=sampling_smote_strategy)
                X_res, y_res = smote.fit_resample(X_rus, y_rus)

                # Imprime só uma vez, no primeiro fold do primeiro trial
                if not smote_printed:
                    print(f"\n🔁 SMOTE aplicado no fold atual:")
                    print(Counter(y_res))
                    smote_printed = True

            else:
                X_res, y_res = X_rus, y_rus

            X_val_transf = preprocessador.transform(X_val)

            rf.fit(X_res, y_res)
            y_pred_val = rf.predict(X_val_transf)

            score = f1_score(y_val, y_pred_val, average='macro')
            scores.append(score)

        return np.mean(scores)


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("🔍 Melhores parâmetros encontrados:")
    print(study.best_params)

    with open('melhores_parametros.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

    rf_final = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=-1)

        # Ajuste para undersampling: nunca pede para aumentar número, só reduzir ou manter igual
    undersampler_final = RandomUnderSampler(
        sampling_strategy={cls: min(LIMITE_POR_CLASSE, count) for cls, count in y_train.value_counts().items()},
        random_state=42
    )

    # SMOTE pode aumentar até LIMITE_POR_CLASSE
    smote_final = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors_value,
        sampling_strategy={cls: LIMITE_POR_CLASSE for cls in y_train.value_counts().index}
    )

    pipeline_final = ImbPipeline([
        ('preprocessador', preprocessador),
        ('undersampler', undersampler_final),
        ('smote', smote_final),
        ('modelo', rf_final)
    ])


    pipeline_final.fit(X_train, y_train)
    y_pred = pipeline_final.predict(X_test)

    teste_sem_rotulo_final = teste.loc[X_test.index].copy()
    teste_sem_rotulo_final['qualidade_ar_prevista'] = y_pred

    print("\n📊 Relatório de Classificação:\n")
    print(classification_report(y_test_real, y_pred))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test_real, y_pred), annot=True, fmt='d',
                xticklabels=pipeline_final.named_steps['modelo'].classes_,
                yticklabels=pipeline_final.named_steps['modelo'].classes_,
                cmap='Purples')
    plt.title("Matriz de Confusão - Random Forest com Otimização e SMOTE + Undersampling")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()

    print("🔍 Amostra da base de teste com previsões:")
    print(teste_sem_rotulo_final[['City', 'Pollutant', 'Value', 'qualidade_ar_prevista']].head())

