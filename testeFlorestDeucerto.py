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
    study.optimize(objective, n_trials=2)

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






    # print("🔍 Amostra da base de teste com previsões:")
    # print(teste_sem_rotulo_final[['City', 'Pollutant', 'Value', 'qualidade_ar_prevista']].head())

    # ===================== FUNÇÕES PARA TESTES ESTATÍSTICOS =====================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

def teste_normalidade(dados):
    n = len(dados)
    if n < 30:
        stat, p = stats.shapiro(dados)
        teste = "Shapiro-Wilk"
        justificativa = "n < 30"
    elif 30 <= n <= 100:
        stat, p = stats.shapiro(dados)
        teste = "Shapiro-Wilk (preferido)"
        justificativa = "30 ≤ n ≤ 100"
    else:
        stat, p = stats.normaltest(dados)
        teste = "D'Agostino-Pearson"
        justificativa = "n > 100"
    return teste, p, justificativa

def escolher_teste_e_calcular(amostra1, amostra2, alpha=0.05):
    # Normalidade
    teste1, p1, just1 = teste_normalidade(amostra1)
    teste2, p2, just2 = teste_normalidade(amostra2)

    normal1 = p1 >= alpha
    normal2 = p2 >= alpha

    print("\n### Teste de Normalidade")
    print(f"Amostra 1 - Teste: {teste1} ({just1}), p-valor = {p1:.4f} → {'Normal' if normal1 else 'Não normal'}")
    print(f"Amostra 2 - Teste: {teste2} ({just2}), p-valor = {p2:.4f} → {'Normal' if normal2 else 'Não normal'}")

    media1, std1 = np.mean(amostra1), np.std(amostra1, ddof=1)
    media2, std2 = np.mean(amostra2), np.std(amostra2, ddof=1)

    z_valor = None
    if normal1 and normal2:
        stat_lev, p_lev = stats.levene(amostra1, amostra2)
        variancias_iguais = p_lev >= alpha
        if variancias_iguais:
            teste_usado = "t-Student (variâncias iguais)"
            justificativa = "As duas amostras são normais e Levene indicou variâncias iguais"
            stat, p = stats.ttest_ind(amostra1, amostra2, equal_var=True)
        else:
            teste_usado = "t-Welch (variâncias diferentes)"
            justificativa = "As duas amostras são normais e Levene indicou variâncias diferentes"
            stat, p = stats.ttest_ind(amostra1, amostra2, equal_var=False)
        z_valor = stat
    else:
        teste_usado = "Mann-Whitney (não paramétrico)"
        justificativa = "Pelo menos uma das amostras não segue distribuição normal"
        stat, p = stats.mannwhitneyu(amostra1, amostra2, alternative='two-sided')

    decisao = "Rejeita H₀" if p < alpha else "Não rejeita H₀"
    ha_diferenca = "Há diferença significativa entre os grupos." if p < alpha else "Não há diferença significativa entre os grupos."

    print("\n### Teste de Hipóteses")
    print(f"Teste escolhido: {teste_usado}")
    print(f"Justificativa: {justificativa}")
    print("\nHipóteses:")
    print("H₀: As médias dos dois grupos são iguais")
    print("H₁: As médias dos dois grupos são diferentes")
    print(f"\nEstatística de teste: {stat:.4f}")
    if z_valor is not None:
        print(f"Valor Z aproximado: {z_valor:.4f}")
    print(f"p-valor: {p:.4f}")
    print(f"Decisão: {decisao}")
    print(f"{ha_diferenca}")
    print(f"\nMédia amostra 1: {media1:.2f}")
    print(f"Média amostra 2: {media2:.2f}")

    # GRÁFICOS
    plt.figure(figsize=(12, 5))

    # Subplot 1: Histogramas
    plt.subplot(1, 2, 1)
    bins = 15
    plt.hist(amostra1, bins=bins, alpha=0.5, color='blue', density=True, label='Amostra 1')
    plt.hist(amostra2, bins=bins, alpha=0.5, color='orange', density=True, label='Amostra 2')

    intervalo = np.linspace(
        np.min(np.concatenate([amostra1, amostra2])) - 5,
        np.max(np.concatenate([amostra1, amostra2])) + 5,
        200
    )
    y1 = stats.norm.pdf(intervalo, media1, std1)
    y2 = stats.norm.pdf(intervalo, media2, std2)
    plt.plot(intervalo, y1, color='blue', linestyle='--')
    plt.plot(intervalo, y2, color='orange', linestyle='--')
    plt.axvline(media1, color='blue', linestyle=':', label=f'Média 1: {media1:.2f}')
    plt.axvline(media2, color='orange', linestyle=':', label=f'Média 2: {media2:.2f}')
    plt.title('Distribuições das Amostras')
    plt.xlabel('Valor')
    plt.ylabel('Densidade')
    plt.legend()

    # Subplot 2: Distribuição Z
    plt.subplot(1, 2, 2)
    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x, 0, 1)
    plt.plot(x, y, label='N(0,1)', color='gray')
    if z_valor is not None:
        plt.fill_between(x, y, where=(x <= -abs(z_valor)) | (x >= abs(z_valor)),
                         color='red', alpha=0.3, label=f'Zona p-valor (|Z| ≥ {abs(z_valor):.2f})')
        plt.axvline(z_valor, color='red', linestyle='--', label=f'Z calc: {z_valor:.2f}')
    z_crit = stats.norm.ppf(1 - alpha/2)
    plt.axvline(-z_crit, color='black', linestyle=':', label=f'Z crítico: ±{z_crit:.2f}')
    plt.axvline(z_crit, color='black', linestyle=':')
    plt.title('Distribuição Normal Padrão (Z)')
    plt.xlabel('Z')
    plt.ylabel('Densidade')
    plt.legend()

    plt.tight_layout()
    plt.show()





# ===================== ENTRADA INTERATIVA POR PAÍS =====================

print("\n🧪 Teste Estatístico Interativo entre Países")

# Mostrar poluentes disponíveis
poluentes_disponiveis = base_balanceada['Pollutant'].value_counts().index.tolist()
print("\n📋 Poluentes disponíveis:")
for i, pol in enumerate(poluentes_disponiveis, 1):
    print(f"{i}. {pol}")
poluente_escolhido = input("\nDigite o nome do poluente para análise (ex: PM10): ").strip()

# Mostrar países disponíveis com esse poluente
paises_disponiveis = base_balanceada[base_balanceada['Pollutant'] == poluente_escolhido]['Country Label'].value_counts().index.tolist()

if len(paises_disponiveis) < 2:
    print("❌ Poluente não encontrado ou com menos de dois países disponíveis.")
else:
    print("\n🌍 Países disponíveis para esse poluente:")
    for i, pais in enumerate(paises_disponiveis[:10], 1):
        print(f"{i}. {pais}")

    pais1 = input("\nDigite o nome do primeiro país: ").strip()
    pais2 = input("Digite o nome do segundo país: ").strip()

    amostra1 = base_balanceada[
        (base_balanceada['Pollutant'] == poluente_escolhido) &
        (base_balanceada['Country Label'] == pais1)
    ]['Value'].dropna()

    amostra2 = base_balanceada[
        (base_balanceada['Pollutant'] == poluente_escolhido) &
        (base_balanceada['Country Label'] == pais2)
    ]['Value'].dropna()

    if len(amostra1) >= 8 and len(amostra2) >= 8:
        print(f"\n🔬 Iniciando teste entre '{pais1}' e '{pais2}' para o poluente '{poluente_escolhido}'")
        escolher_teste_e_calcular(amostra1.values, amostra2.values)
    else:
        print(f"❌ Amostras insuficientes: {len(amostra1)} e {len(amostra2)} registros encontrados.")



# # ===================== ENTRADA INTERATIVA cidade =====================

# print("\n🧪 Teste Estatístico Interativo")

# # Mostrar poluentes disponíveis
# poluentes_disponiveis = base_balanceada['Pollutant'].value_counts().index.tolist()
# print("\n📋 Poluentes disponíveis:")
# for i, pol in enumerate(poluentes_disponiveis, 1):
#     print(f"{i}. {pol}")
# poluente_escolhido = input("\nDigite o nome do poluente para análise (ex: PM10): ").strip()

# # Mostrar cidades com esse poluente
# cidades_disponiveis = base_balanceada[base_balanceada['Pollutant'] == poluente_escolhido]['City'].value_counts().index.tolist()

# if len(cidades_disponiveis) < 2:
#     print("❌ Poluente não encontrado ou com menos de duas cidades disponíveis.")
# else:
#     print("\n🌆 Cidades disponíveis para esse poluente:")
#     for i, cidade in enumerate(cidades_disponiveis[:10], 1):
#         print(f"{i}. {cidade}")

#     cidade1 = input("\nDigite o nome da primeira cidade: ").strip()
#     cidade2 = input("Digite o nome da segunda cidade: ").strip()

#     amostra1 = base_balanceada[
#         (base_balanceada['Pollutant'] == poluente_escolhido) &
#         (base_balanceada['City'] == cidade1)
#     ]['Value'].dropna()

#     amostra2 = base_balanceada[
#         (base_balanceada['Pollutant'] == poluente_escolhido) &
#         (base_balanceada['City'] == cidade2)
#     ]['Value'].dropna()

#     if len(amostra1) >= 8 and len(amostra2) >= 8:
#         print(f"\n🔬 Iniciando teste entre '{cidade1}' e '{cidade2}' para o poluente '{poluente_escolhido}'")
#         escolher_teste_e_calcular(amostra1.values, amostra2.values)
#     else:
#         print(f"❌ Amostras insuficientes: {len(amostra1)} e {len(amostra2)} registros encontrados.")


