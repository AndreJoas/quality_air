# app.py

from flask import Flask, jsonify, request, render_template, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
from flask import flash
import pycountry_convert as pc
import pycountry

app = Flask(__name__)
app.secret_key = 's3cr3t_k3y_2231nskldasSD#$!@$%D'  # Chave secreta para sessões Flask

def obter_continente(pais_nome):
    try:
        codigo_alpha2 = pycountry.countries.lookup(pais_nome).alpha_2
        continente_codigo = pc.country_alpha2_to_continent_code(codigo_alpha2)
        continentes = {
            'AF': 'África',
            'NA': 'América do Norte',
            'OC': 'Oceania',
            'AN': 'Antártica',
            'AS': 'Ásia',
            'EU': 'Europa',
            'SA': 'América do Sul'
        }
        return continentes.get(continente_codigo, 'Desconhecido')
    except:
        return 'Desconhecido'


# === Limites de outliers por unidade e poluente ===

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

def carregar_e_tratar_dados():
    print("\n📦 Carregando a base de dados...")
    df = pd.read_csv('data/world_air_quality.csv')

    tamanho_original = df.shape[0]
    print(f"🔢 Registros originais: {tamanho_original}")

    # Extrair latitude e longitude
    df[['Latitude', 'Longitude']] = df['Coordinates'].str.strip().str.split(',', expand=True)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    # Limpeza inicial
    df.drop(columns=['Source Name', 'Location'], inplace=True)

    # Preencher cidades nulas com KNN
    df_validos = df.dropna(subset=['City', 'Latitude', 'Longitude']).copy()
    df_nulos = df[df['City'].isnull() & df['Latitude'].notnull() & df['Longitude'].notnull()].copy()

    if not df_nulos.empty:
        print(f"🏙️ Preenchendo {len(df_nulos)} cidades ausentes com KNN...")
        le = LabelEncoder()
        df_validos['City_Code'] = le.fit_transform(df_validos['City'])
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(df_validos[['Latitude', 'Longitude']], df_validos['City_Code'])
        y_pred = knn.predict(df_nulos[['Latitude', 'Longitude']])
        df_nulos['City_Prevista'] = le.inverse_transform(y_pred)
        df.loc[df_nulos.index, 'City'] = df_nulos['City_Prevista']

    # Limpeza: valores inválidos e duplicados
    df = df[df['Value'] >= 0].drop_duplicates()
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

    print(f"✅ Base após preenchimento e limpeza inicial: {df.shape[0]} registros\n")

    # =========== REMOÇÃO DE OUTLIERS ============
    print("🚨 Removendo outliers por unidade e poluente...")
    combined_mask = pd.Series([False] * len(df), index=df.index)
    excluidos_por_unidade = {unidade: 0 for unidade in limites_por_unidade.keys()}

    for unidade, limites in limites_por_unidade.items():
        mask_unidade = pd.Series([False] * len(df), index=df.index)
        for poluente, limite in limites.items():
            mask = (
                (df['Pollutant'] == poluente) &
                (df['Value'] > limite) &
                (df['Unit'] == unidade)
            )
            mask_unidade |= mask
        combined_mask |= mask_unidade
        excluidos_por_unidade[unidade] = mask_unidade.sum()

    dados_excluidos = df.loc[combined_mask]
    base_filtrada = df.loc[~combined_mask]

    print(f"🗑️ Registros removidos no total: {len(dados_excluidos)}")
    for unidade, qtd in excluidos_por_unidade.items():
        print(f"   - Unidade '{unidade}': {qtd} removidos")

    print(f"\n📊 Base final após limpeza de outliers: {base_filtrada.shape[0]} registros")
    print("🔬 Poluentes presentes:", list(base_filtrada['Pollutant'].unique()))

    # === Verificação final ===
    print("\n🔎 Verificando valores ainda acima dos limites definidos...\n")
    total_acima = 0
    for unidade, limites in limites_por_unidade.items():
        for poluente, limite in limites.items():
            acima = base_filtrada[
                (base_filtrada['Unit'] == unidade) &
                (base_filtrada['Pollutant'] == poluente) &
                (base_filtrada['Value'] > limite)
            ]
            if not acima.empty:
                count = acima.shape[0]
                total_acima += count
                print(f"⚠️ {count} acima do limite: {poluente} ({unidade}) > {limite}")

    if total_acima == 0:
        print("✅ Nenhum valor acima dos limites foi encontrado. Limpeza eficaz!")
    else:
        print(f"🚨 Ainda há {total_acima} valores fora dos limites!")

    return base_filtrada



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

 

# Dados tratados disponíveis globalmente
df_global = carregar_e_tratar_dados()


# --- Rotas de frontend ---


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/avaliar_modelos', methods=['GET', 'POST'])
def avaliar_modelos():
    import optuna
    from sklearn.model_selection import cross_val_score
    from flask import flash, redirect, url_for

    unidade = request.args.get('unidade')

    if unidade not in ['µg/m³', 'ppm']:
        flash("Selecione uma unidade válida.", "warning")
        return redirect(url_for('avaliar_modelos'))

    # 👇 Recupera o número de trials do formulário (ou usa 7 como padrão)
    trials = request.args.get('trials')
    try:
        trials = int(trials)
    except (ValueError, TypeError):
        trials = 7  # valor padrão caso não seja informado ou inválido

    df = carregar_e_tratar_dados()
    df = df[df['Unit'] == unidade].copy()
    df['qualidade_ar'] = df.apply(lambda row: classificar_qualidade(row['Pollutant'], row['Value']), axis=1)
    df = df[df['qualidade_ar'] != 'Desconhecido']

    treino, teste = train_test_split(df, test_size=0.15, stratify=df['qualidade_ar'], random_state=42)
    base_balanceada = []
    for _, g in treino.groupby('qualidade_ar'):
        classe = g['qualidade_ar'].iloc[0]
        tamanho_original = len(g)
        n_desejado = 4500 if classe in ['Péssima', 'Má', 'Inadequada'] else 1000
        n_desejado = min(n_desejado, tamanho_original)
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

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test_real)

    def pipeline_with_model(model, y_encoded=True):
        y_base = y_train_encoded if y_encoded else y_train
        class_counts = pd.Series(y_base).value_counts()
        undersample_strategy = {cls: min(LIMITE_POR_CLASSE, count) for cls, count in class_counts.items() if count > LIMITE_POR_CLASSE}
        oversample_strategy = {cls: LIMITE_POR_CLASSE for cls, count in class_counts.items() if count < LIMITE_POR_CLASSE}

        return ImbPipeline([
            ('preprocessador', preprocessador),
            ('undersampler', RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)),
            ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors_value, sampling_strategy=oversample_strategy)),
            ('modelo', model)
        ])

    resultados = []

    # === XGBoost ===
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        pipe = pipeline_with_model(model)
        return cross_val_score(pipe, X_train, y_train_encoded, cv=3, scoring='f1_macro').mean()

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=trials, catch=(ValueError,))
    best_xgb = XGBClassifier(**study_xgb.best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    pipe_xgb = pipeline_with_model(best_xgb)
    pipe_xgb.fit(X_train, y_train_encoded)
    y_pred = label_encoder.inverse_transform(pipe_xgb.predict(X_test))

    resultados.append({
        'modelo': "XGBoost",
        'accuracy': round(accuracy_score(y_test_real, y_pred), 4),
        'precision': round(precision_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'recall': round(recall_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'f1': round(f1_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'matriz': confusion_matrix(y_test_real, y_pred).tolist(),
        'labels': sorted(y_test_real.unique().tolist())
    })

    # === Random Forest ===
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'class_weight': 'balanced',
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
        pipe = pipeline_with_model(model, y_encoded=False)
        return cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro').mean()

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=trials, catch=(ValueError,))
    best_rf = RandomForestClassifier(**study_rf.best_params, class_weight='balanced', random_state=42)
    pipe_rf = pipeline_with_model(best_rf, y_encoded=False)
    pipe_rf.fit(X_train, y_train)
    y_pred = pipe_rf.predict(X_test)

    resultados.append({
        'modelo': "Random Forest",
        'accuracy': round(accuracy_score(y_test_real, y_pred), 4),
        'precision': round(precision_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'recall': round(recall_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'f1': round(f1_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'matriz': confusion_matrix(y_test_real, y_pred).tolist(),
        'labels': sorted(y_test_real.unique().tolist())
    })

    # === KNN ===
    def objective_knn(trial):
        params = {'n_neighbors': trial.suggest_int('n_neighbors', 3, 15)}
        model = KNeighborsClassifier(**params)
        pipe = pipeline_with_model(model, y_encoded=False)
        return cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro').mean()

    study_knn = optuna.create_study(direction='maximize')
    study_knn.optimize(objective_knn, n_trials=trials)
    best_knn = KNeighborsClassifier(**study_knn.best_params)
    pipe_knn = pipeline_with_model(best_knn, y_encoded=False)
    pipe_knn.fit(X_train, y_train)
    y_pred = pipe_knn.predict(X_test)

    resultados.append({
        'modelo': "KNN",
        'accuracy': round(accuracy_score(y_test_real, y_pred), 4),
        'precision': round(precision_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'recall': round(recall_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'f1': round(f1_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'matriz': confusion_matrix(y_test_real, y_pred).tolist(),
        'labels': sorted(y_test_real.unique().tolist())
    })

    # === Logistic Regression ===
    def objective_lr(trial):
        params = {
            'C': trial.suggest_float('C', 0.01, 10.0),
            'class_weight': 'balanced',
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42
        }
        model = LogisticRegression(**params)
        pipe = pipeline_with_model(model, y_encoded=False)
        return cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro').mean()

    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(objective_lr, n_trials=trials, catch=(ValueError,))
    best_lr = LogisticRegression(**study_lr.best_params)
    best_lr.set_params(class_weight='balanced', max_iter=1000, random_state=42)
    pipe_lr = pipeline_with_model(best_lr, y_encoded=False)
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)

    resultados.append({
        'modelo': "Logistic Regression",
        'accuracy': round(accuracy_score(y_test_real, y_pred), 4),
        'precision': round(precision_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'recall': round(recall_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'f1': round(f1_score(y_test_real, y_pred, average='macro', zero_division=0), 4),
        'matriz': confusion_matrix(y_test_real, y_pred).tolist(),
        'labels': sorted(y_test_real.unique().tolist())
    })

    flash("✅ Avaliação dos modelos concluída com sucesso!", "success")
    ranking = sorted(resultados, key=lambda x: x['f1'], reverse=True)
    return render_template('avaliarmodelos.html', resultados=ranking, unidade=unidade)




@app.route('/teste_t_rota')
def test_t():
    return render_template('test_t.html')

@app.route('/global')
def global2():
    return render_template('globa.html')

@app.route('/modelos_classificacao')
def modelos_classificacao():
    return render_template('avaliarmodelos.html')


# --- Rotas de dados básicos ---
@app.route('/paises')
def paises():
    paises = sorted(df_global['Country Label'].dropna().unique().tolist())
    return jsonify(paises)

@app.route('/cidades')
def cidades():
    pais = request.args.get('pais')
    if pais:
        cidades = df_global[df_global['Country Label'] == pais]['City'].dropna().unique()
    else:
        cidades = df_global['City'].dropna().unique()
    return jsonify(sorted(cidades.tolist()))

@app.route('/poluentes')
def poluentes():
    pais = request.args.get('pais')
    cidade = request.args.get('cidade')

    df_filtrado = df_global.copy()

    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]
    if cidade:
        df_filtrado = df_filtrado[df_filtrado['City'] == cidade]

    poluentes = sorted(df_filtrado['Pollutant'].dropna().unique().tolist())
    return jsonify(poluentes)

@app.route('/unidades')
def unidades():
    pais = request.args.get('pais')
    cidade = request.args.get('cidade')
    poluente = request.args.get('poluente')

    df_filtrado = df_global.copy()

    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]
    if cidade:
        df_filtrado = df_filtrado[df_filtrado['City'] == cidade]
    if poluente:
        df_filtrado = df_filtrado[df_filtrado['Pollutant'] == poluente]

    unidades = sorted(df_filtrado['Unit'].dropna().unique().tolist())
    return jsonify(unidades)


# --- Dados filtrados com amostragem ---
@app.route('/dados')
def dados():
    pais = request.args.get('pais')
    cidade = request.args.get('cidade')
    poluente = request.args.get('poluente')
    unidade = request.args.get('unidade')
    amostragem = int(request.args.get('amostragem', 100))

    df_filtrado = df_global.copy()
    
    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]
    if cidade:
        df_filtrado = df_filtrado[df_filtrado['City'] == cidade]
    if poluente:
        df_filtrado = df_filtrado[df_filtrado['Pollutant'] == poluente]
    if unidade:
        df_filtrado = df_filtrado[df_filtrado['Unit'] == unidade]

    if amostragem < 100:
        df_filtrado = df_filtrado.groupby('Pollutant', group_keys=False).apply(
            lambda x: x.sample(frac=amostragem/100, random_state=42)
        )

    return jsonify(df_filtrado.to_dict(orient='records'))

# --- Estatísticas descritivas ---
@app.route('/analise_descritiva')
def analise_descritiva():
    pais = request.args.get('pais')
    poluente = request.args.get('poluente')

    df_filtrado = df_global.copy()
    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]
    if poluente:
        df_filtrado = df_filtrado[df_filtrado['Pollutant'] == poluente]

    return jsonify({
    'media': float(df_filtrado['Value'].mean()),
    'mediana': float(df_filtrado['Value'].median()),
    'std': float(df_filtrado['Value'].std()),
    'max': float(df_filtrado['Value'].max()),
    'min': float(df_filtrado['Value'].min()),
    'count': int(len(df_filtrado))
})


# --- Probabilidade de ultrapassar limite ---
@app.route('/probabilidade')
def probabilidade():
    poluente = request.args.get('poluente', None)
    limite = float(request.args.get('limite', '0'))
    pais = request.args.get('pais', None)
    cidade = request.args.get('cidade', None)
    unidade = request.args.get('unidade', None)

    df_filtrado = df_global.copy()

    if poluente:
        df_filtrado = df_filtrado[df_filtrado['Pollutant'] == poluente]
    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]
    if cidade:
        df_filtrado = df_filtrado[df_filtrado['City'] == cidade]
    if unidade:
        df_filtrado = df_filtrado[df_filtrado['Unit'] == unidade]

    total = len(df_filtrado)
    if total == 0:
        prob = 0
    else:
        acima = (df_filtrado['Value'] > limite).sum()
        prob = acima / total

    return jsonify({
    'probabilidade_ultrapassar': float(prob),
    'total_registros': int(total),
    'limite': float(limite),
    'poluente': poluente
})



@app.route('/teste_t')
def teste_t():
    pais1 = request.args.get('pais1')
    pais2 = request.args.get('pais2')
    poluente = request.args.get('poluente')
    limite = request.args.get('limite', None, type=float)

    df1 = df_global[(df_global['Country Label'] == pais1) & (df_global['Pollutant'] == poluente)].copy()
    df2 = df_global[(df_global['Country Label'] == pais2) & (df_global['Pollutant'] == poluente)].copy()

    if limite:
        df1 = df1[df1['Value'] <= limite]
        df2 = df2[df2['Value'] <= limite]

    if len(df1) < 2 or len(df2) < 2:
        return jsonify({'erro': 'Número insuficiente de amostras para teste t.'})

    t_stat, p_val = stats.ttest_ind(df1['Value'], df2['Value'], equal_var=False)

    return jsonify({
        'pais1': pais1,
        'pais2': pais2,
        'poluente': poluente,
        't_stat': t_stat,
        'p_valor': p_val,
        'n_amostras_pais1': len(df1),
        'n_amostras_pais2': len(df2),
        'media_pais1': df1['Value'].mean(),
        'media_pais2': df2['Value'].mean()
    })

@app.route('/grafico_t')
def grafico_t():
    pais1 = request.args.get('pais1')
    pais2 = request.args.get('pais2')
    poluente = request.args.get('poluente')

    df1 = df_global[(df_global['Country Label'] == pais1) & (df_global['Pollutant'] == poluente)]
    df2 = df_global[(df_global['Country Label'] == pais2) & (df_global['Pollutant'] == poluente)]

    amostra1 = df1['Value'].dropna()
    amostra2 = df2['Value'].dropna()

    media1, std1 = np.mean(amostra1), np.std(amostra1, ddof=1)
    media2, std2 = np.mean(amostra2), np.std(amostra2, ddof=1)

    plt.figure(figsize=(10, 5))
    bins = 15
    plt.hist(amostra1, bins=bins, alpha=0.5, color='blue', density=True, label=pais1)
    plt.hist(amostra2, bins=bins, alpha=0.5, color='orange', density=True, label=pais2)

    intervalo = np.linspace(min(amostra1.min(), amostra2.min()), max(amostra1.max(), amostra2.max()), 300)
    plt.plot(intervalo, norm.pdf(intervalo, media1, std1), color='blue', linestyle='--')
    plt.plot(intervalo, norm.pdf(intervalo, media2, std2), color='orange', linestyle='--')
    plt.axvline(media1, color='blue', linestyle=':', label=f'Média {pais1}: {media1:.2f}')
    plt.axvline(media2, color='orange', linestyle=':', label=f'Média {pais2}: {media2:.2f}')

    plt.title(f'Distribuição dos Valores - {poluente}')
    plt.xlabel('Valor')
    plt.ylabel('Densidade')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/grafico_z')
def grafico_z():
    pais1 = request.args.get('pais1')
    pais2 = request.args.get('pais2')
    poluente = request.args.get('poluente')

    df1 = df_global[(df_global['Country Label'] == pais1) & (df_global['Pollutant'] == poluente)]
    df2 = df_global[(df_global['Country Label'] == pais2) & (df_global['Pollutant'] == poluente)]

    amostra1 = df1['Value'].dropna()
    amostra2 = df2['Value'].dropna()

    if len(amostra1) < 2 or len(amostra2) < 2:
        return "Amostras insuficientes", 400

    # Teste t de Welch
    stat, p_val = stats.ttest_ind(amostra1, amostra2, equal_var=False)
    z_valor = stat  # aproximação da estatística t como z

    plt.figure(figsize=(8, 5))

    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x)

    plt.plot(x, y, label='Distribuição Normal Padrão', color='gray')

    # Região crítica
    alpha = 0.05
    z_crit = stats.norm.ppf(1 - alpha / 2)

    plt.fill_between(x, y, where=(x <= -z_crit) | (x >= z_crit), color='red', alpha=0.3,
                     label=f'Região crítica (|z| > {z_crit:.2f})')

    plt.axvline(z_valor, color='red', linestyle='--', label=f'Estatística Z ≈ {z_valor:.2f}')
    plt.axvline(-z_crit, color='black', linestyle=':', label=f'Z crítico: ±{z_crit:.2f}')
    plt.axvline(z_crit, color='black', linestyle=':')

    plt.title('Distribuição Z e Região Crítica')
    plt.xlabel('Z')
    plt.ylabel('Densidade')
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


# --- Correlação entre poluentes ---
@app.route('/correlacao')
def correlacao():
    pais = request.args.get('pais')
    df_filtrado = df_global[df_global['Country Label'] == pais] if pais else df_global

    tabela = df_filtrado.pivot_table(index=['City', 'Last Updated'], columns='Pollutant', values='Value')
    tabela = tabela.dropna()

    return jsonify(tabela.corr(method='pearson').to_dict())

# --- Regressão linear simples ---
@app.route('/regressao')
def regressao():
    poluente_x = request.args.get('poluente_x')
    poluente_y = request.args.get('poluente_y')
    pais = request.args.get('pais')

    df_filtrado = df_global[df_global['Country Label'] == pais] if pais else df_global

    tabela = df_filtrado.pivot_table(index=['City', 'Last Updated'], columns='Pollutant', values='Value').dropna()

    X = tabela[poluente_x].values.reshape(-1, 1)
    y = tabela[poluente_y].values

    model = LinearRegression()
    model.fit(X, y)

    return jsonify({
        'coeficiente': model.coef_[0],
        'intercepto': model.intercept_,
        'r2_score': model.score(X, y)
    })

# --- Inicialização ---
if __name__ == '__main__':
    app.run(debug=True)
