from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.linear_model import LinearRegression
import datetime

app = Flask(__name__)

# --- Carregar e tratar dados (com imputação KNN para cidades) ---
def carregar_e_tratar_dados():
    df = pd.read_csv('data/world_air_quality.csv')

    df[['Latitude', 'Longitude']] = df['Coordinates'].str.strip().str.split(',', expand=True)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    df.drop(columns=['Source Name', 'Location'], inplace=True)

    # Imputar cidades faltantes com KNN baseado em lat/lon
    df_validos = df.dropna(subset=['City', 'Latitude', 'Longitude']).copy()
    df_nulos = df[df['City'].isnull() & df['Latitude'].notnull() & df['Longitude'].notnull()].copy()

    le = LabelEncoder()
    df_validos['Country Code'] = le.fit_transform(df_validos['City'])
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df_validos[['Latitude','Longitude']], df_validos['Country Code'])
    if not df_nulos.empty:
        y_pred = knn.predict(df_nulos[['Latitude','Longitude']])
        df_nulos['City_Prevista'] = le.inverse_transform(y_pred)
        df.loc[df_nulos.index, 'City'] = df_nulos['City_Prevista']

    # Remove negativos e duplicados
    df = df[df['Value'] >= 0]
    df = df.drop_duplicates()

    # Converte data
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

    return df

df_global = carregar_e_tratar_dados()

# --- Rota index (front-end) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Rota para pegar países únicos ---
@app.route('/paises')
def paises():
    paises = sorted(df_global['Country Label'].dropna().unique().tolist())
    return jsonify(paises)

# --- Rota para pegar cidades por país ---
@app.route('/cidades')
def cidades():
    pais = request.args.get('pais', None)
    if pais:
        cidades = sorted(df_global[df_global['Country Label'] == pais]['City'].dropna().unique().tolist())
    else:
        cidades = sorted(df_global['City'].dropna().unique().tolist())
    return jsonify(cidades)

# --- Rota para pegar poluentes únicos ---
@app.route('/poluentes')
def poluentes():
    poluentes = sorted(df_global['Pollutant'].dropna().unique().tolist())
    return jsonify(poluentes)

# --- Rota para pegar dados filtrados com amostragem ---
@app.route('/dados')
def dados():
    pais = request.args.get('pais', None)
    cidade = request.args.get('cidade', None)
    poluente = request.args.get('poluente', None)
    unidade = request.args.get('unidade', None)
    amostragem = request.args.get('amostragem', '100')  # porcentagem 0-100

    df_filtrado = df_global.copy()

    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]
    if cidade:
        df_filtrado = df_filtrado[df_filtrado['City'] == cidade]
    if poluente:
        df_filtrado = df_filtrado[df_filtrado['Pollutant'] == poluente]
    if unidade:
        df_filtrado = df_filtrado[df_filtrado['Unit'] == unidade]

    # Amostragem estratificada por poluente (se amostragem < 100)
    if int(amostragem) < 100:
        df_filtrado = df_filtrado.groupby('Pollutant', group_keys=False).apply(
            lambda x: x.sample(frac=int(amostragem)/100, random_state=42)
        )

    # Retorna dados no formato JSON para front
    return jsonify(df_filtrado.to_dict(orient='records'))



# --- Rota análise descritiva ---
@app.route('/analise_descritiva')
def analise_descritiva():
    pais = request.args.get('pais', None)
    poluente = request.args.get('poluente', None)

    df_filtrado = df_global.copy()
    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]
    if poluente:
        df_filtrado = df_filtrado[df_filtrado['Pollutant'] == poluente]

    # Estatísticas básicas
    media = df_filtrado['Value'].mean()
    mediana = df_filtrado['Value'].median()
    std = df_filtrado['Value'].std()
    max_val = df_filtrado['Value'].max()
    min_val = df_filtrado['Value'].min()

    return jsonify({
        'media': media,
        'mediana': mediana,
        'std': std,
        'max': max_val,
        'min': min_val,
        'count': len(df_filtrado)
    })

# --- Rota aplicação probabilidade: probabilidade de ultrapassar limite ---
@app.route('/probabilidade')
def probabilidade():
    poluente = request.args.get('poluente', None)
    limite = float(request.args.get('limite', '0'))

    df_filtrado = df_global
    if poluente:
        df_filtrado = df_filtrado[df_filtrado['Pollutant'] == poluente]

    total = len(df_filtrado)
    if total == 0:
        prob = 0
    else:
        acima = (df_filtrado['Value'] > limite).sum()
        prob = acima / total

    return jsonify({
        'probabilidade_ultrapassar': prob,
        'total_registros': total,
        'limite': limite,
        'poluente': poluente
    })


@app.route('/teste_t_rota')
def test_t():
    return render_template('test_t.html')

# --- Rota inferência estatística: teste t entre dois países ---
@app.route('/teste_t')
def teste_t():
    pais1 = request.args.get('pais1', None)
    pais2 = request.args.get('pais2', None)
    poluente = request.args.get('poluente', None)

    df1 = df_global[(df_global['Country Label'] == pais1) & (df_global['Pollutant'] == poluente)]
    df2 = df_global[(df_global['Country Label'] == pais2) & (df_global['Pollutant'] == poluente)]

    if len(df1) < 2 or len(df2) < 2:
        return jsonify({'erro': 'Número insuficiente de amostras para teste t.'})

    t_stat, p_val = stats.ttest_ind(df1['Value'], df2['Value'], equal_var=False)

    return jsonify({
        'pais1': pais1,
        'pais2': pais2,
        'poluente': poluente,
        't_stat': t_stat,
        'p_valor': p_val
    })

# --- Rota correlação Pearson ---
@app.route('/correlacao')
def correlacao():
    pais = request.args.get('pais', None)
    df_filtrado = df_global.copy()
    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]

    # Pivot para matriz poluentes por registros
    tabela = df_filtrado.pivot_table(index=['City','Last Updated'], columns='Pollutant', values='Value')

    # Drop linhas com NA
    tabela = tabela.dropna(axis=0, how='any')

    # Calcular correlação Pearson
    corr = tabela.corr(method='pearson')

    return jsonify(corr.to_dict())

# --- Rota regressão linear (simplificada) ---
@app.route('/regressao')
def regressao():
    poluente_x = request.args.get('poluente_x', None)
    poluente_y = request.args.get('poluente_y', None)
    pais = request.args.get('pais', None)

    df_filtrado = df_global.copy()
    if pais:
        df_filtrado = df_filtrado[df_filtrado['Country Label'] == pais]

    tabela = df_filtrado.pivot_table(index=['City','Last Updated'], columns='Pollutant', values='Value').dropna()
    X = tabela[poluente_x].values.reshape(-1,1)
    y = tabela[poluente_y].values

    model = LinearRegression()
    model.fit(X,y)
    coef = model.coef_[0]
    intercept = model.intercept_
    score = model.score(X,y)

    return jsonify({
        'coeficiente': coef,
        'intercepto': intercept,
        'r2_score': score
    })

if __name__ == '__main__':
    app.run(debug=True)
