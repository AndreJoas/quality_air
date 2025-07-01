from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --- Carregar e tratar dados ---
def carregar_e_tratar_dados():
    df = pd.read_csv('data/world_air_quality.csv')

    # Extrair latitude e longitude
    df[['Latitude', 'Longitude']] = df['Coordinates'].str.strip().str.split(',', expand=True)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    # Limpeza inicial
    df.drop(columns=['Source Name', 'Location'], inplace=True)

    # Preencher cidades nulas via KNN
    df_validos = df.dropna(subset=['City', 'Latitude', 'Longitude']).copy()
    df_nulos = df[df['City'].isnull() & df['Latitude'].notnull() & df['Longitude'].notnull()].copy()

    le = LabelEncoder()
    df_validos['City_Code'] = le.fit_transform(df_validos['City'])
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df_validos[['Latitude', 'Longitude']], df_validos['City_Code'])

    if not df_nulos.empty:
        y_pred = knn.predict(df_nulos[['Latitude', 'Longitude']])
        df_nulos['City_Prevista'] = le.inverse_transform(y_pred)
        df.loc[df_nulos.index, 'City'] = df_nulos['City_Prevista']

    # Limpar dados inválidos
    df = df[df['Value'] >= 0].drop_duplicates()
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

    return df

df_global = carregar_e_tratar_dados()

# --- Rotas de frontend ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/teste_t_rota')
def test_t():
    return render_template('test_t.html')

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
    poluentes = sorted(df_global['Pollutant'].dropna().unique().tolist())
    return jsonify(poluentes)

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
        'media': df_filtrado['Value'].mean(),
        'mediana': df_filtrado['Value'].median(),
        'std': df_filtrado['Value'].std(),
        'max': df_filtrado['Value'].max(),
        'min': df_filtrado['Value'].min(),
        'count': len(df_filtrado)
    })

# --- Probabilidade de ultrapassar limite ---
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


# --- Teste T com melhorias ---
@app.route('/teste_t')
def teste_t():
    pais1 = request.args.get('pais1')
    pais2 = request.args.get('pais2')
    poluente = request.args.get('poluente')
    limite = request.args.get('limite', None, type=float)

    df1 = df_global[(df_global['Country Label'] == pais1) & (df_global['Pollutant'] == poluente)].copy()
    df2 = df_global[(df_global['Country Label'] == pais2) & (df_global['Pollutant'] == poluente)].copy()

    if limite is not None:
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
