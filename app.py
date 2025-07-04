# app.py

from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.linear_model import LinearRegression


import pycountry_convert as pc
import pycountry

app = Flask(__name__)


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

# Dados tratados disponíveis globalmente
df_global = carregar_e_tratar_dados()


# --- Rotas de frontend ---

@app.route('/dados_por_continente')
def dados_por_continente():
    df = df_global.copy()

    # Adiciona a coluna Continent se ainda não existir
    if 'Continent' not in df.columns:
        df['Continent'] = df['Country Label'].apply(obter_continente)

    continente = request.args.get('continente')
    if continente:
        df = df[df['Continent'] == continente]

    return jsonify(df.to_dict(orient='records'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/teste_t_rota')
def test_t():
    return render_template('test_t.html')

@app.route('/global')
def global2():
    return render_template('globa.html')

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
