import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def carregar_e_tratar_dados():
    df = pd.read_csv('data/world_air_quality.csv')
    
    # Suas transformações aqui
    df[['Latitude', 'Longitude']] = df['Coordinates'].str.strip().str.split(',', expand=True)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    
    df.drop(columns=['Source Name', 'Location'], inplace=True)

    # Preencher nulos com KNN
    df_validos = df.dropna(subset=['City', 'Latitude', 'Longitude']).copy()
    df_nulos = df[df['City'].isnull() & df['Latitude'].notnull() & df['Longitude'].notnull()].copy()

    le = LabelEncoder()
    df_validos['Country Code'] = le.fit_transform(df_validos['City'])
    X_train = df_validos[['Latitude', 'Longitude']]
    y_train = df_validos['Country Code']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    X_test = df_nulos[['Latitude', 'Longitude']]
    y_pred_codes = knn.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred_codes)
    df_nulos['City_Prevista'] = y_pred_labels
    df.loc[df_nulos.index, 'City'] = df_nulos['City_Prevista']

    # Remover negativos e duplicados
    df = df[df['Value'] >= 0]
    df = df.drop_duplicates()

    return df
