from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# URL del dataset
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_ecommerce.csv"

def load_data(url):
    df = pd.read_csv(url)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    return df

df_ecommerce = load_data(url)

def analyze_fraud_transactions(df):
    results = {}
    if 'Monto' in df.columns and 'Fecha_Hora' in df.columns and 'Es_Fraudulenta' in df.columns:
        df['Hora'] = df['Fecha_Hora'].dt.hour
        df['Dia_Semana'] = df['Fecha_Hora'].dt.dayofweek
        df_encoded_transacciones = pd.get_dummies(df, columns=['Producto'], prefix='Prod', dummy_na=False)
        features_transacciones = ['Monto', 'Hora', 'Dia_Semana'] + [col for col in df_encoded_transacciones.columns if col.startswith('Prod_')]
        features_transacciones = [col for col in features_transacciones if col in df_encoded_transacciones.columns]

        if 'Es_Fraudulenta' in df_encoded_transacciones.columns and all(feature in df_encoded_transacciones.columns for feature in features_transacciones):
            X_trans = df_encoded_transacciones[features_transacciones]
            y_trans = df_encoded_transacciones['Es_Fraudulenta']
            X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.3, random_state=42, stratify=y_trans)

            scaler_trans = StandardScaler()
            X_train_scaled_trans = scaler_trans.fit_transform(X_train_trans)
            X_test_scaled_trans = scaler_trans.transform(X_test_trans)

            model_ecommerce = LogisticRegression(random_state=42)
            model_ecommerce.fit(X_train_scaled_trans, y_train_trans)
            y_pred_ecommerce = model_ecommerce.predict(X_test_scaled_trans)

            results['accuracy'] = accuracy_score(y_test_trans, y_pred_ecommerce)
            results['classification_report'] = classification_report(y_test_trans, y_pred_ecommerce, target_names=['No Fraudulenta', 'Fraudulenta'], zero_division=0)

            if X_test_scaled_trans.shape[1] >= 2:
                fig_trans, ax_trans = plt.subplots()
                scatter_trans = ax_trans.scatter(X_test_scaled_trans[:, 0], X_test_scaled_trans[:, 1], c=y_pred_ecommerce, cmap='coolwarm', alpha=0.7)
                ax_trans.set_xlabel("Feature 1 (Scaled)")
                ax_trans.set_ylabel("Feature 2 (Scaled)")
                ax_trans.set_title("Predicciones de Fraude (Regresión Logística)")
                legend_trans = ax_trans.legend(*scatter_trans.legend_elements(), title="Clase Predicha")
                ax_trans.add_artist(legend_trans)
                buf = io.BytesIO()
                fig_trans.savefig(buf, format='png')
                buf.seek(0)
                results['fraud_prediction_plot'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig_trans)
            else:
                results['fraud_prediction_plot'] = None
                results['fraud_prediction_plot_warning'] = "No se pueden mostrar gráficos de dispersión con menos de 2 características."
        else:
            results['fraud_analysis_warning'] = "No se pueden realizar análisis de transacciones fraudulentas debido a la falta de columnas necesarias."
    else:
        results['fraud_analysis_warning'] = "No se pueden realizar análisis de transacciones fraudulentas debido a la falta de columnas necesarias ('Monto', 'Fecha_Hora', 'Es_Fraudulenta')."
    return results

def detect_fake_accounts(df, eps_ip=0.5, min_samples_ip=2):
    results = {}
    if 'Direccion_IP' in df.columns:
        le_ip = LabelEncoder()
        df['IP_Codificada'] = le_ip.fit_transform(df['Direccion_IP'])
        ip_array = df[['IP_Codificada']].values
        scaler_ip = StandardScaler()
        ip_scaled = scaler_ip.fit_transform(ip_array)

        dbscan_ip = DBSCAN(eps=eps_ip, min_samples=min_samples_ip)
        df['Grupo_IP'] = dbscan_ip.fit_predict(ip_scaled)

        results['ip_clusters_head'] = df[['ID_Usuario', 'Direccion_IP', 'Grupo_IP']].head().to_html(index=False)
        results['ip_clusters_unique'] = np.unique(df['Grupo_IP']).tolist()
        results['fraud_by_ip_group'] = df.groupby('Grupo_IP')['Es_Fraudulenta'].sum().sort_values(ascending=False).to_html()

        fig_ip, ax_ip = plt.subplots()
        scatter_ip = ax_ip.scatter(ip_scaled[:, 0], np.zeros(len(ip_scaled)), c=df['Grupo_IP'], cmap='viridis')
        ax_ip.set_xlabel("IP Codificada (Escalada)")
        ax_ip.set_yticks([])
        ax_ip.set_title("Clusters de Direcciones IP (DBSCAN)")
        legend_ip = ax_ip.legend(*scatter_ip.legend_elements(), title="Grupo IP")
        ax_ip.add_artist(legend_ip)
        buf = io.BytesIO()
        fig_ip.savefig(buf, format='png')
        buf.seek(0)
        results['ip_clusters_plot'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_ip)
    else:
        results['fake_accounts_warning'] = "No se puede realizar el análisis de detección de cuentas falsas porque falta la columna 'Direccion_IP'."
    return results

def analyze_user_behavior(df, frequency_threshold=3):
    results = {}
    if 'ID_Usuario' in df.columns and 'Fecha_Hora' in df.columns:
        frecuencia_usuarios = df.groupby('ID_Usuario')['Fecha_Hora'].count().reset_index(name='Num_Transacciones')
        results['user_frequency_head'] = frecuencia_usuarios.head().to_html(index=False)

        usuarios_alta_actividad = frecuencia_usuarios[frecuencia_usuarios['Num_Transacciones'] > frequency_threshold]['ID_Usuario'].tolist()
        results['high_activity_users'] = usuarios_alta_actividad[:10]
        if usuarios_alta_actividad:
            transacciones_alta_actividad = df[df['ID_Usuario'].isin(usuarios_alta_actividad)]
            results['high_activity_transactions_head'] = transacciones_alta_actividad[['ID_Usuario', 'Fecha_Hora', 'Monto', 'Es_Fraudulenta']].head().to_html(index=False)
        else:
            results['high_activity_message'] = "No se encontraron usuarios con una alta frecuencia de transacciones según el umbral seleccionado."

        fig_freq, ax_freq = plt.subplots()
        ax_freq.hist(frecuencia_usuarios['Num_Transacciones'], bins=20, edgecolor='black')
        ax_freq.axvline(frequency_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Umbral: {frequency_threshold}')
        ax_freq.set_xlabel('Número de Transacciones por Usuario')
        ax_freq.set_ylabel('Número de Usuarios')
        ax_freq.set_title('Distribución de Frecuencia de Transacciones')
        ax_freq.legend()
        buf = io.BytesIO()
        fig_freq.savefig(buf, format='png')
        buf.seek(0)
        results['frequency_distribution_plot'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_freq)
    else:
        results['user_behavior_warning'] = "No se puede realizar el análisis de comportamiento del usuario porque faltan las columnas 'ID_Usuario' o 'Fecha_Hora'."
    return results

@app.route("/", methods=['GET', 'POST'])
def index():
    fraud_results = {}
    fake_accounts_results = {}
    user_behavior_results = {}
    show_fraud = False
    show_fake = False
    show_behavior = False
    eps_ip = 0.5
    min_samples_ip = 2
    frequency_threshold = 3

    if request.method == 'POST':
        show_fraud = 'show_fraud' in request.form
        show_fake = 'show_fake' in request.form
        show_behavior = 'show_behavior' in request.form
        try:
            eps_ip = float(request.form['eps_ip'])
            min_samples_ip = int(request.form['min_samples_ip'])
            frequency_threshold = int(request.form['frequency_threshold'])
        except ValueError:
            pass # Use default values if input is invalid

        if show_fraud:
            fraud_results = analyze_fraud_transactions(df_ecommerce.copy())
        if show_fake:
            fake_accounts_results = detect_fake_accounts(df_ecommerce.copy(), eps_ip, min_samples_ip)
        if show_behavior:
            user_behavior_results = analyze_user_behavior(df_ecommerce.copy(), frequency_threshold)

    return render_template('index.html',
                           head=df_ecommerce.head().to_html(index=False),
                           show_fraud=show_fraud,
                           fraud_results=fraud_results,
                           show_fake=show_fake,
                           fake_accounts_results=fake_accounts_results,
                           eps_ip=eps_ip,
                           min_samples_ip=min_samples_ip,
                           show_behavior=show_behavior,
                           user_behavior_results=user_behavior_results,
                           frequency_threshold=frequency_threshold)

if __name__ == '__main__':
    app.run(debug=True)****
