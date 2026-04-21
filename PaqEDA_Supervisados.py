# -*- coding: utf-8 -*-
"""
Paquete EDA + Supervisados
Estudio de Caso #2 — BCD-6210 Mineria de Datos
Universidad LEAD | I Cuatrimestre 2026

Integrantes:
  - Maria Jose Miranda
  - Julio Orozco
  - Siloe Cristina Campos Viquez

Guiado por:
  - PaqEDA.py                (Dr. Juan Murillo-Morera)
  - GuiaClaseSupervisada.py  (Dr. Juan Murillo-Morera)
"""

# ─────────────────────────────────────────────────────────────
#  LIBRERIAS
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════
#  CLASE EDA
# ═════════════════════════════════════════════════════════════
class analisisEDA():
    """
    Clase para el Analisis Exploratorio de Datos.
    num=1  →  sep=',' con index_col=0
    num=2  →  sep=';'
    """

    def __init__(self, path, num):
        self.__df = self.__datosCargados(path, num)

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, p_df):
        self.__df = p_df

    # ── Carga ──────────────────────────────────────────────────
    def __datosCargados(self, path, num):
        if num == 1:
            return pd.read_csv(path, sep=",", decimal=".", index_col=0)
        if num == 2:
            return pd.read_csv(path, sep=";", decimal=".")

    # ── Inspeccion basica ──────────────────────────────────────
    def tipoDatos(self):
        print(self.__df.dtypes)

    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include=["number"])

    def analisisCompleto(self):
        cols_cat = self.__df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        print(f"Columnas categoricas convertidas a dummies: {cols_cat}")
        self.__df = pd.get_dummies(
            self.__df, columns=cols_cat, drop_first=True).astype(int)

    def eliminarColumnas(self, columnas):
        self.__df.drop(columns=columnas, inplace=True)

    def renombrarColumnas(self, nuevos_nombres):
        self.__df.rename(columns=nuevos_nombres, inplace=True)

    def valores_unicos(self, v):
        unique_values = self.__df[v].unique()
        print(f"Valores unicos en '{v}':")
        for value in unique_values:
            count = (self.__df[v] == value).sum()
            print(f"  {value}: {count}")

    def valores_faltantes(self):
        missing = self.__df.isna().sum()
        print("Valores faltantes por columna:")
        print(missing)
        print()

    def eliminarDuplicados(self):
        antes = self.__df.shape[0]
        self.__df.drop_duplicates(inplace=True)
        despues = self.__df.shape[0]
        print(f"Filas duplicadas eliminadas: {antes - despues}. "
              f"Total actual: {despues}.")

    def eliminarNulos(self):
        nulos_antes = self.__df.isnull().sum().sum()
        filas_antes = self.__df.shape[0]
        print(f"Valores nulos antes: {nulos_antes}")
        self.__df.dropna(inplace=True)
        filas_despues = self.__df.shape[0]
        print(f"Filas eliminadas: {filas_antes - filas_despues}")
        print(f"Valores nulos restantes: {self.__df.isnull().sum().sum()}")

    # ── Estadisticas ───────────────────────────────────────────
    def analisis(self):
        print("Dimensiones:", self.__df.shape)
        print(self.__df.head())
        print("=" * 40)
        print("Estadisticas Descriptivas Generales")
        print("=" * 40)
        print("Media:\n",               self.__df.mean(numeric_only=True))
        print("=" * 40)
        print("Mediana:\n",             self.__df.median(numeric_only=True))
        print("=" * 40)
        print("Desviacion estandar:\n", self.__df.std(numeric_only=True))
        print("=" * 40)
        print("Maximos:\n",             self.__df.max(numeric_only=True))
        print("=" * 40)
        print("Minimos:\n",             self.__df.min(numeric_only=True))
        print("=" * 40)
        print("Cuantiles:\n",           self.__df.quantile(
            [0, 0.25, 0.5, 0.75, 1], numeric_only=True))

    def correlaciones(self):
        corr = self.__df.corr(numeric_only=True)
        print("Matriz de correlaciones:\n")
        print(corr)

    # ── Graficos ───────────────────────────────────────────────
    def graficoBoxplot(self):
        cols_num = self.__df.select_dtypes(include='number').columns
        n = len(cols_num)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas,
                                 figsize=(5 * columnas, 4 * filas), dpi=130)
        axes = axes.flatten()
        colores = sns.color_palette("Set3", n)
        for i, col in enumerate(cols_num):
            sns.boxplot(y=self.__df[col], ax=axes[i], color=colores[i])
            axes[i].set_title(f"Boxplot de {col}", fontsize=9)
            axes[i].set_ylabel(col)
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def histogramas(self):
        cols_num = self.__df.select_dtypes(include='number').columns
        n = len(cols_num)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas,
                                 figsize=(5 * columnas, 4 * filas), dpi=130)
        axes = axes.flatten()
        colores = sns.color_palette("Set2", n)
        for i, col in enumerate(cols_num):
            axes[i].hist(self.__df[col], bins=30, color=colores[i],
                         edgecolor='black', alpha=0.7)
            axes[i].set_title(f"Histograma de {col}", fontsize=9)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def distribucionVariables(self):
        cols_num = self.__df.select_dtypes(include='number').columns
        n = len(cols_num)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas,
                                 figsize=(5 * columnas, 4 * filas), dpi=130)
        axes = axes.flatten()
        colores = sns.color_palette("coolwarm", n)
        for i, col in enumerate(cols_num):
            sns.histplot(self.__df[col], kde=True, ax=axes[i],
                         color=colores[i], bins=30)
            axes[i].set_title(f"Distribucion de {col}", fontsize=9)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def histogramaClase(self, columna_objetivo):
        if columna_objetivo in self.__df.columns:
            plt.figure(figsize=(7, 4), dpi=130)
            colores = sns.color_palette("pastel")
            self.__df[columna_objetivo].value_counts().plot(
                kind='bar', color=colores)
            plt.title(f"Distribucion de la Clase: {columna_objetivo}")
            plt.xlabel(columna_objetivo)
            plt.ylabel("Frecuencia")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print(f"La columna '{columna_objetivo}' no existe en el DataFrame.")

    def datosDensidad(self):
        cols_num = self.__df.select_dtypes(include='number').columns
        n = len(cols_num)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas,
                                 figsize=(5 * columnas, 4 * filas), dpi=130)
        axes = axes.flatten()
        colores = sns.color_palette("husl", n)
        for i, col in enumerate(cols_num):
            sns.kdeplot(data=self.__df, x=col, fill=True,
                        ax=axes[i], color=colores[i], linewidth=2)
            axes[i].set_title(f"Densidad de {col}", fontsize=9)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Densidad")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def graficoCorrelacion(self):
        corr = self.__df.corr(numeric_only=True)
        plt.figure(figsize=(12, 8), dpi=130)
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True,
                    fmt=".2f", linewidths=0.5, linecolor='white', square=True,
                    cbar_kws={"shrink": 0.8, "label": "Correlacion"},
                    annot_kws={"size": 9, "color": "black"})
        plt.title("Mapa de Calor de Correlaciones", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def graficosDispersion(self):
        cols_num = self.__df.select_dtypes(include='number').columns
        if len(cols_num) >= 2:
            sns.pairplot(self.__df[cols_num])
            plt.suptitle("Graficos de Dispersion por Pares", y=1.02)
            plt.tight_layout()
            plt.show()
        else:
            print("No hay suficientes variables numericas.")


# ═════════════════════════════════════════════════════════════
#  CLASE SUPERVISADO
# ═════════════════════════════════════════════════════════════
class Supervisado():
    """
    Clasificacion
    Metodos de analisis por algoritmo (estandar + variaciones):
        KNN / DT / RF / XG / ADA

    Benchmarking:
        BM(): tabla estandar
        BMFamilias(mejores): benchmarking por familias de algoritmos
    """

    def __init__(self, df):
        self.__df = df

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, p_df):
        self.__df = p_df

    # ── Preparacion interna ────────────────────────────────────
    def __preparar_datos(self):
        X = self.__df.drop(columns=['target'])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        y = self.__df['target'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test, y

    # ── Modelos internos ───────────────────────────────────────
    def __modeloKNN(self, X_train, y_train, n_neighbors, algorithm):
        model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                     algorithm=algorithm)
        model.fit(X_train, y_train)
        return model

    def __modeloDT(self, X_train, y_train, min_samples_split, max_depth):
        model = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                       max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model

    def __modeloRF(self, X_train, y_train, n_estimators,
                   min_samples_split, max_depth):
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       min_samples_split=min_samples_split,
                                       max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model

    def __modeloXG(self, X_train, y_train, n_estimators,
                   min_samples_split, max_depth):
        model = GradientBoostingClassifier(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split,
                                           max_depth=max_depth,
                                           random_state=42)
        model.fit(X_train, y_train)
        return model

    def __modeloADA(self, X_train, y_train, estimator, n_estimators):
        model = AdaBoostClassifier(estimator=estimator,
                                   n_estimators=n_estimators,
                                   random_state=42)
        model.fit(X_train, y_train)
        return model

    # ── Prediccion y evaluacion ────────────────────────────────
    def __predecir(self, model, X_test):
        return model.predict(X_test)

    def __evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        indices = self.__indices_general(MC, list(np.unique(y)))
        for k in indices:
            print("\n%s:\n%s" % (k, str(indices[k])))

    def __indices_general(self, MC, nombres=None):
        precision_global    = np.sum(MC.diagonal()) / np.sum(MC)
        error_global        = 1 - precision_global
        precision_categoria = pd.DataFrame(
            MC.diagonal() / np.sum(MC, axis=1)).T
        if nombres is not None:
            precision_categoria.columns = nombres
        return {
            "Matriz de Confusion":     MC,
            "Precision Global":        precision_global,
            "Error Global":            error_global,
            "Precision por categoria": precision_categoria
        }

    # ── Captura interna de resultados ──────────────────────────
    def __evaluar_config(self, model, nombre_config):
        """Entrena y retorna dict con todas las metricas para una config."""
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        MC       = confusion_matrix(y_test, y_pred)
        indices  = self.__indices_general(MC, list(np.unique(y)))
        prec_cat = MC.diagonal() / np.sum(MC, axis=1)
        return {
            'Configuracion': nombre_config,
            'PG': round(indices['Precision Global'], 4),
            'EG': round(indices['Error Global'],     4),
            'PN': round(float(prec_cat[0]),           4),
            'PP': round(float(prec_cat[1]),           4),
            'MC': MC,
            'y_pred': y_pred,
            'y_test': y_test,
            'y':      y
        }

    # ── Metodos publicos de comparacion ───────────────────────
    def compararConfiguraciones(self, lista_configs, titulo):
        """
        Recibe lista de tuplas (nombre, modelo_sklearn).
        Entrena cada uno, imprime tabla comparativa y grafico.
        Retorna lista de dicts con resultados.
        """
        resultados = []
        for nombre, modelo in lista_configs:
            r = self.__evaluar_config(modelo, nombre)
            resultados.append(r)
            print(f"  {nombre:45s}  PG={r['PG']}  EG={r['EG']}")

        # Tabla
        tabla = pd.DataFrame([{
            'Configuracion':  r['Configuracion'],
            'Prec. Global':   r['PG'],
            'Error Global':   r['EG'],
            'Prec. Negativa': r['PN'],
            'Prec. Positiva': r['PP']
        } for r in resultados])
        print(f"\n{'='*65}")
        print(f"TABLA COMPARATIVA — {titulo}")
        print('='*65)
        print(tabla.to_string(index=False))

        # Grafico
        nombres   = [r['Configuracion'] for r in resultados]
        pgs       = [r['PG'] for r in resultados]
        mejor_idx = pgs.index(max(pgs))
        colores   = ['steelblue'] * len(nombres)
        colores[mejor_idx] = 'darkorange'

        plt.figure(figsize=(max(9, len(nombres) * 1.5), 4), dpi=120)
        plt.bar(nombres, pgs, color=colores, edgecolor='black')
        plt.axhline(y=max(pgs), color='darkorange', linestyle='--',
                    alpha=0.7, label=f'Mejor PG: {max(pgs)}')
        plt.title(f'{titulo} — Comparacion de Configuraciones', fontsize=11)
        plt.ylabel('Precision Global')
        plt.ylim(0, 1)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

        return resultados

    def mejorModelo(self, resultados, titulo):
        """
        Selecciona el modelo con mayor PG de la lista retornada por
        compararConfiguraciones() e imprime su analisis completo.
        Retorna el dict del mejor para usarlo en BMFamilias().
        """
        mejor = max(resultados, key=lambda r: r['PG'])
        print(f"\n{'='*65}")
        print(f"  MEJOR MODELO — {titulo}")
        print('='*65)
        print(f"  Configuracion    : {mejor['Configuracion']}")
        print(f"  Precision Global : {mejor['PG']}")
        print(f"  Error Global     : {mejor['EG']}")
        print(f"  Prec. Negativa   : {mejor['PN']}")
        print(f"  Prec. Positiva   : {mejor['PP']}")
        print(f"\n  Matriz de Confusion:")
        print(mejor['MC'])
        print(f"\n  Reporte de Clasificacion:")
        print(classification_report(
            mejor['y_test'], mejor['y_pred'],
            target_names=[str(c) for c in np.unique(mejor['y'])]))
        print('='*65)
        return mejor

    # ── Metodos publicos de clasificacion ─────────────────────
    def KNN(self, n_neighbors=5):
        """4 algoritmos de busqueda"""
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for algorithm in algorithms:
            print(f"\n{'='*50}")
            print(f"KNN | k={n_neighbors} | Algoritmo: {algorithm}")
            print('='*50)
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            y_train = y_train.astype(int)
            y_test  = y_test.astype(int)
            model   = self.__modeloKNN(X_train, y_train, n_neighbors, algorithm)
            y_pred  = self.__predecir(model, X_test)
            self.__evaluar(y_test, y_pred, y)

    def DT(self, min_samples_split=2, max_depth=4):
        print(f"\n{'='*50}")
        print(f"Decision Tree | mss={min_samples_split} | depth={max_depth}")
        print('='*50)
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, y)

    def RF(self, n_estimators=100, min_samples_split=2, max_depth=4):
        print(f"\n{'='*50}")
        print(f"Random Forest | ne={n_estimators} | mss={min_samples_split} "
              f"| depth={max_depth}")
        print('='*50)
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloRF(X_train, y_train, n_estimators,
                                 min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, y)

    def XG(self, n_estimators=100, min_samples_split=2, max_depth=4):
        print(f"\n{'='*50}")
        print(f"Gradient Boosting | ne={n_estimators} | mss={min_samples_split} "
              f"| depth={max_depth}")
        print('='*50)
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloXG(X_train, y_train, n_estimators,
                                 min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, y)

    def ADA(self, n_estimators=100):
        """ADABoost con tres estimadores base """
        estimators = {
            "Decision Tree": DecisionTreeClassifier(
                min_samples_split=2, max_depth=4),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, min_samples_split=2,
                max_depth=4, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, min_samples_split=2,
                max_depth=4, random_state=42)
        }
        for name, estimator in estimators.items():
            print(f"\n{'='*50}")
            print(f"ADABoost | Estimador base: {name} | ne={n_estimators}")
            print('='*50)
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            model  = self.__modeloADA(X_train, y_train, estimator, n_estimators)
            y_pred = self.__predecir(model, X_test)
            self.__evaluar(y_test, y_pred, y)

    # ── Benchmarking estandar ───────
    def __KNNBM(self, n_neighbors=5, algorithm='auto'):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloKNN(X_train, y_train.astype(int),
                                  n_neighbors, algorithm)
        y_pred = self.__predecir(model, X_test)
        MC     = confusion_matrix(y_test, y_pred)
        return self.__indices_general(MC, list(np.unique(y)))

    def __DTBM(self, min_samples_split=2, max_depth=4):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        MC     = confusion_matrix(y_test, y_pred)
        return self.__indices_general(MC, list(np.unique(y)))

    def __RFBM(self, n_estimators=100, min_samples_split=2, max_depth=4):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloRF(X_train, y_train, n_estimators,
                                 min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        MC     = confusion_matrix(y_test, y_pred)
        return self.__indices_general(MC, list(np.unique(y)))

    def __XGBM(self, n_estimators=100, min_samples_split=2, max_depth=4):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloXG(X_train, y_train, n_estimators,
                                 min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        MC     = confusion_matrix(y_test, y_pred)
        return self.__indices_general(MC, list(np.unique(y)))

    def __ADABM(self, estimator=None, n_estimators=100):
        if estimator is None:
            estimator = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model  = self.__modeloADA(X_train, y_train, estimator, n_estimators)
        y_pred = self.__predecir(model, X_test)
        MC     = confusion_matrix(y_test, y_pred)
        return self.__indices_general(MC, list(np.unique(y)))

    def BM(self):
        """
        Benchmarking estandar
        Compara los 5 algoritmos con configuracion por defecto.
        """
        datos  = {"PG": [0.0]*5, "EG": [0.0]*5, "PP": [0.0]*5, "PN": [0.0]*5}
        Tdatos = pd.DataFrame(datos,
                              index=["AlgKnn", "AlgDT", "AlgRF",
                                     "AlgXGBoost", "AlgADABoost"],
                              columns=["PG", "EG", "PP", "PN"])

        algoritmos = [
            ("AlgKnn",      self.__KNNBM),
            ("AlgDT",       self.__DTBM),
            ("AlgRF",       self.__RFBM),
            ("AlgXGBoost",  self.__XGBM),
            ("AlgADABoost", self.__ADABM),
        ]

        for alg_name, alg_method in algoritmos:
            indices = alg_method()
            PP = indices['Precision por categoria']
            Tdatos.loc[alg_name, "PG"] = round(indices['Precision Global'], 4)
            Tdatos.loc[alg_name, "EG"] = round(indices['Error Global'],     4)
            Tdatos.loc[alg_name, "PP"] = round(float(PP.iloc[0, 1]),        4)
            Tdatos.loc[alg_name, "PN"] = round(float(PP.iloc[0, 0]),        4)

        print("\n" + "="*55)
        print("BENCHMARKING — Configuracion Estandar")
        print("PG=Precision Global | EG=Error Global")
        print("PP=Precision Clase Positiva | PN=Precision Clase Negativa")
        print("="*55)
        print(Tdatos.to_string())
        return Tdatos

    # ── Benchmarking por familias ──────────────────────────────
    def BMFamilias(self, mejores):
        """
        Benchmarking entre familias de algoritmos usando los mejores
        modelos identificados en el analisis.

        Parametro:
            mejores: dict  {  'KNN': dict_resultado,
                               'DT':  dict_resultado,
                               'RF':  dict_resultado,
                               'XG':  dict_resultado,
                               'ADA': dict_resultado  }
            Cada valor es el dict retornado por mejorModelo().

        Familias:
            Distancia  → KNN
            Arboles    → DT, RF
            Boosting   → XG, ADA
        """
        familias = {
            "Distancia (KNN)":     ["KNN"],
            "Arboles (DT/RF)":     ["DT", "RF"],
            "Boosting (XG/ADA)":   ["XG", "ADA"],
        }

        filas = []
        for familia, claves in familias.items():
            for clave in claves:
                r = mejores[clave]
                filas.append({
                    "Familia":        familia,
                    "Algoritmo":      clave,
                    "Config":         r['Configuracion'],
                    "Prec. Global":   r['PG'],
                    "Error Global":   r['EG'],
                    "Prec. Negativa": r['PN'],
                    "Prec. Positiva": r['PP'],
                })

        tabla = pd.DataFrame(filas)
        print("\n" + "="*70)
        print("BENCHMARKING POR FAMILIAS DE ALGORITMOS — Potabilidad")
        print("="*70)
        print(tabla.to_string(index=False))

        print("\n--- Mejor algoritmo por familia ---")
        for familia in tabla['Familia'].unique():
            sub = tabla[tabla['Familia'] == familia]
            row = sub.loc[sub['Prec. Global'].idxmax()]
            print(f"  {familia:28s} → {row['Algoritmo']:4s} "
                  f"Config: {row['Config']}  PG={row['Prec. Global']}")

        # Grafico por familias
        from matplotlib.patches import Patch
        col_familia = {
            "Distancia (KNN)":   "steelblue",
            "Arboles (DT/RF)":   "mediumseagreen",
            "Boosting (XG/ADA)": "darkorange",
        }
        bar_colors = [col_familia[f] for f in tabla['Familia']]
        nombres    = tabla['Algoritmo'].tolist()
        x          = np.arange(len(nombres))
        w          = 0.25

        plt.figure(figsize=(11, 5), dpi=120)
        plt.bar(x - w, tabla['Prec. Global'],   w,
                color=bar_colors, edgecolor='black', label='Global')
        plt.bar(x,     tabla['Prec. Negativa'],  w,
                color=bar_colors, edgecolor='black', alpha=0.6,
                hatch='//', label='Negativa')
        plt.bar(x + w, tabla['Prec. Positiva'],  w,
                color=bar_colors, edgecolor='black', alpha=0.4,
                hatch='xx', label='Positiva')
        plt.xticks(x, nombres)
        plt.ylim(0, 1)
        plt.ylabel('Precision')
        plt.title('Benchmarking por Familias — Potabilidad', fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        leyenda = [Patch(color=c, label=f)
                   for f, c in col_familia.items()]
        plt.legend(handles=leyenda, fontsize=9)
        plt.tight_layout()
        plt.show()

        return tabla

# ═════════════════════════════════════════════════════════════
#  CLASE REGRESION  (extiende el estilo de Supervisado)
# ═════════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Regresion():
    """
    Familias de algoritmos:
      Lineal    : LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
      SVM       : SVR
      Arboles   : DecisionTreeRegressor, RandomForestRegressor
    """

    def __init__(self, df):
        self.__df = df

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, p_df):
        self.__df = p_df

    # ── Preparacion interna (igual que Supervisado) ────────────
    def __preparar_datos(self):
        X = self.__df.drop(columns=['target'])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        y = self.__df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    # ── Modelos internos ───────────────────────────────────────
    def __modeloLR(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def __modeloLasso(self, X_train, y_train, alpha):
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        return model

    def __modeloLassoCV(self, X_train, y_train, cv):
        model = LassoCV(cv=cv, max_iter=10000)
        model.fit(X_train, y_train)
        return model

    def __modeloRidge(self, X_train, y_train, alpha):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def __modeloRidgeCV(self, X_train, y_train, alphas, cv):
        model = RidgeCV(alphas=alphas, cv=cv)
        model.fit(X_train, y_train)
        return model

    def __modeloSVR(self, X_train, y_train, kernel, C, epsilon):
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train, y_train)
        return model

    def __modeloDT(self, X_train, y_train, min_samples_split, max_depth):
        model = DecisionTreeRegressor(min_samples_split=min_samples_split,
                                      max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model

    def __modeloRF(self, X_train, y_train, n_estimators,
                   min_samples_split, max_depth):
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      min_samples_split=min_samples_split,
                                      max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model

    # ── Prediccion y evaluacion ────────────────────────────────
    def __predecir(self, model, X_test):
        return model.predict(X_test)

    def __evaluar(self, y_test, y_pred, nombre=""):
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        mae  = round(mean_absolute_error(y_test, y_pred), 4)
        r2   = round(r2_score(y_test, y_pred), 4)
        print(f"\nRMSE : {rmse}")
        print(f"MAE  : {mae}")
        print(f"R²   : {r2}")
        return {'Modelo': nombre, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    # ── Captura interna de resultados ──────────────────────────
    def __evaluar_config(self, model, nombre_config):
        """Entrena y retorna dict con metricas para una configuracion."""
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse   = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        mae    = round(mean_absolute_error(y_test, y_pred), 4)
        r2     = round(r2_score(y_test, y_pred), 4)
        return {
            'Configuracion': nombre_config,
            'RMSE': rmse,
            'MAE':  mae,
            'R2':   r2
        }

    # ── Metodos publicos de comparacion (reutilizados de Supervisado) ─
    def compararConfiguraciones(self, lista_configs, titulo):
        """
        Recibe lista de tuplas (nombre, modelo_sklearn).
        Entrena cada uno, imprime tabla comparativa y grafico.
        Retorna lista de dicts con resultados.
        Criterio de mejor modelo: mayor R².
        """
        resultados = []
        for nombre, modelo in lista_configs:
            r = self.__evaluar_config(modelo, nombre)
            resultados.append(r)
            print(f"  {nombre:45s}  R2={r['R2']}  RMSE={r['RMSE']}")

        # Tabla
        tabla = pd.DataFrame([{
            'Configuracion': r['Configuracion'],
            'RMSE':          r['RMSE'],
            'MAE':           r['MAE'],
            'R²':            r['R2']
        } for r in resultados])
        print(f"\n{'='*65}")
        print(f"TABLA COMPARATIVA — {titulo}")
        print('='*65)
        print(tabla.to_string(index=False))

        # Grafico (R²)
        nombres   = [r['Configuracion'] for r in resultados]
        r2s       = [r['R2'] for r in resultados]
        mejor_idx = r2s.index(max(r2s))
        colores   = ['steelblue'] * len(nombres)
        colores[mejor_idx] = 'darkorange'

        plt.figure(figsize=(max(9, len(nombres) * 1.5), 4), dpi=120)
        plt.bar(nombres, r2s, color=colores, edgecolor='black')
        plt.axhline(y=max(r2s), color='darkorange', linestyle='--',
                    alpha=0.7, label=f'Mejor R²: {max(r2s)}')
        plt.title(f'{titulo} — Comparacion de Configuraciones (R²)', fontsize=11)
        plt.ylabel('R²')
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

        return resultados

    def mejorModelo(self, resultados, titulo):
        """
        Selecciona el modelo con mayor R² e imprime su analisis completo.
        Retorna el dict del mejor para usarlo en BMFamilias().
        """
        mejor = max(resultados, key=lambda r: r['R2'])
        print(f"\n{'='*65}")
        print(f"  MEJOR MODELO — {titulo}")
        print('='*65)
        print(f"  Configuracion : {mejor['Configuracion']}")
        print(f"  R²            : {mejor['R2']}")
        print(f"  RMSE          : {mejor['RMSE']}")
        print(f"  MAE           : {mejor['MAE']}")
        print('='*65)
        return mejor

    # ── Metodos publicos de regresion ──────────────────────────
    def LR(self):
        """Regresion Lineal simple y multiple (mismas X, diferente interpretacion)."""
        print(f"\n{'='*50}")
        print("Regresion Lineal (LinearRegression)")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloLR(X_train, y_train)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, "LinearRegression")
        print(f"  Coeficientes: {dict(zip(self.__df.drop(columns=['target']).columns, model.coef_.round(4)))}")

    def LassoReg(self, alpha=1.0):
        print(f"\n{'='*50}")
        print(f"Lasso | alpha={alpha}")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloLasso(X_train, y_train, alpha)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, f"Lasso(alpha={alpha})")

    def LassoCVReg(self, cv=5):
        print(f"\n{'='*50}")
        print(f"LassoCV | cv={cv}")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloLassoCV(X_train, y_train, cv)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, f"LassoCV(cv={cv})")
        print(f"  Mejor alpha encontrado: {round(model.alpha_, 4)}")

    def RidgeReg(self, alpha=1.0):
        print(f"\n{'='*50}")
        print(f"Ridge | alpha={alpha}")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloRidge(X_train, y_train, alpha)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, f"Ridge(alpha={alpha})")

    def RidgeCVReg(self, alphas=(0.1, 1.0, 10.0), cv=5):
        print(f"\n{'='*50}")
        print(f"RidgeCV | alphas={alphas} | cv={cv}")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloRidgeCV(X_train, y_train, alphas, cv)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, f"RidgeCV")
        print(f"  Mejor alpha encontrado: {round(model.alpha_, 4)}")

    def SVReg(self, kernel='rbf', C=1.0, epsilon=0.1):
        print(f"\n{'='*50}")
        print(f"SVR | kernel={kernel} | C={C} | epsilon={epsilon}")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloSVR(X_train, y_train, kernel, C, epsilon)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, f"SVR(kernel={kernel})")

    def DT(self, min_samples_split=2, max_depth=4):
        print(f"\n{'='*50}")
        print(f"DecisionTreeRegressor | mss={min_samples_split} | depth={max_depth}")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, "DecisionTreeRegressor")

    def RF(self, n_estimators=100, min_samples_split=2, max_depth=4):
        print(f"\n{'='*50}")
        print(f"RandomForestRegressor | ne={n_estimators} | mss={min_samples_split} | depth={max_depth}")
        print('='*50)
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloRF(X_train, y_train, n_estimators,
                                 min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, "RandomForestRegressor")

    # ── Benchmarking estandar ──────────────────────────────────
    def __LRBM(self):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloLR(X_train, y_train)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def __LassoBM(self, alpha=1.0):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloLasso(X_train, y_train, alpha)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def __LassoCVBM(self, cv=5):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloLassoCV(X_train, y_train, cv)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def __RidgeBM(self, alpha=1.0):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloRidge(X_train, y_train, alpha)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def __RidgeCVBM(self, alphas=(0.1, 1.0, 10.0), cv=5):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloRidgeCV(X_train, y_train, alphas, cv)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def __SVRBM(self, kernel='rbf', C=1.0, epsilon=0.1):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloSVR(X_train, y_train, kernel, C, epsilon)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def __DTBM(self, min_samples_split=2, max_depth=4):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def __RFBM(self, n_estimators=100, min_samples_split=2, max_depth=4):
        X_train, X_test, y_train, y_test = self.__preparar_datos()
        model  = self.__modeloRF(X_train, y_train, n_estimators,
                                 min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        return {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
            'R2':   round(r2_score(y_test, y_pred), 4)
        }

    def BM(self):
        """
        Benchmarking estandar — replica el estilo BM() del profesor.
        Compara los 8 algoritmos con configuracion por defecto.
        Metricas: RMSE, MAE, R²
        """
        algoritmos = [
            ("LinearRegression", self.__LRBM),
            ("Lasso",            self.__LassoBM),
            ("LassoCV",          self.__LassoCVBM),
            ("Ridge",            self.__RidgeBM),
            ("RidgeCV",          self.__RidgeCVBM),
            ("SVR",              self.__SVRBM),
            ("DecisionTree",     self.__DTBM),
            ("RandomForest",     self.__RFBM),
        ]

        datos  = {"RMSE": [0.0]*8, "MAE": [0.0]*8, "R2": [0.0]*8}
        Tdatos = pd.DataFrame(datos,
                              index=[a[0] for a in algoritmos],
                              columns=["RMSE", "MAE", "R2"])

        for alg_name, alg_method in algoritmos:
            r = alg_method()
            Tdatos.loc[alg_name, "RMSE"] = r['RMSE']
            Tdatos.loc[alg_name, "MAE"]  = r['MAE']
            Tdatos.loc[alg_name, "R2"]   = r['R2']

        print("\n" + "="*55)
        print("BENCHMARKING REGRESION — Configuracion Estandar")
        print("RMSE=Raiz Error Cuadratico Medio | MAE=Error Absoluto Medio")
        print("R²=Coeficiente de Determinacion (mayor es mejor)")
        print("="*55)
        print(Tdatos.to_string())
        return Tdatos

    # ── Benchmarking por familias ──────────────────────────────
    def BMFamilias(self, mejores):
        """
        Benchmarking entre familias usando los mejores modelos
        encontrados en el analisis.

        Parametro:
            mejores: dict con claves:
                'LR', 'Lasso', 'LassoCV', 'Ridge', 'RidgeCV',
                'SVR', 'DT', 'RF'
            Cada valor es el dict retornado por mejorModelo().

        Familias:
            Lineal: LR, Lasso, LassoCV, Ridge, RidgeCV
            SVM: SVR
            Arboles: DT, RF
        """
        familias = {
            "Lineal (LR/Lasso/Ridge)": ["LR", "Lasso", "LassoCV", "Ridge", "RidgeCV"],
            "SVM (SVR)":               ["SVR"],
            "Arboles (DT/RF)":         ["DT", "RF"],
        }

        filas = []
        for familia, claves in familias.items():
            for clave in claves:
                r = mejores[clave]
                filas.append({
                    "Familia":  familia,
                    "Algoritmo": clave,
                    "Config":   r['Configuracion'],
                    "RMSE":     r['RMSE'],
                    "MAE":      r['MAE'],
                    "R²":       r['R2'],
                })

        tabla = pd.DataFrame(filas)
        print("\n" + "="*70)
        print("BENCHMARKING POR FAMILIAS — Regresion (Diabetes → glucosa)")
        print("="*70)
        print(tabla.to_string(index=False))

        print("\n--- Mejor algoritmo por familia (mayor R²) ---")
        for familia in tabla['Familia'].unique():
            sub = tabla[tabla['Familia'] == familia]
            row = sub.loc[sub['R²'].idxmax()]
            print(f"  {familia:32s} → {row['Algoritmo']:8s} "
                  f"R²={row['R²']}  RMSE={row['RMSE']}")

        # Grafico por familias
        from matplotlib.patches import Patch
        col_familia = {
            "Lineal (LR/Lasso/Ridge)": "steelblue",
            "SVM (SVR)":               "darkorange",
            "Arboles (DT/RF)":         "mediumseagreen",
        }
        bar_colors = [col_familia[f] for f in tabla['Familia']]
        nombres    = tabla['Algoritmo'].tolist()
        x          = np.arange(len(nombres))
        w          = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

        # R²
        axes[0].bar(x, tabla['R²'], color=bar_colors, edgecolor='black')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(nombres)
        axes[0].set_ylabel('R²')
        axes[0].set_title('R² por algoritmo', fontsize=10)
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)

        # RMSE
        axes[1].bar(x, tabla['RMSE'], color=bar_colors, edgecolor='black')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(nombres)
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE por algoritmo (menor es mejor)', fontsize=10)
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)

        leyenda = [Patch(color=c, label=f) for f, c in col_familia.items()]
        axes[0].legend(handles=leyenda, fontsize=8)
        fig.suptitle('Benchmarking por Familias — Regresion (Diabetes → glucosa)',
                     fontsize=11)
        plt.tight_layout()
        plt.show()

        return tabla