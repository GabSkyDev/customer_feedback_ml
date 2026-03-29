"""
Módulo para construção e treinamento do modelo de classificação de sentimentos.

Este módulo contém funções para criar pipelines de processamento de texto,
otimizar hiperparâmetros e treinar modelos de machine learning.
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def criar_pipeline():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um']
        )),
        ('scaler', StandardScaler(with_mean=False)),
        ('logreg', LogisticRegression(
            solver='liblinear',
            random_state=42,
            max_iter=1000
        ))
    ])
    
    return pipeline


def otimizar_hiperparametros(pipeline, X_treino, y_treino, cv=5, n_jobs=-1):
    parametros_grid = {
        'tfidf__max_features': [500, 1000, 2000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'logreg__C': [0.1, 1, 10],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__max_iter': [5000, 6000]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        parametros_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring='accuracy',
        verbose=1
    )
    
    print("Inicializando a otimização de hiperparâmetros...")
    grid_search.fit(X_treino, y_treino)
    
    print(f"Melhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)
    
    return grid_search, grid_search.best_params_


def obter_melhor_modelo(grid_search):
    return grid_search.best_estimator_


def avaliar_modelo(modelo, X_teste, y_teste):
    predicoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, predicoes)
    relatorio = classification_report(
        y_teste,
        predicoes,
        target_names=['Negativo', 'Positivo']
    )
    matriz_confusao = confusion_matrix(y_teste, predicoes)
    
    print(f"Acurácia do Modelo: {acuracia:.2%}\n")
    print("Relatório de classificação:")
    print(relatorio)
    
    return {
        'acuracia': acuracia,
        'predicoes': predicoes,
        'relatorio': relatorio,
        'matriz_confusao': matriz_confusao
    }


def salvar_modelo(modelo, caminho_arquivo='models/model_feeedback.pkl'):
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    joblib.dump(modelo, caminho_arquivo)
    print(f"Modelo salvo com sucesso em: {caminho_arquivo}")
    return caminho_arquivo


def carregar_modelo(caminho_arquivo='models/model_feedbaack.pkl'):
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {caminho_arquivo}")
    
    modelo = joblib.load(caminho_arquivo)
    print(f"Modelo carregado com sucesso de: {caminho_arquivo}")
    return modelo


def construir_e_treinar_modelo(X_treino, y_treino, X_teste, y_teste, salvar=True):
    # Criar pipeline
    pipeline = criar_pipeline()
    
    # Otimizar hiperparâmetros
    grid_search, best_params = otimizar_hiperparametros(pipeline, X_treino, y_treino)
    
    # Obter melhor modelo
    melhor_modelo = obter_melhor_modelo(grid_search)
    
    # Avaliar modelo
    metricas = avaliar_modelo(melhor_modelo, X_teste, y_teste)
    
    # Salvar modelo se solicitado
    caminho_modelo = None
    if salvar:
        caminho_modelo = salvar_modelo(melhor_modelo)
    
    return {
        'modelo': melhor_modelo,
        'grid_search': grid_search,
        'metricas': metricas,
        'caminho_modelo': caminho_modelo
    }
