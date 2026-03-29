"""
Módulo para fazer previsões e exibir resultados do modelo de classificação de sentimentos.

Este módulo contém funções para realizar previsões em novos textos e exibir
os resultados de forma formatada e legível.
"""


def fazer_predicoes(modelo, novos_textos):
    """
    Realiza previsões em novos textos usando o modelo treinado.
    
    Parameters
    ----------
    modelo : sklearn.pipeline.Pipeline
        Modelo treinado para fazer previsões
    novos_textos : list ou str
        Um ou mais textos para classificar
    
    Returns
    -------
    dict
        Dicionário contendo:
        - 'textos': Lista dos textos processados
        - 'predicoes': Array com as previsões (0 = Negativo, 1 = Positivo)
        - 'probabilidades': Array com as probabilidades de cada classe
        - 'sentimentos': Lista com os nomes dos sentimentos
    """
    # Converter em lista se for string única
    if isinstance(novos_textos, str):
        novos_textos = [novos_textos]
    
    # Fazer previsões
    predicoes = modelo.predict(novos_textos)
    probabilidades = modelo.predict_proba(novos_textos)
    
    # Mapear previsões para nomes de sentimentos
    mapa_sentimentos = {0: 'Negativo', 1: 'Positivo'}
    sentimentos = [mapa_sentimentos[pred] for pred in predicoes]
    
    return {
        'textos': novos_textos,
        'predicoes': predicoes,
        'probabilidades': probabilidades,
        'sentimentos': sentimentos
    }


def exibir_predicoes(resultado_predicoes):
    """
    Exibe as previsões de forma formatada e legível.
    
    Parameters
    ----------
    resultado_predicoes : dict
        Dicionário retornado pela função fazer_predicoes()
    """
    textos = resultado_predicoes['textos']
    sentimentos = resultado_predicoes['sentimentos']
    probabilidades = resultado_predicoes['probabilidades']
    
    print("\n" + "="*80)
    print("PREVISÕES DO MODELO")
    print("="*80)
    
    for i, (texto, sentimento, probs) in enumerate(zip(textos, sentimentos, probabilidades), 1):
        confianca = max(probs) * 100
        print(f"\n[Review {i}]")
        print(f"Texto: {texto}")
        print(f"Sentimento: {sentimento}")
        print(f"Confiança: {confianca:.2f}%")
        print(f"  - Negativo: {probs[0]:.2%}")
        print(f"  - Positivo: {probs[1]:.2%}")
        print("-" * 80)
