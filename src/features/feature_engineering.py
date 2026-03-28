import pandas as pd
from src.features.clean_text import limpeza_texto

def feature_engineering(clean_df: pd.DataFrame) -> pd.DataFrame:
    print("Aplicando features...")
    clean_df['texto_limpo'] = clean_df['texto_review'].apply(limpeza_texto)

    clean_df['sentimento_value'] = clean_df['sentimento'].map({'positivo': 1, 'negativo': 0})
    clean_df = clean_df.drop(columns=['review_id', 'texto_review', 'sentimento'])
    
    return clean_df