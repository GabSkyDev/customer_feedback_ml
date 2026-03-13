import pandas as pd

def processing_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Processando a limpeza dos dados...")
    df.dropna(subset= ['texto_review'], inplace=True)
    return df