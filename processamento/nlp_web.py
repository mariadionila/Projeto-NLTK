import nltk
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from pysentimiento import create_analyzer


#instalando os recursos NLTK:
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('rslp')


noticias_web = "../dados/noticias_adultizacao.csv"
df_dados_web = pd.read_csv(noticias_web)
df_dados_web["Texto Limpo"] = df_dados_web["Texto Limpo"].fillna("").astype(str)


df_dados_web.columns = df_dados_web.columns.str.strip() #removendo os espaços extras

stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()
analyzer = create_analyzer(task="sentiment", lang="pt")

# Função para processar o texto
def processar_texto(texto):
    sentencas = sent_tokenize(texto, language='portuguese')
    resultado = []
    scores = []

    for sentenca in sentencas:
        # Pré-processamento opcional, se quiser usar tokens/stem
        palavras = word_tokenize(sentenca, language='portuguese')
        palavras = [p.lower() for p in palavras if p.isalpha()]
        palavras = [p for p in palavras if p not in stop_words]
        palavras_stem = [stemmer.stem(p) for p in palavras]

        # Use o texto original para o analyzer
        sentimento_obj = analyzer.predict(sentenca)
        label = sentimento_obj.output
        probas = sentimento_obj.probas

        # Calcula score
        score = probas.get('POS', 0) - probas.get('NEG', 0)
        scores.append(score)

        resultado.append({
            'sentenca': sentenca,
            'tokens': palavras,
            'tokens_stem': palavras_stem,
            'sentimento_label': label,
            'sentimento_probas': probas
        })

    sentimento_medio = sum(scores)/len(scores) if scores else 0
    return resultado, sentimento_medio

# Aplicar a função ao DataFrame
df_dados_web[['analise_sentimento', 'sentimento_medio']] = df_dados_web['Texto Limpo'].apply(
    lambda x: pd.Series(processar_texto(x)))

# Exibir as primeiras análises
df_dados_web.to_csv("../dados/noticias_web_com_sentimento.csv", index=False)
print("analise de sentimento concluida com sucesso!!")
