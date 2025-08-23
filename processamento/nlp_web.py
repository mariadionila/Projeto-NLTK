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

df_dados_web.columns = df_dados_web.columns.str.strip() #removendo os espaços extras

stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()
analyzer = create_analyzer(task="sentiment", lang="pt")

# Função para processar o texto
def processar_texto(texto):
    
    sentencas = sent_tokenize(texto, language='portuguese')
    resultado = []
    scores= [] #média dos sentimentos
    
    for sentenca in sentencas:
        palavras = word_tokenize(sentenca, language='portuguese') # Tokenização em palavras
        palavras = [p.lower() for p in palavras if p.isalpha()]  # Normalização
        palavras = [p for p in palavras if p not in stop_words]  # Remoção de stopwords
        palavras_stem = [stemmer.stem(p) for p in palavras] # Stemming
        
        # Análise de sentimento
        sentimento_obj = analyzer.predict(' '.join(palavras_stem))
        label = sentimento_obj.output
        probas = sentimento_obj.probas

        if label == 'POS':
            score = probas['POS']
        elif label == 'NEG':
            score = -probas['NEG']
        else:
            score = 0

        scores.append(score)
        
        resultado.append({
            'sentenca': sentenca,
            'tokens': palavras_stem,
            'sentimento_label': label, 
            'sentimento_probas': probas
        })
    sentimento_medio = sum(scores)/len(scores) if scores else 0

    return resultado, sentimento_medio

# Aplicar a função ao DataFrame
df_dados_web[['analise_sentimento', 'sentimento_medio']] = df_dados_web['Texto Completo'].apply(
    lambda x: pd.Series(processar_texto(x)))

# Exibir as primeiras análises
df_dados_web.to_csv("../dados/noticias_web_com_sentimento.csv", index=False)
print("analise de sentimento concluida com sucesso!!")
