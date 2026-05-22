"""
Created on Sun Jul 21 09:54:07 2024

@authors:
    Antonio Pires
    Milton Ávila
    João Gabriel
    Wesley Oliveira

@license:
Este projeto está licenciado sob a Licença Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Você pode compartilhar, adaptar e construir sobre o material, desde que atribua crédito apropriado, não use o material para fins comerciais e distribua suas contribuições sob a mesma licença.
Para mais informações, consulte o arquivo [LICENSE](./LICENSE).
"""
import re
import nltk

class TextUtils:
    _spacy_models = {} # Cache para modelos spaCy

    @staticmethod
    def filter_special_characters(txt: str, change_to = "") -> str:
        return re.sub(r"[^\w\s]", change_to, txt)

    @staticmethod
    def filter_spaces(txt: str, change_to = " ") -> str:
        txt_limpo = re.sub(r"[\x00-\x1F]+", "", txt)
        return re.sub(r"\s+", change_to, txt_limpo).strip()

    @staticmethod
    def filter_numbers(txt: str, change_to = "") -> str:
        return re.sub(r"\d", change_to, txt)

    @staticmethod
    def filter_links(txt: str, change_to="") -> str:
        pattern = r"""
            (?:https?://|www\.) # Protocolo (http/https) ou www
            \S+                 # Um ou mais caracteres que não sejam espaço
        """
        return re.sub(pattern, change_to, txt, flags=re.VERBOSE)

    @staticmethod
    def filter_email(txt: str, change_to="") -> str:
        pattern = r"""
            (?:mailto:)?       # Prefixo opcional mailto:
            [\w.+-]+           # Nome do usuário (letras, números, pontos, etc)
            @                  # Arroba
            [\w-]+             # Domínio
            (?:\.[\w-]+)+      # Extensão (.com, .com.br, etc)
        """
        return re.sub(pattern, change_to, txt, flags=re.VERBOSE)

    @staticmethod
    def filter_cnpj(txt: str, change_to="") -> str:
        # Note como o uso do VERBOSE permite explicar a lógica complexa do CNPJ
        pattern = r"""
            \bcnpj(?:/mf)?            # Palavra 'cnpj' ou 'cnpj/mf'
            (?:\s+sob)?               # ' sob' opcional
            (?:\s+(?:n\S*|numero))?   # ' n', 'nº' ou 'numero' opcional
            \s*:?\s* # Espaços e dois pontos opcionais
            \d{2,3}\.?\d{3}\.?\d{3}/? # Início do número (12.345.678/)
            \d{4}-?\d{2}\b            # Final do número (0001-90)
            |                         # --- OU APENAS O NÚMERO ---
            \b\d{2,3}\.?\d{3}\.?      # Formato numérico puro
            \d{3}/\d{4}-\d{2}\b
        """
        return re.sub(pattern, change_to, txt, flags=re.VERBOSE | re.IGNORECASE)

    @staticmethod
    def filter_cpf(txt: str, change_to = "") -> str:
        pattern = r"\bcpf(?:/mf)?(?:\s+sob)?(?:\s+(?:n\S*|numero))?\s*:?\s*\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b|\b\d{3}\.?\d{3}\.?\d{3}-\d{2}\b"
        return re.sub(pattern, change_to, txt, flags=re.IGNORECASE)

    @staticmethod
    def filter_rg(txt: str, change_to="") -> str:
        pattern = r"""
            \brg\b                  # Palavra 'rg' com limite de borda
            (?:\s+(?:n\S*|numero))? # Opcional: ' n°', ' n.', ' numero'
            \s*:?\s*                # Opcional: dois pontos e espaços
            \d{4,14}\b              # De 4 a 14 dígitos numéricos
        """
        return re.sub(pattern, change_to, txt, flags=re.IGNORECASE | re.VERBOSE)

    @staticmethod
    def filter_cep(txt: str, change_to="") -> str:
        pattern = r"""
            \bcep\b                 # Palavra 'cep'
            (?:\s+(?:n\S*|numero))? # Opcional: ' n°', etc
            \s*:?\s*                # Opcional: dois pontos e espaços
            \d{5}-?\d{3}\b          # Formato 00000-000 ou 00000000
            |                       # --- OU ---
            \b\d{5}-\d{3}\b         # Apenas o número formatado 00000-000
        """
        return re.sub(pattern, change_to, txt, flags=re.IGNORECASE | re.VERBOSE)

    @staticmethod
    def filter_oab(txt: str, change_to="") -> str:
        pattern = r"""
            \boab\b                 # Palavra 'oab'
            \s*[/\-]?\s*            # Opcional: barra ou hífen
            [a-z]{2}                # Sigla do estado (ex: SP, RJ)
            \s*                     # Espaço opcional
            (?:
                \d{1,3}(?:[.\s]?\d{3})+ # Formato com pontos (ex: 123.456)
                |                       # --- OU ---
                \d{4,12}                # Apenas números sequenciais
            )\b
        """
        return re.sub(pattern, change_to, txt, flags=re.IGNORECASE | re.VERBOSE)

    @staticmethod
    def filter_telefone(txt: str, change_to="") -> str:
        pattern = r"""
            (?:(?<=\s)|^)          # Lookbehind para espaço ou início
            \(?\d{2}\)?            # DDD com parênteses opcionais
            [-\s.]*                # Separador opcional entre DDD e número (hífen, espaço ou ponto)
            (?:9[-\s.]?)?          # 9 opcional (pode ter hífen/espaço depois)
            \d{4}[-.\s]?\d{4}      # Prefixo e sufixo (8 dígitos restantes)
            (?=(?:\s|[.,;:)]|$))   # Lookahead para pontuação ou fim
            |                      # --- OU ---
            (?:(?<=\s)|^)9\d{4}    # Celular começando com 9 sem DDD
            [-.\s]?\d{4}
            (?=(?:\s|[.,;:)]|$))
            |                      # --- OU ---
            (?:(?<=\s)|^)9\d{8}    # Formato grudado total
            (?=(?:\s|[.,;:)]|$))
        """

        return re.sub(pattern, change_to, txt, flags=re.VERBOSE)

    @staticmethod
    def remove_stopwords(txt: str, language = "portuguese") -> str:
        try:
            from nltk.corpus import stopwords
            # Check if stopwords for the language are available
            stopwords.words(language)
        except LookupError:
            raise RuntimeError(
                f"Recurso NLTK 'stopwords' para '{language}' ausente. "
                f"Por favor, execute no seu terminal: python -m nltk.downloader stopwords"
            )
        
        stopwords_set = set(stopwords.words(language))
        tokens = txt.split()
        filtered_tokens = [word for word in tokens if word not in stopwords_set]
        return " ".join(filtered_tokens)

    @staticmethod
    def remove_html(txt: str) -> str:
        """Remove tags HTML de uma string usando BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("BeautifulSoup is required for remove_html. Please install it: pip install beautifulsoup4")
        
        soup = BeautifulSoup(txt, 'html.parser')
        return soup.get_text()

    @staticmethod
    def lemmatize(txt: str, core = 'pt_core_news_sm') -> str:
        """Lemmatiza palavras em uma string usando Spacy."""
        if core not in TextUtils._spacy_models:
            try:
                import spacy
                TextUtils._spacy_models[core] = spacy.load(core)
            except OSError:
                raise OSError(
                    f"Modelo SpaCy '{core}' não encontrado. "
                    f"Por favor, faça o download executando no seu terminal: "
                    f"python -m spacy download {core}"
                )
        
        nlp = TextUtils._spacy_models[core]
        doc = nlp(txt)
        return " ".join([token.lemma_.lower() for token in doc])

    @staticmethod
    def stemming(txt: str, stem_language = "portuguese") -> str:
        """Deriva palavras em uma string usando SnowballStemmer."""

        try:
            from nltk.stem.snowball import SnowballStemmer
            stemmer_pt = SnowballStemmer(stem_language)
        except ImportError:
            raise ImportError("NLTK é necessário. Instale: pip install nltk")

        stemmer_pt = SnowballStemmer(stem_language)
    
        words = txt.split()
        stemmed_words = [stemmer_pt.stem(word) for word in words]
        return " ".join(stemmed_words)

    @staticmethod
    def tokenize(txt: str) ->list[str]:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            raise RuntimeError(
                "Recurso NLTK 'punkt' ausente. "
                "Por favor, execute no seu terminal: python -m nltk.downloader punkt"
            )
        return nltk.word_tokenize(txt)
    
    @staticmethod
    def normalizacao_hibrida(txt: str, core = "pt_core_news_sm", stem_language = "portuguese") -> list[str]:
        """
        Executa os passos: POS Tagging, Remoção de Pontuação, 
        Lematização, Stemming Seletivo e Filtro de tamanho mínimo.
        Retorna uma lista de tokens (strings).
        """
        # Garante o carregamento do modelo spaCy
        if core not in TextUtils._spacy_models:
            try:
                import spacy
                TextUtils._spacy_models[core] = spacy.load(core)
            except OSError:
                raise OSError(f"Modelo SpaCy '{core}' não encontrado. Execute: python -m spacy download {core}")
        
        # Garante o carregamento do SnowballStemmer do NLTK
        try:
            from nltk.stem.snowball import SnowballStemmer
            stemmer_pt = SnowballStemmer(stem_language)
        except ImportError:
            raise ImportError("NLTK é necessário. Instale: pip install nltk")

        nlp = TextUtils._spacy_models[core]
        doc = nlp(txt)
        
        tokens_final = []
        
        for token in doc:
            # 1. Remoção de pontuação e espaços
            if token.is_punct or token.is_space:
                continue
                
            # 2. Lematização base
            lema = token.lemma_.lower()
            
            # 3. Stemming seletivo para Verbos (VERB/AUX) e Advérbios (ADV)
            if token.pos_ in ["VERB", "AUX", "ADV"]:
                token_processado = stemmer_pt.stem(lema)
            else:
                # Substantivos (NOUN, PROPN), adjetivos (ADJ), etc., ficam no lema
                token_processado = lema

            # 4. Filtro de comprimento mínimo
            if len(token_processado) >= 2:
                tokens_final.append(token_processado)
                
        return tokens_final