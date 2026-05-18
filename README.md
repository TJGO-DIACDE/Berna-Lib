# Berna TJGO DIACDE Lib

Biblioteca desenvolvida pelo **TJGO (Diretoria de Inteligência Artificial, Ciência de Dados e Estatística)**. Este pacote oferece ferramentas robustas para pré-processamento de texto e cálculo de similaridade textual, essenciais para processamento de linguagem natural (NLP) e preparação de dados para LLMs.

O núcleo das operações de texto é fornecido pela classe `TextUtils`, acessível através da seguinte interface:

### **Interface Fluente (`ProcessLinked`):** Para aplicações diretas e encadeadas.

Para cálculo de similaridade textual, a biblioteca disponibiliza a classe `Berna`.

> **Nota:** Este repositório público contém uma seleção de ferramentas usadas internamente pelo TJGO.

---


## Instalação
* **PyPI:** [berna-tjgo-diacde-lib](https://pypi.org/project/berna-tjgo-diacde-lib/)
Para instalar a biblioteca, utilize o gerenciador de pacotes `pip`:

```bash
pip install berna-tjgo-diacde-lib

```

### Dependências

O projeto utiliza as seguintes bibliotecas:

*   `spacy`
*   `nltk`
*   `beautifulsoup4`
*   `snowballstemmer`

**Recursos Necessários (Download Manual):**

Para o funcionamento correto da biblioteca, são necessários os seguintes recursos que devem ser baixados manualmente:

*   **NLTK:**
    *   `stopwords` (para remoção de stopwords):
        ```bash
        python -m nltk.downloader stopwords
        ```
    *   `punkt` (para tokenização):
        ```bash
        python -m nltk.downloader punkt
        ```
*   **spaCy:**
    *   Modelo de linguagem em português (ex: `pt_core_news_sm` para lematização):
        ```bash
        python -m spacy download pt_core_news_sm
        ```

É crucial garantir que estes recursos estejam instalados antes de utilizar as funcionalidades que dependem deles.

---

## Módulo de Pré-Processamento

### Interface Fluente com `ProcessLinked`

A classe `ProcessLinked` adota o padrão *Builder*, permitindo que você monte uma receita de pré-processamento personalizada e a execute sob demanda. Isso é ideal tanto para higienização de dados sensíveis (LGPD) quanto para a preparação de textos para LLMs.

#### Cenário A: Anonimização de Peças Jurídicas (Substituição de Dados)

Em vez de apenas deletar as informações, você pode usar o argumento `change_for` para mascarar dados sensíveis antes de enviar o texto para uma API externa ou modelo público.

```python
from berna_tjgo_diacde_lib import ProcessLinked

# 1. Define a receita de anonimização (o pipeline é reutilizável)
anonimizador = (
    ProcessLinked()
    .filter_cpf(change_for="[CPF_ANONIMIZADO]")
    .filter_cnpj(change_for="[CNPJ_ANONIMIZADO]")
    .filter_oab(change_for="[OAB_ANONIMIZADO]")
    .filter_telefone(change_for="[TEL_ANONIMIZADO]")
    .filter_spaces() # Garante espaçamento limpo após as trocas
)

# 2. Aplica a mesma receita em diferentes textos
peticao_1 = "O autor João, inscrito no CPF 123.456.789-00 e OAB/GO 00.000, liga no (62) 99999-9999."
peticao_2 = "A empresa X (CNPJ 12.345.678/0001-99) não respondeu aos chamados."

print(anonimizador.as_str(peticao_1))
# Saída: O autor João, inscrito no [CPF_ANONIMIZADO] e [OAB_ANONIMIZADO], liga no [TEL_ANONIMIZADO].

print(anonimizador.as_str(peticao_2))
# Saída: A empresa X ([CNPJ_ANONIMIZADO]) não respondeu aos chamados.

```

#### Cenário B: Pipeline para NLP Tradicional (Tokenização e Lematização)

Se o seu objetivo é alimentar modelos de Bag-of-Words, TF-IDF ou calcular similaridade pura, você pode transformar o texto diretamente em uma lista de tokens limpos usando o gatilho final `as_tokens()`.

```python
from berna_tjgo_diacde_lib import ProcessLinked

documento_sujo = """
<p>O réu estava <b>correndo</b> risco de vida na comarca de Goiânia.</p>
Acesse o processo em https://tjgo.jus.br ou envie e-mail para processual@tjgo.jus.br.
"""

# Monta o pipeline de limpeza profunda
pipeline_nlp = (
    ProcessLinked()
    .remove_html()               # Remove tags <p> e <b>
    .filter_links()              # Remove a URL do TJGO
    .filter_email()              # Remove o e-mail institucional
    .filter_special_characters() # Remove pontuações restantes
    .lemmatize()                 # Reduz palavras (ex: "correndo" -> "correr")
    .remove_stopwords()          # Remove "O", "estava", "de", "para", etc.
)

# Dispara o processamento convertendo direto para lista de tokens
tokens_limpos = pipeline_nlp.as_tokens(documento_sujo)

print(tokens_limpos)
# Saída provável: ['réu', 'correr', 'risco', 'vida', 'comarca', 'goiânia', 'acesse', 'processo', 'enviar', 'email]

```

### Métodos Disponíveis (via `TextUtils`)

| Método | Descrição |
| --- | --- |
| `filter_special_characters` | Remove pontuação e caracteres especiais. |
| `filter_spaces` | Padroniza múltiplos espaços para um único espaço. |
| `filter_numbers` | Remove caracteres numéricos. |
| `filter_links` | Remove URLs e links. |
| `filter_email` | Remove endereços de e-mail. |
| `filter_cnpj`/`cpf`/`rg` | Filtros específicos para documentos brasileiros. |
| `filter_cep`/`oab` | Filtros para códigos postais e registros da OAB. |
| `remove_stopwords` | Remove palavras comuns (ex: "e", "de", "o"). |
| `remove_html` | Remove tags HTML via BeautifulSoup. |
| `lemmatize` | Reduz palavras à forma base (lema) usando spaCy. |
| `stemming` | Reduz palavras à raiz usando snowballstemmer. |
| `tokenize` | Divide o texto em uma lista de tokens (palavras ou sentenças, dependendo da implementação interna do NLTK). |


---

## Módulo de Cálculo de Similaridade

Este módulo oferece funcionalidades para calcular a similaridade textual entre dois documentos utilizando diferentes métricas. A classe `Berna` é o ponto de entrada para estas operações, utilizando internamente o método `TextUtils.tokenize` para preparar o texto antes da comparação.

### Classe `Berna`

A classe `Berna` permite comparar duas strings e retorna o percentual de similaridade com base em métricas comuns:

*   **Similaridade de Cosseno (`cosseno`):** Mede o cosseno do ângulo entre dois vetores não nulos em um espaço multidimensional.
*   **Índice de Jaccard (`jaccard`):** Mede a similaridade entre dois conjuntos finitos, sendo definida como o tamanho da interseção dividido pelo tamanho da união dos conjuntos.

#### Exemplo de Uso:

```python
from berna_tjgo_diacde_lib import Berna

doc1 = "Este é o primeiro documento para teste de similaridade."
doc2 = "Este documento é o segundo para testar a similaridade textual."

similarity_calculator = Berna(doc1, doc2)

cosine_similarity = similarity_calculator.cosseno
jaccard_similarity = similarity_calculator.jaccard

print(f"Similaridade de Cosseno: {cosine_similarity:.2f}%")
print(f"Similaridade de Jaccard: {jaccard_similarity:.2f}%")

# Saída esperada (aproximada):
# Similaridade de Cosseno: XX.XX%
# Similaridade de Jaccard: YY.YY%
```

---

## Licença

Este projeto está licenciado sob a Licença Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Você pode compartilhar, adaptar e construir sobre o material, desde que atribua crédito apropriado, não use o material para fins comerciais e distribua suas contribuições sob a mesma licença.
Para mais informações, consulte o arquivo [LICENSE](./LICENSE).

---

## Créditos e Desenvolvedores

Desenvolvido pelo **Tribunal de Justiça do Estado de Goiás** - [Diretoria de Inteligência Artificial, Ciência de Dados e Estatística](https://github.com/TJGO-DIACDE).

**Time de Desenvolvimento:**

* [Antônio Pires](https://github.com/apcastrojr) - `<apcastro@tjgo.jus.br>`
* [Milton Ávila](https://github.com/Milton-Avila) - `<miltonavila.dev@gmail.com>`
* [João Gabriel](https://github.com/joaograndotto) - `<grandottojoao@gmail.com>`
* [Wesley Oliveira](https://github.com/waejl) - `<wesley@woliveira.me>`

---

*Contato oficial: [estatistica@tjgo.jus.br*]()