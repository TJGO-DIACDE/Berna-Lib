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
from typing import Self, Callable

from .text_utils import TextUtils

# Linked Methods, Static
class ProcessLinked:
    def __init__(self) -> Self:
        self._steps: list[tuple[Callable, dict[str, any]]] = []

    def filter_special_characters(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_special_characters, change_to=change_to)
        return self

    def filter_spaces(self, change_to = " ") -> Self:
        self._add_step(TextUtils.filter_spaces, change_to=change_to)
        return self
    
    def lower(self) -> Self:
        self._add_step(str.lower)
        return self
    
    def normalizar_hibrido(self, core = "pt_core_news_sm", stem_language = "portuguese") -> Self:
        self._add_step(TextUtils.normalizacao_hibrida, core=core, stem_language=stem_language)
        return self

    def filter_numbers(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_numbers, change_to=change_to)
        return self

    def filter_links(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_links, change_to=change_to)
        return self

    def filter_email(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_email, change_to=change_to)
        return self

    def filter_cnpj(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_cnpj, change_to=change_to)
        return self

    def filter_cpf(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_cpf, change_to=change_to)
        return self

    def filter_rg(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_rg, change_to=change_to)
        return self

    def filter_cep(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_cep, change_to=change_to)
        return self

    def filter_oab(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_oab, change_to=change_to)
        return self

    def filter_telefone(self, change_to = "") -> Self:
        self._add_step(TextUtils.filter_telefone, change_to=change_to)
        return self

    def remove_stopwords(self, language = "portuguese") -> Self:
        self._add_step(TextUtils.remove_stopwords, language=language)
        return self

    def remove_html(self) -> Self:
        self._add_step(TextUtils.remove_html)
        return self

    def lemmatize(self, core = "pt_core_news_sm") -> Self:
        self._add_step(TextUtils.lemmatize, core=core)
        return self

    def stemming(self) -> Self:
        self._add_step(TextUtils.stemming)
        return self

    # Ending Trigger
    def as_tokens(self, txt: str) -> list[str]:
        resultado = self._process(txt)
        if isinstance(resultado, list):
            return resultado
        return TextUtils.tokenize(resultado)

    def as_str(self, txt: str) -> str:
        resultado = self._process(txt)
        if isinstance(resultado, list):
            return " ".join(resultado)
        return resultado

    # Internal Utils
    def _process(self, txt: str) -> str:
        for proc, kwargs in self._steps:
            txt = proc(txt, **kwargs)
        return txt

    def _add_step(self, fn: Callable, **kwargs):
        self._steps.append((fn, kwargs))

if __name__ == "__main__":
    documento_sujo = """
    <p>O réu não trabalhou, agindo rapidamente de forma prejudicial.</p>
    Marcador_Pagina_12
    """

    pipeline_juridico = (
        ProcessLinked()
        .remove_html()               # Passo 1: HTML
        .filter_spaces()             # Passo 1: Espaços e caracteres de controle
        .lower()                     # Passo 2: Lowercasing
        .normalizar_hibrido()        # Passos 3, 4, 5 e 6 combinados (Alta performance)
    )

    tokens_processados = pipeline_juridico.as_tokens(documento_sujo)
    print(tokens_processados)