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

    def filter_special_characters(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_special_characters, change_for=change_for)
        return self

    def filter_spaces(self, change_for = " ") -> Self:
        self._add_step(TextUtils.filter_spaces, change_for=change_for)
        return self

    def filter_numbers(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_numbers, change_for=change_for)
        return self

    def filter_links(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_links, change_for=change_for)
        return self

    def filter_email(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_email, change_for=change_for)
        return self

    def filter_cnpj(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_cnpj, change_for=change_for)
        return self

    def filter_cpf(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_cpf, change_for=change_for)
        return self

    def filter_rg(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_rg, change_for=change_for)
        return self

    def filter_cep(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_cep, change_for=change_for)
        return self

    def filter_oab(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_oab, change_for=change_for)
        return self

    def filter_telefone(self, change_for = "") -> Self:
        self._add_step(TextUtils.filter_telefone, change_for=change_for)
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
    
    def lower(self) -> Self:
        self._add_step(str.lower)
        return self

    # Ending Trigger
    def as_tokens(self, txt: str) -> list[str]:
        return TextUtils.tokenize(self._process(txt))

    def as_str(self, txt: str) -> str:
        return self._process(txt)

    # Internal Utils
    def _process(self, txt: str) -> str:
        for proc, kwargs in self._steps:
            txt = proc(txt, **kwargs)
        return txt

    def _add_step(self, fn: Callable, **kwargs):
        self._steps.append((fn, kwargs))

if __name__ == "__main__":
    documento_sujo = """
    <p>O réu estava <b>CORRENDO</b> risco de vida na comarca de Goiânia.</p>
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
    # Saída provável: ['réu', 'correr', 'risco', 'vida', 'comarca', 'Goiânia', 'acesse', 'processo', 'enviar']