'''
Created on Sun Jul 21 09:54:07 2024

@authors:
    Antonio Pires
    Milton Ávila
    João Gabriel
    Wesley Oliveira

@license:
Este projeto está licenciado sob a Licença Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Você pode compartilhar, adaptar e construir sobre o material, desde que atribua crédito apropriado, não use o material para fins comerciais e distribua suas contribuições sob a mesma licença.
Para mais informações, consulte o arquivo [LICENSE](./LICENSE).
'''
from nltk.cluster.util import cosine_distance
import nltk

from .text_utils import TextUtils

class Berna:
    def __init__(self, doc1: str, doc2: str) -> None:
        if not isinstance(doc1, str) or not isinstance(doc2, str) or not doc1 or not doc2:
            raise ValueError("Sentenças inválidas ou vazias.")

        self.vec_terms1 = TextUtils.tokenize(doc1)
        self.vec_terms2 = TextUtils.tokenize(doc2)
        self._jaccard = None
        self._cosseno = None

    @property
    def jaccard(self) -> float:
        if self._jaccard is None:
            set1, set2 = set(self.vec_terms1), set(self.vec_terms2)
            
            union_terms = len(set1 | set2)
            intersection_terms = len(set1 & set2)

            if union_terms == 0: return 0.0 # Evita divisão por zero
            return round((intersection_terms / union_terms) * 100, 4)
        return self._jaccard

    @property
    def cosseno(self) -> float:
        if self._cosseno is None:
            union_terms = list(set(self.vec_terms1) | set(self.vec_terms2))

            l1 = [0] * len(union_terms)
            l2 = [0] * len(union_terms)

            for w in self.vec_terms1:
                l1[union_terms.index(w)] += 1
            for w in self.vec_terms2:
                l2[union_terms.index(w)] += 1

            return round((1 - cosine_distance(l1, l2)) * 100, 4)
        return self._cosseno