###################################################
# ExMatrix app.
# Israel Oliveira.
#
# Módulo para carregamento e uso do modelo.
#
###################################################


from utils import *
from model import *
import pickle
from loguru import logger


class Recommender:
    def __init__(self, model_path: Path = "../model/", tag: str = "default"):
        """
            Classe para carregar e usar o modelo.

            Args:
            -----
                model_path: (Path) Local onde será salvo o modelo (Pickle)
                tag: (str) Tag para identificar o modelo salvo.

            Todo:
            -----
                1) Adicionar método para troca da função de cálculo de distância.
                2) Adicionar função para troca dos valores dos parâmetros.
        """
        self.model_file = model_path + "model_{}.pkl".format(tag)
        self.model = pickle.load(open(self.model_file, "rb"))
        logger.info("Modelo carregado")
        self.L = 3
        self.Fk = 1
        self.limit = 10
        self.stateless = True

    def For(self, in_list: List[Raw], k: int = 1):
        """
            Extensão da função de recomendações.
            
            Args:
            -----
                in_list: (List[Raw]) Lista de IDs (de 1 a N).
                k: (int) Retorna k recomendações.

            Returns:
            --------
                (List[Raw]) Lista de IDs recomendados.

            Notes:
            ------
                Os demais parâmetros estão nessa classe como atributos.
        """
        self.model.stateles = self.stateless
        return self.model.recomender(in_list, k=k, L=self.L, Fk=self.Fk, limit=self.limit)


# EOF
