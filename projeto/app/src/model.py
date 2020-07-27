#############################################
# ExMatrix app.
# Israel Oliveira.
#
# Módulo com o modelo base da aplicação.
#
#############################################


import pandas as pd
from typing import List, NewType
from utils import *
from loguru import logger
from collections import Counter, defaultdict
from copy import deepcopy

Uid = NewType("uid", int)
Raw = NewType("raw", str)


class ExMatrix:
    def __init__(
        self,
        process_values=escalaropt_entropy,
        factorize=f_NMF,
        vector_distance=BrayCurtis,
        stateless: bool = True,
    ):
        """
            Modelo base da aplicação.

            Encoder: transforma os IDs, nas duas vias.
            Recommender: recomenda IDs baseado nos IDs de entrada.

            Args:
            -----
                process_values: função para processar a entrada ProcDS.
                factorize: função para reduzir a dimensionalidade ou
                        apenas transformar a entrada ProcDS
                vector_distance: função para calcular a distância entre os IDs.
                stateless: (bool) para definir se as distâncias calculadas
                        devem ser salvas.

            Notes:
            ------
                stateless: Em experimentos prévios, foi percebido um consumo de
                        memória alto. O limite é o quadrado da quantidade de
                        IDs no dataset.
        """
        self.matrix_dict = {}
        self.stateless = stateless
        self.M = None
        self.pu = None
        self.raw = None
        self.uid = None
        self.vector_distance = vector_distance
        self.factorize = factorize
        self.process_values = process_values

    def fit(self, dataset: TrainDS, score: ScoreColumns):
        """
            Gera os dicionários de encoder.
            Gera a matrix dos IDs.
        """
        self.raw = dataset['id'].to_dict()
        self.uid = {raw: uid for uid, raw in self.raw.items()}
        self.all_raw = dataset['id'].tolist()
        self.all_uid = dataset.index
        logger.info("Processando valores.")
        dataset = dataset.drop(columns=['id'])
        dataset = self.process_values(dataset, score)
        ds_size = dataset.values.shape[0]
        logger.info("Fatorizando.")
        self.M = self.factorize(dataset.values)
        if ds_size != self.M.shape[0]:
            raise ValueError("A fatoração não está correta!")
        del dataset
        logger.info("Matriz pronta.")

    def _get_neighbors(
        self, uid: Uid, k: int = 1, black_list: List[Uid] = []
    ) -> List[Uid]:
        """
            Calcula todas as distâncias entre 'uid' de entrada e todos os outros 'uid'.
            A distância calculada é armazenda e não calculada novamente quando
            `stateless = False`.
        """
        k = k if k >= 0 else 0
        if uid not in self.matrix_dict.keys():
            self.matrix_dict[uid] = self.vector_distance(self.M, self.M[uid])
        out = [
            x[0]
            for x in sorted(
                [
                    (uid2, self.matrix_dict[uid][uid2])
                    for uid2 in self.all_uid
                    if (uid2 not in black_list)
                ],
                key=lambda x: x[1],
            )
        ][:k]
        if self.stateless:
            del self.matrix_dict
            self.matrix_dict = {}
        return out

    def _uid2raw(self, uid: Uid) -> str:
        """
            uid -> raw.
            Valor interno para externo, o nome original do usuário.
        """
        return self.raw[uid]

    def _raw2uid(self, raw: Raw) -> int:
        """
            raw -> uid.
            Valor externo para interno, o id interno do usuários..
        """
        return self.uid[raw]

    def recomender(
        self, in_list: List[Raw], k: int = 1, L: int = 3, Fk: int = 1, limit: int = 10
    ) -> List[Raw]:
        """
            Faz as recomendacoes baseada nas seguintes heurísticas:
            1) Calcula os vizinhos próximos de cada ID de entrada.
            2) Aplica uma ordenação por distância.
            3) Desempada por votação.

            Args:
            -----
                in_list: (List[Raw]) Lista de IDs (de 1 a N).
                k: (int) Retorna k recomendações.
                L: (int) Aumenta o pedido de vizinhos, devido a
                   possibilidade de blacklist.
                Fk: (int) Expande a lista de recomendações,
                    pode ajudar nas heurísticas 2 e 3.
                limit: (int) Limita a quantidade loops na obtenção
                       dos vizinhos. 

            Returns:
            --------
                (List[Raw]) Lista de IDs recomendados.
        """
        # Pega quantas recomendações por usuário em `in_list`,
        # mas sem deixar faltar
        N_in = len(in_list)
        k = k if k > 0 else 1
        R_per_in = L * (k // N_in + min(k % N_in, 1))

        # Pega os `uid`
        uid_in_list = [self._raw2uid(raw) for raw in in_list]

        # Pega os vizinhos mais próximos de cada uid de entrada.
        done = False
        Rounds = 0
        while limit and (not done):
            Rounds += 1
            # Ele sempre pega todos novamente.
            recomendations_list = [
                self._get_neighbors(uid, R_per_in, uid_in_list) for uid in uid_in_list
            ]
            # Quando limit = 0, encerra.
            limit -= 1
            # Quando tem gente o suficiente, encerra.
            if len(set(flat(recomendations_list))) >= Fk * k:
                done = True
            # Depois do primeiro loop, pega um a mais.
            R_per_in += 1

        # Aqui gera um dicionário ordenando por votacao.
        count_rec = Counter(flat(recomendations_list))  # A votação!!
        count_rec = list(count_rec.items())
        ct_pos = defaultdict(list)
        while count_rec:
            tmp = count_rec.pop(0)
            ct_pos[tmp[1]].append(tmp[0])

        # Aqui considera a posição de vizinhos mais proximos.
        nn_pos_inv = defaultdict(list)
        tmp = deepcopy(recomendations_list)
        while tmp:
            tmp2 = tmp.pop(0)
            n = 0
            while tmp2:
                n += 1
                tmp3 = tmp2.pop(0)
                nn_pos_inv[tmp3].append(n)

        # Vai separando por votação e ordem de proximidade como desempate.
        votos_list = list(ct_pos.keys())
        out_uid = []
        while votos_list and k:
            votos = max(votos_list)
            votos_list.remove(votos)
            tmp = sorted(
                [(tmp, min(nn_pos_inv[tmp])) for tmp in ct_pos[votos]], key=lambda x: x[1]
            )
            while tmp and k:
                out_uid.append(tmp.pop(0)[0])
                k -= 1

        # converte para Raw e "joga fora".
        return [self._uid2raw(uid) for uid in out_uid]


# EOF
