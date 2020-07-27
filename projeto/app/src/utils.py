#############################################
# ExMatrix app.
# Israel Oliveira.
#
# Módulo com funções úteis para a aplicação.
#
#############################################


import functools
import operator
import pandas as pd
import numpy as np
from typing import NewType, List, Dict
from loguru import logger
from sklearn.decomposition import NMF

Path = NewType("Path", str)
TrainDS = NewType("TrainDS", pd.DataFrame)
ScoreColumns = NewType("ScoreColumns", Dict[str, float])
TestDS = NewType("TestDS", pd.DataFrame)
ProcDS = NewType("ProcDS", pd.DataFrame)
Matrix = NewType("Matrix", np.ndarray)
Vector = NewType("Vector", np.ndarray)


def flat(a: List[List[list]]) -> List[list]:
    """
        Para achatar em um nível uma lista de listas.

    Arg:
    ----
        a: Uma lista com listas

    Return:
    -------
        Uma lista com um nível de lista a mesmo.

    Notes:
    ------
        Se input for lista com strings, ele soletra a string =D
    
    """
    return functools.reduce(operator.iconcat, a, [])


def normalize(x: Vector) -> Vector:
    """
        Normaliza um vetor.
    """
    return (
        (x - np.min(x)) / (np.max(x) - np.min(x))
        if (np.max(x) - np.min(x)) > 0
        else (x - np.min(x))
    )


def load_dataset(
    path_data: Path = "../data/",
    train_list: List[int] = [0],
    test_list: List[int] = [0, 1, 2],
    train_test_merged: bool = False,
) -> (TrainDS, TestDS):
    """
        Carrega o dataset.

    Arg:
    ----
        path_data: local onde estão os CSVs.
        train_list: a numeração, sendo train_0.csv, train_1.csv... 
        test_list: a numeração, sendo test_0.csv, test_1.csv...
        train_test_merged: Faz o dataset de treino ter apenas os IDs presentes nos dados de teste.

    Return:
    -------

        Tuple(ds_train, ds_test): os datasets.
    
    """
    df_train = None
    if train_list:
        logger.info("Carregando dataset de treino...")
        df_train = (
            pd.concat(
                [pd.read_csv(path_data + "train_{}.csv".format(i)) for i in train_list]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        df_train = df_train.loc[:, ~df_train.columns.str.contains("^Unnamed")]
        logger.info("...pronto!")

    df_test = None
    if test_list:
        logger.info("Carregando dataset de teste...")
        df_ep_list = [pd.read_csv(path_data + "test_{}.csv".format(i)) for i in test_list]
        tmp = []
        for i in range(len(df_ep_list)):
            df_ep_list[i]["P"] = i + 1
            tmp.append(df_ep_list[i][["id", "P"]])
        df_test = pd.concat(tmp).drop_duplicates().reset_index(drop=True)
        del df_ep_list
        del tmp
        logger.info("...pronto!")
        if train_test_merged and train_list:
            df_train = (
                df_train.merge(df_test, on="id")
                .drop(columns=["P"])
                .reset_index(drop=True)
            )
    return df_train, df_test


def feat_proc(dataset: TrainDS) -> (ProcDS, ScoreColumns):
    """
        Aplica uma transformação nas colunas de formas agnóstica para uma forma numérica.

    Arg:
    ----
        dataset: Dados no formato TrainDS.

    Return:
    -------
        ProcDS: Dados transformados, todas as colunas numéricas.
        ScoreColumns: dicionário com as contagens de missing values.
    
    """
    logger.info("Processando as features...")
    missing_count = {}
    remove_cols = []
    feat_cols = list(dataset.columns)
    feat_cols.remove("id")
    for col in feat_cols:
        try:
            missing_count[col] = sum(dataset[col].isna()) / dataset[col].nunique()
            dataset[col] = dataset[col].fillna(0) * 1
        except ZeroDivisionError:
            remove_cols.append(col)

    feat_cols = [col for col in feat_cols if col not in remove_cols]

    def normalize(x):
        return (
            (x - np.min(x)) / (np.max(x) - np.min(x))
            if (np.max(x) - np.min(x)) > 0
            else (x - np.min(x))
        )

    for col in feat_cols:
        try:
            dataset[col] = normalize(dataset[col].tolist())
        except:
            maping = {val: i + 1 for i, val in enumerate(dataset[col].unique())}
            dataset[col] = dataset[col].apply(lambda x: maping[x])
            dataset[col] = normalize(dataset[col].tolist())

    remove_cols = []
    for col in feat_cols:
        if dataset[col].nunique() == 1:
            remove_cols.append(col)
    feat_cols = [col for col in feat_cols if col not in remove_cols]
    missing_count = {key: val for key, val in missing_count.items() if key in feat_cols}
    logger.info("...pronto!")
    return dataset[["id"] + feat_cols], missing_count


def escalaropt_entropy(df: ProcDS, score: ScoreColumns) -> ProcDS:
    """
        Aplica uma transformação nas colunas já numérica.
        Essa transformação é baseada na entropia dos dados,
        escalando os valores normalizados.

    Arg:
    ----
        ProcDS: Dados no formato ProcDS normalizados nas colunas.
        ScoreColumns: dicionário com as contagens de missing values.

    Return:
    -------
        ProcDS: Dados no formato ProcDS rescalados nas colunas.
    
    """
    df_score = pd.DataFrame(score.items(), columns=["col", "score"])
    df_score["escala_opt"] = normalize(
        [(-sum((df[col] + 1) * np.log(df[col] + 1))) for col in df_score["col"]]
    )
    df_score["escala_opt"] = df_score["escala_opt"].apply(lambda x: max(x, 0.1))
    df_tmp = pd.DataFrame()
    for _, row in df_score.iterrows():
        df_tmp[row.col] = row.escala_opt * df[row.col]
    return df_tmp


def BrayCurtis(X: Matrix, vec: Vector) -> Vector:
    """
        Calcula a distância de Bray Curtis entre o vetor `vec`
        e cada vetor da matriz `X`.
    """
    return abs((X - vec)).sum(-1) / abs((X - vec)).sum(-1).sum(-1)


def f_NMF(M: Matrix, n_components: int = 62) -> Matrix:
    """
        Queremos apenas a transformação da matriz de entrada.
    """
    out = NMF(n_components=n_components)
    return out.fit_transform(M)


# EOF
