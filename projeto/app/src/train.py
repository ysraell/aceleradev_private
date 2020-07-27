#!/usr/bin/env python3


###################################################
# ExMatrix app.
# Israel Oliveira.
#
# Módulo/programa para treino e geração do modelo.
#
###################################################

from utils import *
from model import *
import pickle
from loguru import logger
import sys


def train(
    train_path: Path = "../data/", model_path: Path = "../model/", tag: str = "default"
):
    """
        Módulo para treino e salvamento do modelo.

        Args:
        -----
            train_path: (Path) Local onde estão os CSVs.
            model_path: (Path) Local onde será salvo o modelo (Pickle)
            tag: (str) Tag para identificar o modelo salvo.
    """
    try:
        model = ExMatrix()
        df_train, _ = load_dataset(path_data=train_path, test_list=None)
        ds, score = feat_proc(df_train)
        logger.info("Treinando e modelo...")
        model.fit(ds, score)
        logger.info("...pronto.")
        model_file = model_path + "model_{}.pkl".format(tag)
        pickle.dump(
            model, open(model_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL,
        )
        logger.info('Model salvo: "{}"..'.format(model_file))
    except Exception as e:
        # Printing this causes the exception to be in the training job logs, as well.
        logger.error("Erro durante o treino: " + str(e))

        # A non-zero exit code causes the training job to be marked as Failed.
        exitStatus = 255

        sys.exit(exitStatus)


if __name__ == "__main__":

    train()

# EOF
