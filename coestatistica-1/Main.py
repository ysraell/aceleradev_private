########
# Israel Oliveira - 22/06/2020

import pandas as pd
import json


df = pd.read_csv("desafio1.csv")

with open("submission.json", "r") as f:
    submission = json.load(f)

def moda(df):
    return int(df.mode()[0])


def median(df):
    return int(df.median())


def mean(df):
    return df.mean()


def std(df):
    return df.std()


operator = {"moda": moda, "mediana": median, "media": mean, "desvio_padrao": std}

target = "pontuacao_credito"
col_filter = "estado_residencia"

for estado in submission.keys():
    vals = df.loc[df[col_filter] == estado][target]
    for operation in operator.keys():
        submission[estado][operation] = operator[operation](vals)

with open("submission.json", "w") as f:
    json.dump(submission, f)

#EOF