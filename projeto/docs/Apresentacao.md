![cover](images/logo.png)    ![cover](images/ds.svg) 
# AceleraDev Data Science: Projeto Prático

## Table of contents
1. [Introdução](#intro)
2. [Sistema de recomendação? Como? (Prólogo)](#Prólogo)
3. [EDA: *hands-on!*](#handson)
4. [Experimento A](#EA)
5. [Experimento B](#EB)
6. [Experimento C](#EC)
7. [Experimento D](#ED)
8. [Experimento E](#EE)
9. [Experimento F](#EF)
10. ["Enlatando" o código](#lata)


# 1. Introdução <a name="intro"></a>

## Objetivo <a name="Objetivo"></a>
O objetivo deste produto é fornecer um serviço automatizado que recomenda leads para um usuário dado sua atual lista de clientes (Portfólio).

## Contextualização 
Algumas empresas gostariam de saber quem são as demais empresas em um determinado mercado (população) que tem maior probabilidade se tornarem seus próximos clientes. Ou seja, a sua solução deve encontrar no mercado quem são os leads mais aderentes dado as características dos clientes presentes no portfólio do usuário.

Mais informações: [Instruções do Projeto](Instrucoes_Projeto.md)

### Autor: Israel Oliveira
![cover](images/eu.png)

Resumo profissional:

- Téc. em informática (jun/2004).

2002-2007: Técnico focado em Linux, serviços e segurança de internet e automação de servidores (shell script).

- Física Licenciatura (jan/2011).

2007-2014: Professor de matemática, física, programação (e afins), ensinos fundamental, médio e técnico.

- M.Sc Eng. Elétrica (abr/2016) e Dr. Eng. Elétrica incompleto.

2014-2018: Pesquisa em controle e automação, robótica, e reconhecimento de padrões.

- Cientista de dados.

De 2018 até o momento: atividades de ciência de dados, engenharia de ML/AI, prototipagem, desenvolvimento de software e DevOps. Foco em ciência de dados e eng. de ML/AI.


# 2. Sistema de recomendação? Como? (Prólogo) <a name="Prólogo"></a>

### O que eu tenho?

Revisei o capítulo 16, [*Recommender Systems*](http://d2l.ai/chapter_recommender-systems/index.html) do livro online [*Dive into Deep Learning*](http://d2l.ai/index.html), para relembrar e aprender novas possibilidades. Tive experiências prévias com modelagens baseadas em [*TF-IDF*](https://pt.wikipedia.org/wiki/Tf%E2%80%93idf),  [*Factorization Machines* (Rendle, 2010)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) e um método proprietário (patente requerida) baseado em [grafo](http://dataexperience.com.br/computacao-e-teoria-de-grafos-qual-a-relacao/) e [*Word2Vec*](https://en.wikipedia.org/wiki/Word2vec). Das minhas seis patentes requeridas, 4 são de sistemas de recomendação. Uma delas é sobre um método de indexação inteligente focado no uso do [*Elasticsearch*](https://pt.wikipedia.org/wiki/Elasticsearch).

### Primeira ideia!

Pensei em utilizar uma representação matricial, semelhante aos métodos baseados em fatoração matricial (FM). De fato, é uma das metodologias mais utilizadas nesse tipo de problema, baixa complexidade de treino e uso e é matematicamente robusta, principalmente com dados esparsos. Outro fator motivador foi ter uma experiência prévia com a biblioteca [*Surprise*](http://surpriselib.com/), que é facada em sistemas de recomendação. A estrutura da biblioteca me parece muito bem organizada e flexível. Desenvolvi um sistema de recomendação *user-user* estendendo uma das classes dessa biblioteca.

# 3. EDA: *hands-on!* <a name="handson"></a>

Notebook: [First_View](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/First_View.ipynb).

#### Principais observações:
- Alto número de dados faltantes (*missing data*) e "razoavelmente" distribuídos: quase toda as colunas possuem dados faltantes e a maioria em grandes quantidades. Um dataset **esparso**.
- Variado em formatos. Seria um problema em um *encoding* automatizado?
- Com os portfólios foi possível obter as correlações. Mas o qual era a confiança no uso desses valores de correlação para uma seleção de colunas (*feature selection*)?

#### Conclusões:
- Iniciar os experimentos com uma engenharia de *feature* objetivando um *encoding* aitomatizado.
- Formatar para uso da biblioteca *Surprise*.
- Metodologia de recomendação baseado uma decomposição [*SVD*](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) para gerar os fatores *user* (espaço matemático dos usuários).
- Usar similaridade vetorial para recomendação.

# 4. Experimento A. <a name="EA"></a>

#### Revisão da biblioteca *Surprise*:

Notebook: [Surprise_Review](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/Surprise_Review.ipynb).

- Revisei a documentação e como customizar um *dataset*.
- Aproveitei os métodos auxiliares (*helpers*) para *dataset*, treino e cálculo de métricas.

#### Engenharia de *feature*:

Notebook: [Experiment_A](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/Experiment_A.ipynb)

- Iniciei com apenas 20 colunas das 167, pegando as de maior correlação absoluta com os portfólios.
- *Encoding* de categóricos pra inteiros, de `1` até quantidade máxima de elementos únicos.
- Todo valor faltante foi convertido para `0`.
- Todas as colunas (já em formato numérico) normalizadas, escaladas entre `[0, 100]` e convertidas para `uint8` (*unsigned int 8 bits*: números inteiros de 8 bits). 
- Como o valor máximo era `100` e inteiro, usei a representação mais econômica.

#### Experimento com o módulo `SVD`:

- O tempo de treino ficou em média $405 s = 6,75 \text{min}$. E isso já não me pareceu bom, visto que estava usando menos de $20\%$ das colunas.
- RMSE em torno de $18$, visto que a escola de *rating* era 100. Data as circunstâncias, não me pareceu muito ruim. Todavia, não era uma maravilha!

# 5. Experimento B. <a name="EB"></a>

[Experiment_B](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/Experiment_B.ipynb)

- Revisão do processamento do *dataset* para uso no treino.
- Extensão da classe `SVD` adicionando o cálculo das distâncias e ordenamento pelos mais próximos.
- Uso da função de busca por parâmetros de treino `GridSearchCV`.
- Foi possível um RMSE em torno de $9$.
- Usando ainda apenas 20 das 167 colunas.

# 6. Experimento C. <a name="EC"></a>

[Experiment_C](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/Experiment_C.ipynb)

- Uso dos portfólios para o passo de teste das recomendações. Que é diferente do teste de estimativa de *rating* usado para cálculo do RMSE da função de validação cruzada da biblioteca.
- Escrevi as funções de *encoder* e *decoder*, usando os recursos da biblioteca. Facilitou muito!
- Escrevi a heurística de recomendação: ordena pelos vizinhos mais próximos dos *leads* (*user*) de entrada e desempata por votação.
- Usei um recurso para manter o valor das distâncias já calculadas, o que reduzia significativamente o tempo necessário. Todavia, em experimentos futuros, o consumo de memória era muito alto. É possível contornar esse problema usando uma representação esparsa eficiente de inteiros, como a [*Dictionary Of Keys based sparse matrix*](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html), a qual cheguei a usar mas não para isso.

# 7. Experimento D. <a name="ED"></a>

Notebooks: [Pythran/Basic_Tutorial](https://github.com/ysraell/examples/blob/master/Pythran/Basic_Tutorial.ipynb), [Experiment_D](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/Experiment_D.ipynb) e [Experiment_D2](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/Experiment_D2.ipynb)

- Apliquei melhorias no processamento do *dataset*, mas continuava usando poucas colunas.
- Usei [*Pythran*](https://pythran.readthedocs.io/en/latest/) e [*JAX*](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) para reduzir o tempo de cálculo das distâncias, da fatoração e outras possíveis.
- O passo de treino continua lento, ficando complicado de uma busca por parÂmetros.
- Resolvi que deveria abandonar a biblioteca *Surprise* e escrever o módulo de treino.
- Num primeiro momento houve um certo ganho de performance, principalmente no passo de treino.
- E os primeiros resultados de acurácia de recomendação `1-1` foi de 0 para o portfólio 1, $22%$ para o 2 e $23%$ para o 3. Nisso já comecei a ficar desconfiado desse primeiro portfólio. Conversando com outros colegas, eles também relataram que esse portfólio estava com métricas bem ruins e o 2 e o 3 estavam boas.

# 8. Experimento E. <a name="EE"></a>

[Experiment_E](https://github.com/ysraell/aceleradev_private/blob/master/projeto/EDA/Experiment_E.ipynb)

- Comecei a usar todo o *dataset*, todas as colunas.
- Consolidei o processamento dos dados pré treino, agora robusto e normalizado. Excluindo apenas as colunas que não apresentavam informação alguma, reduzindo do total de 182 colunas para 167.
- Consolidei o algoritmo e seus métodos.
- Escrevi diferentes funções para cálculo de distância: Manhattan, Camberra, Bray Curtis e Cosseno.
- Funções para transformar os dados (já encodados e normalizados) de forma otimizada: 1) considerando valores faltantes, 2) pelo desvio padrão e 3) pela entropia.
- Implementei o uso das funções PCA, FastICA, FactorAnalysis, IncrementalPCA, TruncatedSVD e NMF, todas do módulo `decomposition` do [*scikit-learn*](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
- Fiz experimentos de performance (tempo e consumo de memória) com todas as funções para melhor planejar o experimento F, no qual faria a busca por parâmetros.

