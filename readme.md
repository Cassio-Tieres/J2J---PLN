# J2J PLN
Projeto de Processamento de Linguagem Natural (PLN) desenvolvido para reduzir custo de consumo de tokens de modelos pré-treinados em conversões de JSON para JSON.

# Técnicas implementadas
* Tokenização
* Seq2Seq (modelo sequence-to-sequence)
* Rede Neural Artificial (RNA)
* Treinamento com pares (entrada -> saída)

# Problema
Em nossos projetos de Inteligência Artificial integrando modelos de LLM pré-treinados, como GPT, sempre precisamos reforçar a saída de JSONs rigidamente estruturados para conseguirmos manipular respostas de um agente ou assistente de IA.

Inicialmente, nós fazíamos toda essa manipulação através do prompt do assistant na plataforma do modelo que utilizamos, porém, fazer todo o tratamento de dados dentro de um modelo pré-treinado exige um custo, visto que o consumo de tokens é o principal gerador de receita das plataformas.

# Estrutura de pastas
````
|_dados/
    |_ treino.jsonl # pares de entrada e saída
    |_ vocab.json # vocabulário de tokens gerados a partir dos dados
|_ modelo/
    |_ encoder.py # classe do codificador da entrada
    |_ decoder.py # classe do decodificador da saída
    |_ seq2seq.py # classe de integração do codificador e do decodificador
|_ treino.py # script de treinamento do modelo
|_ evolucao.py # script de evolução do modelo com dados novos
|_ utils/
    |_ tokenizador.py # tokenizador e construtor de vocabulário
    |_ dataset.py # Prepação e carregamento de dados
|_ config.py # Configurações gerais e hiperparametros
````

# Solução
Para solucionar o problema de consumo de tokens na manipulação da resposta das LLMs, quando se trata de respostas geradas em JSON, foi projetado o J2J, um modelo de Processamento de Linguagem Natural, que utiliza embbedings, encoder e decoder para analisar entradas em JSONs e gerar JSONs estruturados da maneira que precisarmos.