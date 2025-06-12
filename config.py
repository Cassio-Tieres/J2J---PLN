from utils.tokenizador import Tokenizador

tokens = Tokenizador()
tokens.cria_vocab("data/treino.jsonl")
tokens.salva_vocab("data/vocab.json")