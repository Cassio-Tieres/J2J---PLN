from utils.tokenizador import Tokenizador
from utils.dataset import Seq2SeqDS

tokens = Tokenizador()
tokens.cria_vocab("data/treino.jsonl")
tokens.salva_vocab("data/vocab.json")
tokens.carrega_vocab("data/vocab.json")

dataset = Seq2SeqDS("data/treino.jsonl", tokens, max_len=100)