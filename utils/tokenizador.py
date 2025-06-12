import json
import re
from collections import Counter
from pathlib import Path

class Tokenizador:
    def __init__(self, vocab_min_freq=1):
        self.vocab_min_freq = vocab_min_freq
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
    
    def tokenize(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
        return tokens
    
    def cria_vocab(self, caminho):
        contador = Counter()

        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                try:
                    dado = json.loads(linha)
                except json.JSONDecodeError:
                    print(f"Linha com erro de JSON: {linha}")
                    continue
                input_tokens = self.tokenize(str(dado['input']))
                output_tokens = self.tokenize(str(dado['output']))
                contador.update(input_tokens + output_tokens)

        vocab_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        vocab_tokens += [token for token, freq in contador.items() if freq >= self.vocab_min_freq]
        
        self.token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab = self.token_to_id
    
    def salva_vocab(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=4)
    
    def carrega_vocab(self, caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.vocab = self.token_to_id
    
    def codificador(self, texto):
        tokens = self.tokenize(texto)
        ids = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
        return ids
    
    def decodificador(self, ids):
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        return " ".join(tokens)
