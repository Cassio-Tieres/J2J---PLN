import torch
import json
from torch.utils.data import Dataset
from pathlib import Path

class Seq2SeqDS(Dataset):
    def __init__(self, caminho, tokenizador, max_len=100):
        self.pares = []
        self.tokenizador = tokenizador
        self.max_len = max_len

        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                dado = json.loads(linha)
                entrada = tokenizador.tokenize(str(dado['input']))
                saida = tokenizador.tokenize(str(dado['output']))

                # tokens especiais
                entrada_ids = [tokenizador.token_to_id["<SOS>"]] + \
                              [tokenizador.token_to_id.get(t, tokenizador.token_to_id["<UNK>"]) for t in entrada] + \
                              [tokenizador.token_to_id["<EOS>"]]

                saida_ids = [tokenizador.token_to_id["<SOS>"]] + \
                            [tokenizador.token_to_id.get(t, tokenizador.token_to_id["<UNK>"]) for t in saida] + \
                            [tokenizador.token_to_id["<EOS>"]]
                
                entrada_ids = entrada_ids[:max_len] + [tokenizador.token_to_id["<PAD>"]] * (max_len - len(entrada_ids))
                saida_ids = saida_ids[:max_len] + [tokenizador.token_to_id["<PAD>"]] * (max_len - len(saida_ids))

                self.pares.append((entrada_ids, saida_ids))
    def __len__(self):
        return len(self.pares)
    
    def __getitem__(self, idx):
        entrada_ids, saida_ids = self.pares[idx]
        entrada_tensor = torch.tensor(entrada_ids, dtype=torch.long)
        saida_tensor = torch.tensor(saida_ids, dtype=torch.long)
        return entrada_tensor, saida_tensor