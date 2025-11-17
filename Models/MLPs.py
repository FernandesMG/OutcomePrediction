import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Tabular encoder (continuous-only version). Add cat embeddings if needed.
class TabMLP(nn.Module):
    def __init__(self, in_dim, hid=(64, 32), p=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hid:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x_tab):
        return self.net(x_tab.float())


# ---------- Mixed tabular encoder (continuous + categorical) ----------
class TabEncoder(nn.Module):
    """
    x_cont: [B, num_cont] float (already standardized if possible)
    x_cat:  [B, num_cat]  long  (0..card_i; where 0 = UNK/other)
    """
    def __init__(self, num_cont, cat_cardinalities, emb_dropout=0.1,
                 cont_mlp_hidden=(32,), out_hidden=(64, 32)):
        super().__init__()
        self.num_cont = num_cont
        self.num_cat  = len(cat_cardinalities)

        # per-feature embedding dims
        def emb_dim(card):
            return int(min(32, max(4, round(1.6 * (card ** 0.56)))))

        self.embs = nn.ModuleList([
            nn.Embedding(num_embeddings=card + 1,  # +1 for UNK id=0
                         embedding_dim=emb_dim(card),
                         padding_idx=0)
            for card in cat_cardinalities
        ])
        self.emb_dropout = nn.Dropout(emb_dropout)
        cat_dim = sum(e.embedding_dim for e in self.embs)

        # small MLP for continuous features (LayerNorm is per-sample; safe for tiny batches)
        cont_layers, d = [], num_cont
        for h in cont_mlp_hidden:
            cont_layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(0.05)]
            d = h
        self.cont = nn.Sequential(*cont_layers) if cont_layers else nn.Identity()
        cont_out = d if cont_layers else num_cont

        fused_in = cat_dim + cont_out

        # final projector to a compact tabular embedding
        layers, d = [], fused_in
        for h in out_hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(0.1)]
            d = h
        self.proj = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x_cont, x_cat):
        # cat: embed each column and concat
        if self.num_cat > 0:
            cat_vecs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
            cat = torch.cat(cat_vecs, dim=1)
            cat = self.emb_dropout(cat)
        else:
            cat = None

        cont = self.cont(x_cont.float()) if self.num_cont > 0 else None

        if cat is None:
            fused = cont
        elif cont is None:
            fused = cat
        else:
            fused = torch.cat([cont, cat], dim=1)

        return self.proj(fused)
