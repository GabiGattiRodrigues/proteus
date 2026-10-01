"""Proteu - gerador de base sintetica de clientes de cartao de credito.

A base NAO e real. Cada cliente nasce de um arquetipo latente (que o modelo
nao enxerga) e recebe uma bandeira sorteada com probabilidades que dependem do
arquetipo. As premissas de bandeira abaixo sao APROXIMADAS e editaveis: inspiradas
em ordens de grandeza publicas (BCB / Abecs), nao em numeros exatos.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 20000

CATEGORIAS = ["supermercado", "restaurantes", "viagens", "saude", "combustivel", "ecommerce", "assinaturas"]

# Arquetipos latentes: participacao na base e perfil medio
# gasto = gasto mensal (R$), ticket = ticket medio (R$), freq = compras/mes,
# parc = % do gasto parcelado, nparc = media de parcelas quando parcela,
# intl = % internacional, onl = % online, ben = % do gasto em pontos/cashback,
# uso_lim = % do limite usado, idade = media
ARQ = {
    "rotina_familiar": dict(p=0.27, gasto=3200, ticket=95,  freq=34, parc=0.18, nparc=3.0, intl=0.01, onl=0.22, ben=0.10, uso_lim=0.45, idade=41,
                            cat=[0.38, 0.10, 0.02, 0.14, 0.22, 0.09, 0.05]),
    "planejador":      dict(p=0.22, gasto=4200, ticket=310, freq=14, parc=0.62, nparc=7.0, intl=0.01, onl=0.38, ben=0.06, uso_lim=0.72, idade=38,
                            cat=[0.20, 0.08, 0.04, 0.17, 0.08, 0.38, 0.05]),
    "digital":         dict(p=0.21, gasto=2300, ticket=62,  freq=38, parc=0.12, nparc=2.5, intl=0.06, onl=0.78, ben=0.14, uso_lim=0.38, idade=29,
                            cat=[0.10, 0.18, 0.04, 0.05, 0.06, 0.42, 0.15]),
    "controlado":      dict(p=0.20, gasto=1100, ticket=70,  freq=15, parc=0.06, nparc=2.0, intl=0.00, onl=0.18, ben=0.05, uso_lim=0.22, idade=47,
                            cat=[0.42, 0.08, 0.01, 0.20, 0.17, 0.08, 0.04]),
    "viajante":        dict(p=0.10, gasto=14500, ticket=520, freq=28, parc=0.16, nparc=4.0, intl=0.24, onl=0.46, ben=0.42, uso_lim=0.34, idade=44,
                            cat=[0.12, 0.24, 0.30, 0.06, 0.05, 0.15, 0.08]),
}

# Probabilidade de bandeira por arquetipo: [Visa, Mastercard, Elo, Amex]
BANDEIRA_P = {
    "rotina_familiar": [0.42, 0.34, 0.23, 0.01],
    "planejador":      [0.40, 0.42, 0.16, 0.02],
    "digital":         [0.44, 0.44, 0.09, 0.03],
    "controlado":      [0.43, 0.28, 0.28, 0.01],
    "viajante":        [0.26, 0.34, 0.04, 0.36],
}
BANDEIRAS = ["Visa", "Mastercard", "Elo", "Amex"]


def gerar(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    nomes = list(ARQ)
    probs = np.array([ARQ[a]["p"] for a in nomes])
    arq = rng.choice(nomes, size=n, p=probs / probs.sum())
    rows = []
    for a in arq:
        c = ARQ[a]
        gasto = c["gasto"] * rng.lognormal(0, 0.35)
        ticket = max(15, c["ticket"] * rng.lognormal(0, 0.25))
        freq = max(2, gasto / ticket * rng.lognormal(0, 0.10))
        parc = float(np.clip(rng.normal(c["parc"], 0.07), 0, 0.95))
        nparc = float(np.clip(rng.normal(c["nparc"], 1.2), 2, 12)) if parc > 0.02 else 0.0
        intl = float(np.clip(rng.normal(c["intl"], 0.02 + c["intl"] * 0.3), 0, 0.8))
        onl = float(np.clip(rng.normal(c["onl"], 0.08), 0.02, 0.98))
        ben = float(np.clip(rng.normal(c["ben"], 0.06), 0, 0.9))
        uso = float(np.clip(rng.normal(c["uso_lim"], 0.12), 0.02, 1.0))
        idade = int(np.clip(rng.normal(c["idade"], 8), 18, 80))
        share = rng.dirichlet(np.array(c["cat"]) * 40 + 0.5)
        b = rng.choice(BANDEIRAS, p=BANDEIRA_P[a])
        rows.append([a, b, gasto, ticket, freq, parc, nparc, intl, onl, ben, uso, idade, *share])
    cols = ["arquetipo_latente", "bandeira", "gasto_mensal", "ticket_medio", "compras_mes", "pct_parcelado",
            "parcelas_medias", "pct_internacional", "pct_online", "pct_beneficios", "uso_limite", "idade"] + \
           [f"cat_{c}" for c in CATEGORIAS]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "cliente_id", np.arange(1, n + 1))
    return df.round(4)


if __name__ == "__main__":
    df = gerar()
    df.to_csv("dados_sinteticos.csv", index=False)
    print(df.shape)
    print(df.bandeira.value_counts(normalize=True).round(3))
