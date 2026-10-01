"""Proteu - nucleo da clusterizacao (onde o cluster e rodado).

COMO FUNCIONA, EM PORTUGUES SIMPLES
-----------------------------------
Objetivo: agrupar clientes que SE COMPORTAM de forma parecida no cartao.
O modelo nunca ve a bandeira nem o "arquetipo latente" da base sintetica.
So ve o comportamento. A bandeira entra DEPOIS, so pra descrever cada grupo.

Passo 1 - Escolher as variaveis de comportamento (gasto, ticket, frequencia,
          parcelamento, online, internacional, beneficios, uso do limite e
          mix de categorias).
Passo 2 - Padronizar (StandardScaler): coloca tudo na mesma "regua". Sem isso,
          o gasto (milhares de reais) engoliria o % de parcelado (0 a 1).
Passo 3 - K-Means: sorteia k centros, atribui cada cliente ao centro mais
          proximo, recalcula os centros, e repete ate estabilizar.
Passo 4 - Escolher k: testamos k de 3 a 8 e olhamos a silhueta (quao separados
          estao os grupos; quanto maior, melhor).
Passo 5 - Dar nome (persona): comparamos o perfil medio de cada grupo com
          perfis-modelo e atribuimos o nome que mais combina.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from gerar_dados import CATEGORIAS, BANDEIRAS

# Passo 1: variaveis que o modelo enxerga
FEATURES = ["gasto_mensal", "ticket_medio", "compras_mes", "pct_parcelado", "parcelas_medias",
            "pct_internacional", "pct_online", "pct_beneficios", "uso_limite"] + [f"cat_{c}" for c in CATEGORIAS]

# gasto e ticket tem cauda longa: usamos log pra um cliente muito rico nao dominar a regua
LOG_FEATURES = ["gasto_mensal", "ticket_medio", "compras_mes"]

ROTULOS = {
    "gasto_mensal": "Gasto mensal", "ticket_medio": "Ticket medio", "compras_mes": "Compras por mes",
    "pct_parcelado": "% parcelado", "parcelas_medias": "Parcelas (media)", "pct_internacional": "% internacional",
    "pct_online": "% online", "pct_beneficios": "% em beneficios", "uso_limite": "Uso do limite",
    **{f"cat_{c}": f"% {c}" for c in CATEGORIAS},
}

# Personas: perfis-modelo (em unidades z, "acima/abaixo da media geral") usados so para NOMEAR grupos
# Biblioteca de personas-modelo (perfis em unidades z: acima/abaixo da media geral).
# Serve so para NOMEAR grupos: cada grupo do K-Means recebe a persona mais parecida, sem repetir nome.
# Tem mais personas do que o maior k (8), entao todo k de 3 a 8 consegue nomes distintos.
PERSONAS = {
    "O Viajante Premium": dict(desc="Gasta muito, viaja, usa o cartao fora do pais e vive de pontos e beneficios.",
                               acao="Ofertas de experiencia, salas VIP, milhas bonificadas, comunicacao exclusiva.",
                               perfil=dict(gasto_mensal=2.5, ticket_medio=2, pct_internacional=2.5, pct_beneficios=2.5, cat_viagens=2.7)),
    "O Planejador": dict(desc="Compra em parcelas longas, usa bastante do limite e concentra em compras maiores.",
                         acao="Parcelamento sem juros, simulacao de parcelas, aumento de limite no momento certo.",
                         perfil=dict(pct_parcelado=1.8, parcelas_medias=1.5, uso_limite=1.4, ticket_medio=1, cat_ecommerce=1)),
    "O Digital": dict(desc="Jovem, compra online toda hora, ticket baixo, mistura e-commerce e assinaturas.",
                      acao="Cashback em apps e assinaturas, cartao virtual, comunicacao por push e in-app.",
                      perfil=dict(pct_online=1.7, cat_assinaturas=1.3, cat_ecommerce=1.2, compras_mes=0.8)),
    "O Assinante": dict(desc="Vive de servicos recorrentes (streaming, apps, mensalidades), quase tudo online.",
                        acao="Ofertas em servicos recorrentes, cashback em assinaturas, alertas de renovacao.",
                        perfil=dict(cat_assinaturas=2.2, pct_online=1.6, cat_supermercado=-1.1), icone="assinante"),
    "O Comprador Online": dict(desc="Compra principalmente em lojas virtuais; pouco supermercado, muito e-commerce.",
                               acao="Cupons e cashback em lojas online, frete gratis, cartao virtual.",
                               perfil=dict(pct_online=1.7, cat_ecommerce=1.5, cat_supermercado=-1.1), icone="comprador_online"),
    "A Rotina Familiar": dict(desc="Supermercado, combustivel e saude: muitas compras pequenas e previsiveis.",
                              acao="Cashback em supermercado e posto, debito automatico, fidelidade do dia a dia.",
                              perfil=dict(cat_supermercado=0.8, cat_combustivel=1.1, compras_mes=0.8, cat_ecommerce=-0.8, pct_online=-0.7)),
    "O Motorista de Rotina": dict(desc="Gasta forte em combustivel e deslocamento, pouco online.",
                                  acao="Cashback em posto, pedagio, estacionamento e manutencao do carro.",
                                  perfil=dict(cat_combustivel=1.6, cat_ecommerce=-0.8, pct_online=-0.7), icone="motorista_de_rotina"),
    "O Supermercadista": dict(desc="Muitas compras no supermercado e no dia a dia, pouco e-commerce.",
                              acao="Cashback e clube de vantagens em supermercado e farmacia, compra recorrente.",
                              perfil=dict(compras_mes=1.0, cat_supermercado=0.9, cat_ecommerce=-0.8), icone="rotina_familiar"),
    "O Cuidador": dict(desc="Gasto concentrado em saude e bem-estar, ticket baixo e uso moderado do limite.",
                       acao="Parcerias com farmacias, clinicas e planos de saude; parcelamento de procedimentos.",
                       perfil=dict(cat_saude=1.7, gasto_mensal=-1.2, uso_limite=-1.0), icone="cuidador"),
    "O Controlado": dict(desc="Gasta pouco, paga a vista e usa pouco do limite.",
                         acao="Educacao financeira, incentivo de uso, anuidade zero e ativacao gradual.",
                         perfil=dict(gasto_mensal=-1.3, uso_limite=-1.0, pct_parcelado=-0.8, cat_supermercado=1.1)),
    "O Gourmet": dict(desc="Restaurantes, bares e delivery pesam mais que a media.",
                      acao="Beneficios em restaurantes, delivery e experiencias gastronomicas.",
                      perfil=dict(cat_restaurantes=2.0, ticket_medio=0.5), icone="gourmet"),
    "O Alto Padrao": dict(desc="Gasto e ticket altos, mas sem o perfil de viagem internacional.",
                          acao="Limite maior, atendimento diferenciado e beneficios de categoria superior.",
                          perfil=dict(gasto_mensal=1.5, ticket_medio=1.2, uso_limite=0.5), icone="alto_padrao"),
}


def preparar(df):
    """Passos 1 e 2: seleciona as variaveis, aplica log nas de cauda longa e padroniza."""
    X = df[FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return pd.DataFrame(Xs, columns=FEATURES, index=df.index)


def avaliar_k(Xs, ks=range(3, 9), seed=42, amostra=5000):
    """Passo 4: para cada k, roda o K-Means e mede a silhueta (numa amostra, pra ser rapido)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), size=min(amostra, len(Xs)), replace=False)
    out = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        sil = silhouette_score(Xs.iloc[idx], km.labels_[idx])
        out.append(dict(k=k, silhueta=sil, inercia=km.inertia_))
    return pd.DataFrame(out)


def rodar_cluster(df, k=5, seed=42):
    """Passo 3: K-Means. Devolve df com a coluna 'cluster' e o z-score medio de cada grupo."""
    Xs = preparar(df)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xs)
    df = df.copy()
    df["cluster"] = km.labels_
    z = Xs.groupby(km.labels_).mean()  # perfil de cada grupo em "desvios-padrao acima/abaixo da media"
    return df, z


# Rotulos curtos para quando o grupo NAO se encaixa em nenhuma persona-modelo (nomes gerados dos dados)
TRACOS = {
    "gasto_mensal": ("Gasto alto", "Gasto baixo"), "ticket_medio": ("Ticket alto", "Ticket baixo"),
    "compras_mes": ("Muitas compras", "Poucas compras"), "pct_parcelado": ("Parcelador", "A vista"),
    "parcelas_medias": ("Parcelas longas", "Parcelas curtas"), "pct_internacional": ("Global", "So no Brasil"),
    "pct_online": ("Online", "Loja fisica"), "pct_beneficios": ("Cacador de beneficios", "Sem beneficios"),
    "uso_limite": ("Limite no talo", "Limite folgado"),
    **{f"cat_{c}": (c.capitalize(), f"Sem {c}") for c in CATEGORIAS},
}
# Sugestao de acao conforme o traco mais forte
ACOES = {
    "gasto_mensal": "Programas de relacionamento e beneficios premium.", "pct_parcelado": "Parcelamento sem juros e simulacao de parcelas.",
    "pct_online": "Cashback em apps e e-commerce, cartao virtual, comunicacao in-app.", "pct_internacional": "Ofertas de viagem, cambio e milhas.",
    "pct_beneficios": "Campanhas de pontos e milhas.", "uso_limite": "Gestao de limite e alertas de uso.",
    "cat_supermercado": "Cashback em supermercado.", "cat_combustivel": "Cashback em posto de combustivel.",
    "cat_assinaturas": "Ofertas em servicos recorrentes e streaming.", "cat_ecommerce": "Cupons e cashback em lojas online.",
    "cat_viagens": "Pacotes de viagem e salas VIP.", "cat_restaurantes": "Beneficios em restaurantes e delivery.",
    "cat_saude": "Parcerias com farmacia e saude.", "compras_mes": "Incentivo a frequencia com recompensas por uso.",
}
# Icone por traco (reaproveita os desenhos das personas)
ICONE_TRACO = {"pct_parcelado": "planejador", "parcelas_medias": "planejador", "uso_limite": "planejador", "pct_online": "digital",
               "cat_assinaturas": "digital", "cat_ecommerce": "digital", "pct_internacional": "viajante_premium",
               "pct_beneficios": "viajante_premium", "cat_viagens": "viajante_premium", "gasto_mensal": "viajante_premium",
               "cat_supermercado": "rotina_familiar", "cat_combustivel": "rotina_familiar", "compras_mes": "rotina_familiar",
               "cat_saude": "rotina_familiar"}
LIMITE_ENCAIXE = 1.0   # erro medio quadratico maximo para aceitar o nome de uma persona-modelo


def _slug(n):
    from personas_art import slug
    return slug(n)


def nomear(z):
    """Passo 5: da nome a cada grupo. REFEITO a cada k.

    1) Compara cada grupo com as personas-modelo (erro quadratico medio nas variaveis que definem a persona).
    2) Faz a melhor atribuicao sem repetir nome e SO aceita se o encaixe for bom (erro <= LIMITE_ENCAIXE).
    3) Grupos sem bom encaixe ganham um nome gerado a partir dos tracos mais fortes do proprio grupo.
    Devolve {cluster: dict(nome, icone, desc, acao, modelo)}.
    """
    nomes = list(PERSONAS)
    custo = np.zeros((len(z), len(nomes)))
    for i, cl in enumerate(z.index):
        for j, n in enumerate(nomes):
            alvo = PERSONAS[n]["perfil"]
            custo[i, j] = np.mean([(z.loc[cl, f] - v) ** 2 for f, v in alvo.items()])
    linhas, cols = linear_sum_assignment(custo)
    info, usados = {}, set()
    for i, j in zip(linhas, cols):
        if custo[i, j] <= LIMITE_ENCAIXE:
            n = nomes[j]
            info[z.index[i]] = dict(nome=n, icone=PERSONAS[n].get("icone", _slug(n)), desc=PERSONAS[n]["desc"], acao=PERSONAS[n]["acao"], modelo=True)
            usados.add(n)
    for cl in z.index:
        if cl in info:
            continue
        s = z.loc[cl].sort_values(key=abs, ascending=False)
        pos = [f for f in s.index if s[f] >= 0.5][:2]
        neg = [f for f in s.index if s[f] <= -0.8][:1]
        partes = [TRACOS[f][0] for f in pos] or [TRACOS[neg[0]][1]] if (pos or neg) else ["Perfil medio"]
        nome = "Perfil " + " + ".join(partes)
        base, c = nome, 2
        while nome in {v["nome"] for v in info.values()}:
            nome, c = f"{base} ({c})", c + 1
        top = pos[0] if pos else (neg[0] if neg else None)
        desc = ("Destaca-se por: " + "; ".join(f"{ROTULOS[f].lower()} ({s[f]:+.1f})" for f in s.index[:3]) + " (desvios-padrao vs. media).")
        info[cl] = dict(nome=nome, icone=ICONE_TRACO.get(top, "controlado" if neg and neg[0] in ("gasto_mensal", "uso_limite") else "generico"),
                        desc=desc, acao=ACOES.get(top, "Acompanhar de perto e testar ofertas simples."), modelo=False)
    return info


def resumo_clusters(df, mapa):
    g = df.groupby("cluster")
    r = g[FEATURES + ["idade"]].mean()
    r.insert(0, "clientes", g.size())
    r.insert(1, "pct_base", g.size() / len(df))
    r.insert(0, "persona", [mapa[c] for c in r.index])
    return r


def mix_bandeira(df, mapa):
    """Quanto de cada bandeira cai em cada persona (e vice-versa)."""
    d = df.assign(persona=df["cluster"].map(mapa))
    por_bandeira = pd.crosstab(d["bandeira"], d["persona"], normalize="index")
    por_persona = pd.crosstab(d["persona"], d["bandeira"], normalize="index")
    return por_bandeira, por_persona


def perfil_bandeira(df):
    return df.groupby("bandeira")[FEATURES + ["idade"]].mean().reindex(BANDEIRAS)
