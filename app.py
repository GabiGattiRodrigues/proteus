import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gerar_dados import BANDEIRAS, BANDEIRA_P, ARQ, gerar
from personas_art import avatar_html
from proteu_core import (FEATURES, ROTULOS, avaliar_k, mix_bandeira, nomear, perfil_bandeira,
                         preparar, resumo_clusters, rodar_cluster)

VERSAO = "v6"
st.set_page_config(page_title="Proteu", page_icon="assets/proteu.png", layout="wide")


@st.cache_data
def carregar():
    try:
        return pd.read_csv("dados_sinteticos.csv")
    except FileNotFoundError:
        return gerar()


@st.cache_data
def silhuetas(df):
    return avaliar_k(preparar(df))


@st.cache_data
def clusterizar(df, k, versao=None):
    d, z = rodar_cluster(df, k)
    info = nomear(z)          # os nomes sao REFEITOS para cada k
    return d, z, info


@st.cache_data
def kmeans_didatico(k):
    """Mini K-Means em 2 variaveis (100 clientes, 5 grupos 'naturais') rodando com o k escolhido; guarda cada iteracao."""
    rng = np.random.default_rng(7)
    cent_nat = [(1.2, 0.08), (2.8, 0.45), (4.6, 0.15), (6.6, 0.55), (9.0, 0.12)]
    pts = np.vstack([rng.normal(c, [0.45, 0.05], (20, 2)) for c in cent_nat])
    # chute inicial ruim de proposito: k clientes sorteados entre os de menor gasto (todos amontoados no mesmo canto)
    canto = np.argsort(pts[:, 0])[: max(2 * k, 12)]
    cent = pts[rng.choice(canto, size=k, replace=False)].copy()
    esc = pts.std(0)
    passos = []
    for _ in range(12):
        lab = np.argmin((((pts[:, None, :] - cent[None]) / esc) ** 2).sum(-1), axis=1)
        passos.append((cent.copy(), lab))
        novo = np.array([pts[lab == c].mean(0) if (lab == c).any() else cent[c] for c in range(k)])
        if np.allclose(novo, cent):
            break
        cent = novo
    return pts, passos


COR_CORTE = "#E8590C"   # k escolhido (laranja)
COR_IDEAL = "#2B9348"   # k ideal pela silhueta (verde)


def marca_corte(fig, x, k, ideal=False, pos="top right"):
    """Linha de corte com o mesmo visual nos 3 graficos: laranja = k escolhido; verde = k ideal (silhueta)."""
    cor = COR_IDEAL if ideal else COR_CORTE
    txt = f"ideal (silhueta): k = {k}" if ideal else f"corte: k = {k}"
    fig.add_vline(x=x, line_dash="dot" if ideal else "dash", line_color=cor, line_width=2, annotation_text=txt,
                  annotation_font_color=cor, annotation_position=pos)


def marca_cortes(fig, x_atual, k, x_ideal, melhor):
    """Desenha o corte do k escolhido e, se for diferente, o do k ideal, com rotulos em lados opostos (nao se sobrepoem)."""
    if k != melhor:
        ideal_a_esquerda = x_ideal < x_atual
        marca_corte(fig, x_ideal, melhor, ideal=True, pos="top left" if ideal_a_esquerda else "top right")
        marca_corte(fig, x_atual, k, pos="top right" if ideal_a_esquerda else "top left")
    else:
        marca_corte(fig, x_atual, k)


def destaca_ponto(fig, sil, col, k, cor=COR_CORTE):
    """Bolinha no ponto do k."""
    p = sil[sil.k == k]
    fig.add_trace(go.Scatter(x=p["k"], y=p[col], mode="markers", marker=dict(size=14, color=cor), showlegend=False,
                             hoverinfo="skip"))


def corte_dendro(Z, k, n=8):
    """Altura do corte que gera k grupos (entre a ultima juncao feita e a proxima)."""
    return (Z[n - k - 1, 2] + Z[n - k, 2]) / 2 if k < n else Z[0, 2] / 2


def reais(v):
    return f"R$ {v/1000:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".") if v >= 10000 else f"R$ {v:,.0f}".replace(",", ".")


df = carregar()

with st.sidebar:
    st.image("assets/proteu.png", width="stretch")
    st.markdown("### Proteu")
    st.caption("Personas de cartao de credito: um perfil geral e como cada bandeira se desdobra nele.")
    k = st.slider("Numero de grupos (k)", 3, 8, 5,
                  help="Quantos grupos o K-Means vai montar. Tudo no app recalcula, inclusive os NOMES dos grupos.")
    st.info("Base 100% sintetica, calibrada em ordens de grandeza publicas. Veja a aba Premissas.")
    st.caption(f"Proteu {VERSAO}")

dc, z, info = clusterizar(df, k, VERSAO)
mapa = {cl: v["nome"] for cl, v in info.items()}
icone_de = {v["nome"]: v["icone"] for v in info.values()}
res = resumo_clusters(dc, mapa)
por_bandeira, por_persona = mix_bandeira(dc, mapa)
nomes_ord = [mapa[i] for i in z.index]

st.title("Proteu: quem e o cliente por tras do cartao?")
st.caption("Primeiro agrupamos clientes pelo COMPORTAMENTO. Depois olhamos como Visa, Mastercard, Elo e Amex se distribuem nesses grupos.")

t1, t2, t3, t5, t6 = st.tabs(["Como o cluster funciona", "Escolhendo k", "Personas gerais", "Por bandeira", "Premissas"])

# ================================================================ ABA 1
with t1:
    st.header("O cluster, explicado do zero")
    st.success("""
**Em 1 minuto:** um banco tem milhares de clientes de cartao e nao da para falar com cada um de um jeito. Mas tambem nao funciona
mandar a **mesma oferta para todo mundo**. Esta analise **descobre sozinha os "tipos" de cliente** olhando so como cada um usa o cartao
(quanto gasta, onde, se parcela, se viaja...). Cada tipo vira uma **persona**, com nome, cara e um jeito certo de abordar.
Depois olhamos como as bandeiras (Visa, Mastercard, Elo, Amex) se espalham entre essas personas.

**O que se ganha:** em vez de uma campanha generica, cada persona recebe a oferta que combina com ela, por exemplo milhas para quem viaja,
parcelamento facil para quem parcela e incentivo de uso para quem quase nao usa o cartao.
""")
    st.markdown("""
**Analogia:** imagine uma festa com 20 mil convidados. Voce pede para eles formarem rodinhas **so pelo jeito que conversam e se divertem**,
sem dizer a profissao de ninguem. Depois olha cada rodinha e da um nome ("o pessoal da danca", "o pessoal do bar").
Nossas "rodinhas" sao os **grupos** e os "nomes" sao as **personas**. E o modelo **nao ve a bandeira do cartao**: ela so entra no fim, para descrever cada grupo.
""")

    st.subheader("Acompanhe 3 clientes do comeco ao fim")
    st.caption("Exemplo ilustrativo, so para entender a ideia.")
    st.dataframe(pd.DataFrame({
        "Cliente": ["Ana", "Joao", "Maria"],
        "Gasto mensal": ["R$ 15.000", "R$ 4.500", "R$ 2.800"],
        "% parcelado": ["10%", "65%", "15%"],
        "% no exterior": ["25%", "0%", "0%"],
        "Onde gasta": ["Viagens e restaurantes", "E-commerce", "Supermercado e posto"],
        "O que o algoritmo faz": ["Gasta muito e viaja: fica longe dos outros e perto de quem e como ela",
                                  "Parcela quase tudo: agrupa com outros parceladores",
                                  "Compras pequenas e do dia a dia: agrupa com quem faz o mesmo"],
        "Persona": ["O Viajante Premium", "O Planejador", "A Rotina Familiar"],
    }), hide_index=True, width="stretch")
    st.markdown("O algoritmo **nunca recebeu esses nomes**. Ele so notou que Ana, Joao e Maria se comportam de jeitos muito diferentes e juntou cada um com quem "
                "e parecido. Os nomes (persona) quem da somos nos, depois, olhando o perfil de cada grupo.")

    st.subheader("Os 6 passos, em portugues claro")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**1. Escolher o que olhar.** Para cada cliente medimos 18 coisas: quanto gasta, ticket medio, quantas compras faz, se parcela, se compra online, se usa no
exterior, se usa beneficios, quanto do limite usa e onde gasta (supermercado, restaurante, viagem...).

**2. Colocar tudo na mesma regua.** Gasto vai de centenas a dezenas de milhares; "% parcelado" vai de 0 a 1.
Se nao ajustarmos, o gasto "grita mais alto" e o modelo ignora o resto. Ajustamos para que todas as coisas pesem parecido
(exemplos mais abaixo: *log* e *z-score*).

**3. Formar os grupos (K-Means).** O algoritmo coloca **k** "pontos centrais", cada cliente vai para o centro mais perto, os centros andam para o
meio do seu grupo, e repete ate parar de mudar. Voce ve isso acontecendo no grafico logo abaixo.
""")
    with c2:
        st.markdown("""
**4. Decidir quantos grupos (k).** Testamos de 3 a 8 e vemos quando mais grupos deixam de ajudar (aba *Escolhendo k*).

**5. Dar nome (persona).** O algoritmo entrega "grupo 0, 1, 2...". Temos uma biblioteca de 12 personas-modelo
(ex.: "gasta muito + viaja + usa beneficios" = Viajante Premium) e cada grupo recebe a mais parecida, sem repetir nome.
Isso e **refeito a cada k**: com poucos grupos os nomes sao mais amplos; com muitos, mais especificos.

**6. Cruzar com a bandeira.** So agora olhamos: de cada grupo, quanto e Visa, Mastercard, Elo ou Amex?
E de cada bandeira, em quais grupos ela aparece mais? (aba *Por bandeira*)

*Onde roda no codigo:* `proteu_core.py`, comentado passo a passo.
""")

    st.subheader(f"Veja o K-Means acontecendo (exemplo com 100 clientes e k = {k} grupos)")
    st.caption("Cada ponto e um cliente. Eixo horizontal: gasto mensal (R$ mil). Eixo vertical: % do gasto parcelado. "
               "Os losangos pretos sao os centros. A base de exemplo tem **5 grupos naturais**; o K-Means roda com o **k do slider da lateral**. "
               "Mova o slider de iteracao para ver o algoritmo trabalhando.")
    pts, passos = kmeans_didatico(k)
    it = st.slider("Iteracao", 0, len(passos) - 1, 0, help="0 = chute inicial dos centros (ruim de proposito); depois ele vai se ajustando.")
    cent, lab = passos[it]
    PAL = ["#1A3C8C", "#E8590C", "#2B9348", "#7B2CBF", "#D1495B", "#0F9AA8", "#C9A227", "#6C757D"]
    fig = go.Figure()
    for c in range(k):
        m = lab == c
        fig.add_trace(go.Scatter(x=pts[m, 0], y=pts[m, 1], mode="markers", name=f"Grupo {c+1}", marker=dict(color=PAL[c], size=9, opacity=.75)))
    fig.add_trace(go.Scatter(x=cent[:, 0], y=cent[:, 1], mode="markers", name="Centros",
                             marker=dict(symbol="diamond", size=17, color="black", line=dict(color="white", width=2))))
    fig.update_layout(height=420, xaxis_title="Gasto mensal (R$ mil)", yaxis_title="% parcelado", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", key=f"kmeans_toy_{k}")
    if k < 5:
        leitura = (f"Com **k={k}**, menos que os 5 grupos naturais, o algoritmo e obrigado a **juntar** grupos diferentes: alguns centros acabam no meio de "
                   "duas nuvens, descrevendo mal os dois. E o que acontece quando escolhemos poucos grupos.")
    elif k == 5:
        leitura = ("Com **k=5**, igual ao numero de grupos naturais, cada centro acaba no meio de uma nuvem: o encaixe ideal.")
    else:
        leitura = (f"Com **k={k}**, mais que os 5 grupos naturais, o algoritmo **fatia** nuvens que ja eram coesas em pedacos menores. "
                   "Os grupos ficam mais detalhados, mas menos distintos entre si (e mais dificeis de explicar).")
    st.info("**O que olhar:** na iteracao 0 os centros estao amontoados no mesmo canto (chute ruim de proposito). A cada passo, cada cliente troca para o "
            "centro mais perto e cada centro anda para o meio do seu grupo, ate parar de mudar. " + leitura)

    st.subheader("Conceitos, um a um, com exemplos")
    with st.expander("Padronizar (z-score): colocar tudo na mesma regua"):
        st.markdown("""
Formula: **z = (valor do cliente - media da base) / desvio-padrao**. O resultado diz *quantos "desvios" o cliente esta acima ou abaixo da media*.

**Exemplo:** a media de gasto e R$ 4.000 e o desvio-padrao e R$ 2.000. A Ana gasta R$ 6.000: z = (6.000 - 4.000) / 2.000 = **+1** (um desvio acima).
Ja o Bruno parcela 60% das compras, onde a media e 30% e o desvio-padrao e 15 pontos: z = (60 - 30) / 15 = **+2**.
Em reais e em % nao da pra comparar, mas em z da: o Bruno e **mais "fora do comum" no parcelamento** do que a Ana no gasto.
""")
    with st.expander("Log: nao deixar o cliente muito rico dominar"):
        st.markdown("""
Gasto tem **cauda longa**: poucos clientes gastam muitissimo. O log "comprime" os valores altos.

**Exemplo:** passar de R$ 1.000 para R$ 2.000 e dobrar o gasto (diferenca de R$ 1.000). Passar de R$ 20.000 para R$ 21.000 tambem e uma
diferenca de R$ 1.000, mas e so +5%. Sem log, o modelo trataria as duas como iguais. Com log, a primeira pesa muito mais, que e o que faz sentido.
""")
    with st.expander("Centroide: o 'cliente medio' do grupo"):
        st.markdown("""
Centroide = a media de cada variavel dentro do grupo, o "ponto central".

**Exemplo:** tres clientes de um grupo gastam R$ 3.000, R$ 4.000 e R$ 5.000 e parcelam 10%, 20% e 30%. O centroide e
**gasto R$ 4.000 e 20% parcelado**: um cliente "ficticio" que representa bem o grupo. Na aba *Personas gerais* voce ve a tabela de centroides.
""")
    with st.expander("Silhueta: os grupos estao bem separados?"):
        st.markdown("""
Para cada cliente, comparamos a **distancia media aos colegas do proprio grupo (a)** com a **distancia media ao grupo vizinho mais proximo (b)**.
Silhueta = (b - a) / maior entre a e b. Vai de -1 a 1.

**Exemplo:** o Carlos esta, em media, a 1 "passo" dos colegas de grupo (a = 1) e a 4 passos do grupo vizinho (b = 4). Silhueta = (4 - 1) / 4 = **0,75**:
ele esta bem encaixado. Se a = 3 e b = 3,3, silhueta = 0,09: ele esta "na divisa" entre dois grupos. A silhueta do modelo e a media de todos os clientes.
""")
    with st.expander("Inercia e cotovelo: quando parar de criar grupos"):
        st.markdown("""
**Inercia** = soma das distancias de cada cliente ao centro do seu grupo. Quanto menor, mais "apertados" os grupos. Ela **sempre cai** quando aumentamos k.

**Exemplo:** e como arrumar cadeiras numa sala. Ir de 1 para 2 mesas melhora muito (cada um fica perto de alguem). De 8 para 9 mesas melhora quase nada.
O ponto onde a melhora desacelera e o **cotovelo**: dali pra frente, mais grupos so complicam a leitura.
""")
    st.subheader("PCA: o 'mapa' dos clientes (explicado do zero)")
    st.markdown("""
**O problema:** cada cliente tem **18 numeros** (nossas 18 variaveis). Para desenhar um cliente num grafico a gente so consegue usar **2 numeros** (esquerda-direita e cima-baixo).
Como mostrar 20 mil clientes de 18 numeros numa tela? E ai que entra o **PCA**.

**A ideia, com 3 exemplos do dia a dia:**
- **Globo e mapa-mundi:** a Terra e uma bola (3 dimensoes), mas o mapa-mundi e uma folha (2 dimensoes). Perde-se um pouquinho de precisao, mas continua dando para ver
  **quem esta perto de quem**. O PCA faz o mesmo com os clientes.
- **Boletim escolar:** em vez de olhar 18 notas de 18 materias, voce resume em 2 notas: "desempenho em Exatas" e "desempenho em Humanas". Cada nota nova **junta materias que andam juntas**.
- **Foto de uma escultura:** voce escolhe o angulo em que ela aparece mais completa. A foto e 2D, mas mostra o essencial.

**Importante:** o PCA e **so para desenhar**. O K-Means usa as **18 variaveis completas**, nao o mapa de 2 eixos. O mapa e a "foto" para a gente enxergar o resultado.
""")
    st.markdown("#### Exemplo bem simples: de 2 numeros para 1")
    st.caption("60 clientes ficticios com 2 numeros: gasto mensal e ticket medio. Quem gasta mais costuma ter ticket maior: os dois andam juntos. "
               "Mova o slider e veja os clientes deslizando para a linha laranja (o 'eixo' que o PCA encontrou).")
    rng_pca = np.random.default_rng(11)
    gx = rng_pca.normal(4, 1.8, 60)
    ty = 60 + 33 * gx + rng_pca.normal(0, 22, 60)
    M = np.column_stack([gx, ty])
    Ms = (M - M.mean(0)) / M.std(0)
    U, S, Vt = np.linalg.svd(Ms, full_matrices=False)
    eixo = Vt[0]
    proj = np.outer(Ms @ eixo, eixo)
    mant = S[0] ** 2 / (S ** 2).sum()
    t_proj = st.slider("Projetar os clientes no eixo (0 = nada, 1 = tudo na linha)", 0.0, 1.0, 0.0, 0.1, key="pca_toy_slider")
    P = (1 - t_proj) * Ms + t_proj * proj
    fp = go.Figure()
    xs_line = np.array([-3, 3])
    fp.add_trace(go.Scatter(x=xs_line * eixo[0], y=xs_line * eixo[1], mode="lines", line=dict(color="#E8590C", width=3), name="Eixo do PCA"))
    fp.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", marker=dict(size=9, color="#1A3C8C", opacity=.7), name="Clientes"))
    fp.update_layout(height=380, xaxis_title="Gasto mensal (padronizado)", yaxis_title="Ticket medio (padronizado)",
                     xaxis=dict(range=[-3.2, 3.2]), yaxis=dict(range=[-3.2, 3.2]))
    st.plotly_chart(fp, width="stretch", key="pca_toy")
    st.info(f"**O que olhar:** quando tudo esta sobre a linha laranja, **um unico numero** (a posicao ao longo da linha) descreve cada cliente quase tao bem quanto os dois. "
            f"Aqui a linha guarda **{mant:.0%} da informacao**. Como gasto e ticket andam juntos, nao precisamos dos dois: o PCA criou um 'eixo do tamanho do cliente'. "
            "Com 18 variaveis acontece o mesmo, so que em mais dimensoes: ele cria os 2 eixos que guardam mais informacao.")

    st.markdown("#### Agora com os dados reais: o que significa cada eixo?")
    from sklearn.decomposition import PCA
    Xs = preparar(dc)
    pca = PCA(2, random_state=1).fit(Xs)
    var1, var2 = pca.explained_variance_ratio_
    amostra = dc.sample(3000, random_state=1)
    pts2 = pca.transform(Xs.loc[amostra.index])

    def descreve(comp):
        w = pd.Series(comp, index=FEATURES).sort_values()
        def limpa(i):
            return ROTULOS[i].lower().replace("% ", "").replace("(media)", "").strip()
        mais = ", ".join(limpa(i) for i in w.index[::-1][:3])
        menos = ", ".join(limpa(i) for i in w.index[:3])
        return mais, menos
    m1, n1 = descreve(pca.components_[0])
    m2, n2 = descreve(pca.components_[1])
    st.markdown(f"""
O PCA cria 2 "notas" novas para cada cliente. Cada nota mistura as 18 variaveis:

- **Eixo 1 (horizontal) guarda {var1:.0%} da informacao.** Quanto mais para a **direita**, mais *{m1}*. Quanto mais para a **esquerda**, mais *{n1}*.
- **Eixo 2 (vertical) guarda {var2:.0%} da informacao.** Quanto mais **acima**, mais *{m2}*. Quanto mais **abaixo**, mais *{n2}*.
- Juntos, os 2 eixos guardam **{var1 + var2:.0%}** do que diferencia os clientes. O resto ({1 - var1 - var2:.0%}) fica de fora do desenho: por isso grupos que parecem encostar no mapa podem ser mais distintos nas 18 variaveis.
""")
    plot = pd.DataFrame(pts2, columns=["Eixo 1", "Eixo 2"])
    plot["Persona"] = amostra["cluster"].map(mapa).values
    fpc = px.scatter(plot, x="Eixo 1", y="Eixo 2", color="Persona", opacity=0.6, height=480)
    fpc.update_layout(xaxis_title=f"Eixo 1 ({var1:.0%} da informacao): esquerda = {n1.split(',')[0]} | direita = {m1.split(',')[0]}",
                      yaxis_title=f"Eixo 2 ({var2:.0%}): baixo = {n2.split(',')[0]} | cima = {m2.split(',')[0]}")
    st.plotly_chart(fpc, width="stretch", key=f"pca_real_{k}")
    st.info("**Como ler o mapa:** cada ponto e um cliente (amostra de 3.000) e a cor e a persona. **Pontos proximos = clientes parecidos.** "
            "Nuvens bem separadas = grupos bem definidos. Nuvens se tocando = clientes 'na divisa' entre duas personas, o que e normal em comportamento de consumo.")

    st.subheader("Dicionario rapido")
    st.dataframe(pd.DataFrame([
        ("Cluster / grupo", "Um conjunto de clientes parecidos entre si", "As rodinhas da festa"),
        ("Persona", "O nome e a 'cara' que damos a um grupo", "'O Viajante Premium'"),
        ("K-Means", "O algoritmo que forma os grupos", "Cada um vai para o centro mais perto"),
        ("k", "O numero de grupos que pedimos", "k = 5 => 5 personas"),
        ("Centroide", "O 'cliente medio' do grupo", "3 clientes gastam R$ 3, 4 e 5 mil => centroide R$ 4 mil"),
        ("Desvio-padrao / z-score", "Quantos 'degraus' acima ou abaixo da media", "+2 = bem acima do cliente medio"),
        ("Silhueta", "Nota de 0 a 1 para 'os grupos estao bem separados?'", "Perto de 1 = rodinhas bem distintas"),
        ("Inercia / cotovelo", "Quao 'apertados' estao os grupos; o cotovelo e onde mais grupos deixam de ajudar", "Passar de 8 para 9 mesas melhora pouco"),
        ("PCA", "Um 'mapa' em 2 eixos para desenhar clientes de 18 variaveis", "Globo virando mapa-mundi"),
        ("Dendrograma", "Arvore genealogica: mostra quais grupos se parecem e se juntariam primeiro", "Primos proximos se juntam por uma ponte baixa"),
        ("Bandeira", "Visa, Mastercard, Elo, Amex; nao entra no calculo, so descreve", "'70% dos Amex sao Viajante Premium'"),
    ], columns=["Termo", "Em portugues claro", "Exemplo"]), hide_index=True, width="stretch")


# ================================================================ ABA 2
with t2:
    st.header("Quantos grupos? (k)")
    sil = silhuetas(df)
    st.markdown("""
Nao existe k "certo" absoluto. Usamos duas pistas (conceitos explicados na primeira aba): a **silhueta** (quanto maior, melhor)
e o **cotovelo da inercia**. Depois, o bom senso: o k escolhido precisa render grupos que **a gente consiga explicar** para o marketing.
""")
    melhor = int(sil.loc[sil.silhueta.idxmax(), "k"])
    a, b = st.columns(2)
    with a:
        f = px.line(sil, x="k", y="silhueta", markers=True, title="Silhueta por k (quanto maior, melhor)")
        marca_cortes(f, k, k, melhor, melhor)
        if k != melhor:
            destaca_ponto(f, sil, "silhueta", melhor, COR_IDEAL)
        destaca_ponto(f, sil, "silhueta", k)
        st.plotly_chart(f, width="stretch")
    with b:
        f = px.line(sil, x="k", y="inercia", markers=True, title="Inercia por k (procure o cotovelo)")
        marca_cortes(f, k, k, melhor, melhor)
        if k != melhor:
            destaca_ponto(f, sil, "inercia", melhor, COR_IDEAL)
        destaca_ponto(f, sil, "inercia", k)
        st.plotly_chart(f, width="stretch")
    st.caption("A linha **laranja** e o **corte** no k que voce escolheu no slider. Quando ele e diferente do k **ideal** pela silhueta, "
               "aparece tambem uma linha **verde pontilhada** no k ideal. As mesmas linhas aparecem nos tres graficos desta aba "
               "(silhueta, cotovelo e dendrograma), para voce comparar a sua escolha com a recomendada.")
    if k != melhor:
        st.warning(f"Voce escolheu **k={k}**, mas o k ideal pela silhueta e **{melhor}** (linha verde). Nao e erro: escolher outro k e uma decisao de negocio, "
                   "mas vale olhar o que se ganha (mais detalhe? grupos explicaveis?) e o que se perde (grupos menos separados) em relacao ao ideal.")
    st.info(f"**Como ler:** a silhueta mais alta esta em k={melhor}. O app abre em k=5 porque rende 5 personas bem distintas e ainda com silhueta razoavel. "
            "Na inercia, a queda fica bem mais suave depois de k=5 ou 6, que e o cotovelo.")
    st.subheader("Dendrograma: como os grupos se juntam quando diminuimos o k")
    st.markdown("""
Os graficos acima dizem *quantos* grupos escolher. O **dendrograma** mostra *o que acontece* ao escolher: ele e uma **arvore genealogica**
dos grupos. Aqui partimos de **8 grupos pequenos** (k=8) e vamos juntando os mais parecidos, um par de cada vez, ate sobrar um so.

**Como e construido (clustering hierarquico, metodo de Ward):**
1. Comecamos com os 8 grupos sozinhos (as pontas da arvore).
2. Juntamos os **dois mais parecidos** em um so.
3. Repetimos (podendo juntar grupos que ja foram unidos) ate sobrar um unico grupo.

**Como ler:**
- Cada juncao e uma "ponte" horizontal. A **altura da ponte** (eixo X) e a distancia entre o que esta sendo unido.
- **Ponte baixa = grupos parecidos (podem virar um so sem perder muito). Ponte alta = grupos diferentes (juntar seria forcar).**
- A **linha laranja** e o **corte** no seu k atual (a mesma dos graficos de silhueta e cotovelo acima): ela corta a arvore, e cada ramo que ela atravessa vira um grupo. A linha verde pontilhada, quando aparece, e o corte no k ideal pela silhueta: compare quantos ramos cada linha atravessa.
  Mexa no slider de k e veja a linha andar nos tres graficos.

**Exemplo:** se duas pontas se unem por uma ponte baixinha, separar as duas em k=8 foi um detalhe e em k=7 ja viram um grupo so.
Se um grupo so entra na arvore por uma ponte altissima, ele e muito diferente de todos: **nem com k pequeno vale a pena junta-lo**.
E nas pontes que voce atravessa ao reduzir o k que se enxerga o custo de simplificar.

*O dendrograma nao altera os grupos do K-Means; ele so ajuda a entender a proximidade entre eles e a escolher o k.*
""")
    from plotly.figure_factory import create_dendrogram
    from scipy.cluster.hierarchy import linkage
    _, z8, info8 = clusterizar(df, 8, VERSAO)
    rot8 = [f"{i+1}. {info8[c]['nome']}" for i, c in enumerate(z8.index)]
    Z8 = linkage(z8.values, "ward")
    dend = create_dendrogram(z8.values, labels=rot8, orientation="left", linkagefun=lambda x: linkage(x, "ward"))
    marca_cortes(dend, corte_dendro(Z8, k), k, corte_dendro(Z8, melhor), melhor)
    dend.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=50), xaxis_title="Distancia entre grupos (Ward)")
    st.plotly_chart(dend, width="stretch")
    n_alto = Z8[-1, 2]
    ordem_j = [(Z8[i, 2], i) for i in range(len(Z8))]
    st.info(f"**Leitura com os dados atuais:** a juncao mais baixa da arvore tem altura {Z8[0, 2]:.1f} e a mais alta, {n_alto:.1f}. "
            f"Quando a diferenca entre a ponte que voce cruza ao ir de k={min(k + 1, 8)} para k={k} e as pontes anteriores e pequena, reduzir o k custa pouco. "
            "Quando ela da um salto grande, voce esta forcando a uniao de grupos bem diferentes.")
    with st.expander("Ver a tabela de distancias entre os 8 grupos"):
        from scipy.spatial.distance import cdist
        dist = pd.DataFrame(cdist(z8.values, z8.values), index=rot8, columns=rot8)
        st.caption("Distancia entre os centroides (na regua padronizada). Menor = mais parecidos. O dendrograma resume essa tabela em uma arvore.")
        fd = px.imshow(dist.round(1), color_continuous_scale="Blues_r", aspect="auto", text_auto=".1f", labels=dict(color="Distancia"), height=460)
        fd.update_xaxes(side="top", tickangle=-30)
        st.plotly_chart(fd, width="stretch")

    st.subheader(f"O que o k = {k} entrega agora")
    st.markdown("Os nomes abaixo foram **refeitos para este k**: grupos parecidos com uma persona-modelo recebem o nome dela; "
                "os demais recebem um nome gerado pelos seus tracos mais fortes.")
    cols = st.columns(min(len(info), 4))
    for n, (cl, v) in enumerate(info.items()):
        with cols[n % len(cols)]:
            st.markdown(avatar_html(v["icone"], 70), unsafe_allow_html=True)
            st.markdown(f"**{v['nome']}**  \n{res.loc[cl, 'pct_base']:.0%} da base")
    st.info("**Experimente:** va para k=3 e veja personas se fundindo (ex.: Controlado e Rotina Familiar viram um grupo so). "
            "Va para k=8 e veja um mesmo perfil se dividindo em variantes (ex.: o Digital vira Assinante e Comprador Online; a Rotina Familiar vira Supermercadista e Motorista de Rotina).")

# ================================================================ ABA 3
with t3:
    st.header("Personas gerais")
    ordem = res.sort_values("pct_base", ascending=False)
    for cl, r in ordem.iterrows():
        v = info[cl]
        with st.container(border=True):
            h1, h2, h3 = st.columns([1.1, 2.2, 2.2], gap="large")
            with h1:
                st.markdown(avatar_html(v["icone"], 120), unsafe_allow_html=True)
                st.subheader(r["persona"])
                st.metric("% da base", f"{r['pct_base']:.0%}", help=f"{int(r['clientes']):,} clientes".replace(",", "."))
            with h2:
                st.markdown(f"**Quem e:** {v['desc']}")
                st.markdown(f"**Como abordar:** {v['acao']}")
                m1, m2 = st.columns(2)
                m1.metric("Gasto mensal", reais(r["gasto_mensal"]))
                m2.metric("Ticket medio", reais(r["ticket_medio"]))
                m3, m4 = st.columns(2)
                m3.metric("% parcelado", f"{r['pct_parcelado']:.0%}")
                m4.metric("Idade media", f"{r['idade']:.0f}")
            with h3:
                topz = z.loc[cl].reindex(z.loc[cl].abs().sort_values(ascending=False).index).head(6)[::-1]
                fb = go.Figure(go.Bar(x=topz.values, y=[ROTULOS[i] for i in topz.index], orientation="h",
                                      marker_color=["#C0392B" if x > 0 else "#2E86C1" for x in topz.values],
                                      text=[f"{x:+.1f}" for x in topz.values], textposition="outside", cliponaxis=False))
                fb.update_layout(title=dict(text="O que mais destaca o grupo (desvios-padrao vs. media)", font=dict(size=13)),
                                 height=270, margin=dict(l=10, r=30, t=40, b=10), xaxis=dict(zeroline=True, range=[-3.5, 3.5]),
                                 showlegend=False)
                st.plotly_chart(fb, width="stretch", key=f"bar_{cl}_{k}")

    with st.expander("Ver os centroides de cada variavel (tabela completa)"):
        st.markdown("""
O **centroide** e o "cliente medio" de cada grupo. Duas formas de ver:
- **Valores reais:** a media de cada variavel dentro do grupo (em reais, % ou unidades), mais facil de ler.
- **Padronizado:** o centroide na regua que o K-Means usa de verdade (0 = media de todos os clientes; +1 = um desvio-padrao acima).
""")
        modo = st.radio("Como mostrar", ["Valores reais", "Padronizado (z)"], horizontal=True)
        if modo == "Valores reais":
            tab = res[FEATURES + ["idade"]].T
            tab.columns = nomes_ord
            tab.index = [ROTULOS.get(i, "Idade (anos)") for i in tab.index]

            def fmt(nome, x):
                if nome.startswith("%") or nome == "Uso do limite":
                    return f"{x:.0%}"
                if nome in ("Gasto mensal", "Ticket medio"):
                    return reais(x)
                return f"{x:.1f}"
            tab = pd.DataFrame({c: [fmt(i, x) for i, x in zip(tab.index, tab[c])] for c in tab.columns}, index=tab.index)
            st.dataframe(tab, width="stretch", height=640)
        else:
            tz = z.T
            tz.columns = nomes_ord
            tz.index = [ROTULOS[i] for i in tz.index]
            ftz = px.imshow(tz.round(2), color_continuous_scale="RdBu_r", zmin=-3, zmax=3, aspect="auto", text_auto="+.2f",
                            labels=dict(color="z"), height=640)
            ftz.update_xaxes(side="top")
            st.plotly_chart(ftz, width="stretch")
        st.caption("Gasto, ticket e compras por mes entram no modelo em escala log; na tabela de valores reais mostramos a media simples do grupo.")

    st.subheader("Mapa de calor: como cada persona se diferencia")
    st.markdown("""
**Como ler:** cada **linha** e uma persona e cada **coluna** e uma variavel. A cor responde:
*"esse grupo tem mais ou menos disso do que o cliente medio da base inteira?"*
- **Vermelho** = **mais** que a media (quanto mais escuro, maior a diferenca).
- **Azul** = **menos** que a media.
- **Branco** = parecido com a media.

O numero dentro do quadradinho e a diferenca em **desvios-padrao** (+2,0 = bem acima; -1,0 = abaixo; perto de 0 = na media).
""")
    hz = z.copy()
    hz.index = nomes_ord
    hz.columns = [ROTULOS[c] for c in hz.columns]
    f = px.imshow(hz.round(1), color_continuous_scale="RdBu_r", zmin=-3, zmax=3, aspect="auto", height=420,
                  text_auto=".1f", labels=dict(color="Desvios-padrao"))
    f.update_xaxes(side="top", tickangle=-45)
    st.plotly_chart(f, width="stretch")
    st.markdown("**Exemplos de leitura com os dados atuais:**")
    stk = hz.stack()
    p_max, v_max = stk.abs().idxmax()
    val = hz.loc[p_max, v_max]
    p_min, v_min = stk.abs().idxmin()
    st.markdown(f"- O quadradinho mais forte do mapa e **{p_max} x {v_max}** = **{val:+.1f}**. Traduzindo: esse grupo esta "
                f"{abs(val):.1f} desvios-padrao {'acima' if val > 0 else 'abaixo'} da media em *{v_max.lower()}*. E isso que define a personalidade dele.")
    st.markdown(f"- O mais proximo de zero e **{p_min} x {v_min}** = **{hz.loc[p_min, v_min]:+.1f}**: nessa variavel o grupo e igual ao cliente medio, "
                "entao ela **nao ajuda a distinguir** esse grupo dos outros.")
    st.caption("Dica: para descobrir 'quem e' uma persona, leia a linha dela procurando os quadradinhos mais escuros (vermelho ou azul).")

# ================================================================ ABA 5
with t5:
    st.header("E por bandeira?")
    st.markdown("""
Duas leituras complementares:
- **Composicao por bandeira:** "de todos os clientes Amex, quantos % sao de cada persona?" (a persona da bandeira).
- **Perfil medio:** como o cliente medio de cada bandeira se comporta.

**Exemplo:** se 70% dos clientes Amex caem na persona *Viajante Premium* e so 6% dos Visa, entao o cartao Amex "puxa" para esse perfil.
""")
    f = px.bar(por_bandeira.reset_index().melt(id_vars="bandeira", var_name="Persona", value_name="share"),
               x="bandeira", y="share", color="Persona", title="Composicao de personas dentro de cada bandeira",
               category_orders={"bandeira": BANDEIRAS}, height=420)
    f.update_yaxes(tickformat=".0%", title="")
    f.update_xaxes(title="")
    st.plotly_chart(f, width="stretch")

    escolha = st.selectbox("Ver a persona de uma bandeira", BANDEIRAS)
    sub = dc[dc.bandeira == escolha]
    topo = por_bandeira.loc[escolha].sort_values(ascending=False)
    pb = perfil_bandeira(dc)
    geral = dc[FEATURES + ["idade"]].mean()
    dif = ((pb.loc[escolha] - geral) / dc[FEATURES + ["idade"]].std()).sort_values(key=abs, ascending=False).head(4)
    with st.container(border=True):
        i1, i2 = st.columns([1, 3])
        with i1:
            st.markdown(avatar_html(icone_de[topo.index[0]], 120), unsafe_allow_html=True)
            st.caption(f"Persona dominante: {topo.index[0]}")
        with i2:
            st.subheader(f"Persona {escolha}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Clientes na base", f"{len(sub):,}".replace(",", "."))
            c2.metric("Gasto mensal medio", reais(sub.gasto_mensal.mean()))
            c3.metric("Idade media", f"{sub.idade.mean():.0f}")
            st.markdown(f"**Persona dominante:** {topo.index[0]} ({topo.iloc[0]:.0%} dos clientes {escolha}); "
                        f"em segundo, {topo.index[1]} ({topo.iloc[1]:.0%}).")
            st.markdown("**O que diferencia essa bandeira da media geral:** " +
                        "; ".join(f"{ROTULOS.get(i, 'Idade')} ({'acima' if x > 0 else 'abaixo'}, {x:+.2f} dp)" for i, x in dif.items()))
            st.caption("Lembrete: as diferencas entre bandeiras vem das premissas da simulacao (aba Premissas), nao de dados reais.")
    st.subheader("Qual bandeira e mais 'concentrada' em cada persona?")
    st.caption("Leia por linha: de todos os clientes de uma persona, qual % e de cada bandeira.")
    st.dataframe(por_persona.style.format("{:.0%}"), width="stretch")

# ================================================================ ABA 6
with t6:
    st.header("Premissas dos dados")
    st.warning("A base e **sintetica**. Cada cliente nasce de um arquetipo escondido (que o modelo nao ve) e recebe uma bandeira "
               "sorteada com as probabilidades abaixo. Os numeros sao ordens de grandeza inspiradas em dados publicos (BCB, Abecs), "
               "nao valores exatos, e podem ser editados em `gerar_dados.py`.")
    tab = pd.DataFrame(BANDEIRA_P, index=BANDEIRAS).T
    tab.insert(0, "% da base", [ARQ[a]["p"] for a in tab.index])
    st.markdown("**Probabilidade de bandeira por arquetipo latente**")
    st.dataframe(tab.style.format("{:.0%}"), width="stretch")
    st.markdown("**Teste de sanidade:** o cluster conseguiu redescobrir os arquetipos escondidos?")
    cruz = pd.crosstab(dc["arquetipo_latente"], dc["cluster"].map(mapa), normalize="index")
    st.dataframe(cruz.style.format("{:.0%}"), width="stretch")
    st.caption("Linhas = arquetipo verdadeiro (so existe porque a base e sintetica); colunas = persona que o K-Means encontrou. "
               "Numeros altos concentrados em uma coluna por linha significam que o metodo recupera bem a estrutura. Com dados reais esse teste nao existe.")
    st.download_button("Baixar a base sintetica (CSV)", df.to_csv(index=False), "dados_sinteticos.csv")
