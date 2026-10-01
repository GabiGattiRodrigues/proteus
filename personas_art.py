"""Ilustracoes das personas (SVG simples, desenhadas no codigo).

Quer usar suas proprias imagens? Coloque PNGs em assets/personas/ com o nome da persona sem acento e sem espaco,
ex.: assets/personas/viajante_premium.png. Se o arquivo existir, o app usa ele no lugar do desenho.
"""
import base64
import os
import re
import unicodedata

PASTA = os.path.join(os.path.dirname(__file__), "assets", "personas")


def slug(nome):
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    s = re.sub(r"^(o|a)\s+", "", s.lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _svg(fundo, corpo):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="100%">'
            f'<circle cx="100" cy="100" r="96" fill="{fundo}"/>{corpo}</svg>')


ARTE = {
    # mala de viagem + aviao
    "viajante_premium": _svg("#FFE8CC", """
        <path d="M30 70 L70 82 L100 52 L112 56 L96 88 L128 98 L136 88 L146 91 L140 108 L58 84 L50 94 L40 90 L46 76 Z" fill="#1A3C8C"/>
        <rect x="55" y="105" width="90" height="62" rx="9" fill="#8B5A2B"/>
        <rect x="80" y="93" width="40" height="16" rx="6" fill="none" stroke="#5C3A1A" stroke-width="6"/>
        <rect x="55" y="128" width="90" height="7" fill="#C9A227"/>
        <rect x="93" y="124" width="14" height="16" rx="3" fill="#E8C547"/>"""),
    # calendario com parcelas
    "planejador": _svg("#E3F0FF", """
        <rect x="45" y="50" width="110" height="110" rx="10" fill="#FFFFFF" stroke="#1A3C8C" stroke-width="5"/>
        <rect x="45" y="50" width="110" height="26" rx="10" fill="#1A3C8C"/>
        <rect x="68" y="38" width="8" height="22" rx="4" fill="#E8590C"/><rect x="124" y="38" width="8" height="22" rx="4" fill="#E8590C"/>
        <g fill="#9DB7E5"><circle cx="67" cy="98" r="7"/><circle cx="100" cy="98" r="7"/><circle cx="133" cy="98" r="7"/>
        <circle cx="67" cy="125" r="7"/><circle cx="100" cy="125" r="7"/></g>
        <circle cx="133" cy="125" r="9" fill="#2B9348"/><path d="M128 125 l4 4 l8 -9" stroke="#fff" stroke-width="3" fill="none"/>"""),
    # celular com notificacoes
    "digital": _svg("#F3E5FF", """
        <rect x="62" y="30" width="76" height="140" rx="14" fill="#2B2D42"/>
        <rect x="69" y="44" width="62" height="108" rx="6" fill="#FFFFFF"/>
        <rect x="76" y="54" width="48" height="14" rx="4" fill="#7B2CBF"/>
        <rect x="76" y="76" width="48" height="10" rx="4" fill="#D9C2F0"/><rect x="76" y="92" width="34" height="10" rx="4" fill="#D9C2F0"/>
        <rect x="76" y="110" width="48" height="30" rx="6" fill="#FFD6E8"/>
        <path d="M100 133 c-10 -8 -14 -12 -14 -18 a7 7 0 0 1 14 -2 a7 7 0 0 1 14 2 c0 6 -4 10 -14 18z" fill="#E63978"/>
        <circle cx="140" cy="40" r="14" fill="#E63978"/><text x="140" y="46" font-size="16" font-weight="700" text-anchor="middle" fill="#fff" font-family="sans-serif">3</text>"""),
    # casa + carrinho
    "rotina_familiar": _svg("#E3F6E8", """
        <path d="M100 38 L40 92 H56 V150 H144 V92 H160 Z" fill="#F4A261"/>
        <path d="M100 38 L34 96 L44 104 L100 54 L156 104 L166 96 Z" fill="#B5472B"/>
        <rect x="88" y="112" width="24" height="38" rx="3" fill="#6B3E26"/>
        <rect x="62" y="104" width="18" height="18" rx="2" fill="#BFE3F5"/><rect x="120" y="104" width="18" height="18" rx="2" fill="#BFE3F5"/>
        <path d="M150 140 h26 l-6 22 h-18 z" fill="#2B9348"/><circle cx="156" cy="170" r="4" fill="#333"/><circle cx="170" cy="170" r="4" fill="#333"/>"""),
    # cofrinho
    "controlado": _svg("#FFE3EC", """
        <ellipse cx="100" cy="112" rx="58" ry="44" fill="#F29BB4"/>
        <path d="M62 78 L72 56 L88 76 Z" fill="#E67A9C"/>
        <ellipse cx="160" cy="112" rx="14" ry="11" fill="#E67A9C"/><circle cx="156" cy="112" r="2.5" fill="#8A2D4B"/><circle cx="165" cy="112" r="2.5" fill="#8A2D4B"/>
        <circle cx="136" cy="98" r="5" fill="#3A1A25"/>
        <rect x="80" y="156" width="14" height="22" rx="4" fill="#E67A9C"/><rect x="116" y="156" width="14" height="22" rx="4" fill="#E67A9C"/>
        <rect x="86" y="72" width="34" height="7" rx="3.5" fill="#8A2D4B"/>
        <circle cx="103" cy="48" r="13" fill="#E8C547" stroke="#C9A227" stroke-width="3"/><text x="103" y="54" font-size="15" font-weight="700" text-anchor="middle" fill="#8A6D00" font-family="sans-serif">R$</text>"""),
    # botao de play + renovacao (assinaturas)
    "assinante": _svg("#EDE7FF", """
        <rect x="42" y="56" width="116" height="80" rx="12" fill="#5B3CC4"/>
        <path d="M88 80 L88 112 L116 96 Z" fill="#FFFFFF"/>
        <path d="M60 150 a40 40 0 0 0 80 0" fill="none" stroke="#5B3CC4" stroke-width="7" stroke-linecap="round"/>
        <path d="M140 150 l-3 -13 l13 3 z" fill="#5B3CC4"/>"""),
    # sacola de compras com cursor
    "comprador_online": _svg("#E0F7F7", """
        <path d="M58 82 H142 L150 160 H50 Z" fill="#0F9AA8"/>
        <path d="M80 82 v-12 a20 20 0 0 1 40 0 v12" fill="none" stroke="#0A6F7A" stroke-width="8"/>
        <circle cx="82" cy="106" r="5" fill="#fff"/><circle cx="118" cy="106" r="5" fill="#fff"/>
        <path d="M120 128 l0 30 l9 -8 l7 15 l8 -4 l-7 -15 l12 -2 z" fill="#FFFFFF" stroke="#333" stroke-width="3"/>"""),
    # bomba de combustivel
    "motorista_de_rotina": _svg("#FFF1D6", """
        <rect x="58" y="46" width="62" height="112" rx="8" fill="#D1495B"/>
        <rect x="68" y="58" width="42" height="32" rx="4" fill="#FFFFFF"/>
        <rect x="48" y="156" width="82" height="12" rx="4" fill="#7A2430"/>
        <path d="M120 70 h18 a8 8 0 0 1 8 8 v50 a8 8 0 0 0 16 0 v-44 l-12 -12" fill="none" stroke="#2B2D42" stroke-width="7" stroke-linecap="round"/>
        <circle cx="89" cy="116" r="12" fill="#FFD166"/>"""),
    # coracao com cruz (saude)
    "cuidador": _svg("#FFE3E3", """
        <path d="M100 160 C40 118 36 76 66 62 C82 55 96 62 100 76 C104 62 118 55 134 62 C164 76 160 118 100 160 Z" fill="#E5383B"/>
        <rect x="92" y="82" width="16" height="48" rx="3" fill="#FFFFFF"/><rect x="76" y="98" width="48" height="16" rx="3" fill="#FFFFFF"/>"""),
    # prato com talheres
    "gourmet": _svg("#FFF0E0", """
        <circle cx="100" cy="102" r="46" fill="#FFFFFF" stroke="#C8553D" stroke-width="7"/>
        <circle cx="100" cy="102" r="26" fill="#F4A261"/>
        <path d="M34 60 v50 M28 60 v22 M40 60 v22 M34 82 v28" stroke="#6B6B6B" stroke-width="5" stroke-linecap="round" fill="none"/>
        <path d="M168 60 c-10 8 -10 30 0 40 v10 M168 60 v50" stroke="#6B6B6B" stroke-width="5" stroke-linecap="round" fill="none"/>"""),
    # diamante
    "alto_padrao": _svg("#E6F0FF", """
        <path d="M60 78 L82 52 H118 L140 78 L100 150 Z" fill="#4CC9F0"/>
        <path d="M60 78 H140 M82 52 L100 78 L118 52 M100 78 L100 150 M82 78 L100 150 M118 78 L100 150" stroke="#FFFFFF" stroke-width="4" fill="none"/>
        <path d="M150 54 l4 10 l10 4 l-10 4 l-4 10 l-4 -10 l-10 -4 l10 -4 z" fill="#FFD166"/>"""),
}

GENERICO = _svg("#EEEEEE", '<circle cx="100" cy="100" r="40" fill="#BBBBBB"/><text x="100" y="116" font-size="48" text-anchor="middle" fill="#fff" font-family="sans-serif" font-weight="700">?</text>')


def avatar_html(nome, largura=150):
    """Devolve HTML com o PNG do usuario (se existir) ou o desenho SVG."""
    s = slug(nome)
    png = os.path.join(PASTA, f"{s}.png")
    if os.path.exists(png):
        b64 = base64.b64encode(open(png, "rb").read()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{largura}" style="border-radius:50%"/>'
    svg = ARTE.get(s, GENERICO)
    return f'<div style="width:{largura}px">{svg}</div>'
