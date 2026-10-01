# Proteu

Analise de persona em cartao de credito: clusterizacao comportamental (K-Means) e como Visa, Mastercard, Elo e Amex se distribuem nas personas.

![Proteu](assets/proteu.png)

## Rodar
Duplo clique em `iniciar.bat` (cria o ambiente, gera a base e abre o app).
Ou: `pip install -r requirements.txt` e `streamlit run app.py`.

## Arquivos
- `gerar_dados.py`: base sintetica e premissas por bandeira (edite aqui)
- `proteu_core.py`: onde o cluster roda, comentado passo a passo
- `app.py`: app Streamlit (a aba "Como o cluster funciona" explica tudo em linguagem simples)

## Aviso
Os dados sao sinteticos. As diferencas entre bandeiras vem das premissas da simulacao, inspiradas em ordens de grandeza publicas, e nao sao achados sobre o mercado real.
