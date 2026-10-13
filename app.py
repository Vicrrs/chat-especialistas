"""App Streamlit: configurador do Chat Multi-Especialistas (Databricks Apps).

Abas:
  1 · Modelos e Genies   escolhe as IAs, Genie Spaces, SQL Warehouse e preços
  2 · Agentes            cria os agentes: nome, prompt de negócio, tabelas, Genie, documentos
  3 · Estudar tabelas    o sistema estuda as tabelas e escreve o conhecimento de cada agente
  4 · Agente geral       gera o agents.yaml e dispara o job que avalia, registra e publica
  5 · Conversar          testa o agente publicado
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from chat_especialistas.app_servicos import Servicos, extrair_resposta  # noqa: E402
from chat_especialistas.configurador.descoberta import mensagem_erro  # noqa: E402
from chat_especialistas.configurador.estudo import EstudoAgente, estudar_agente  # noqa: E402
from chat_especialistas.configurador.gerador import (  # noqa: E402
    PRECO_LLAMA_70B,
    AgenteRascunho,
    Selecao,
    config_para_yaml,
    estado_de_json,
    estado_para_json,
    gerar_config,
    slug,
)

st.set_page_config(page_title="Chat Multi-Especialistas", page_icon="🤖", layout="wide")

PRECO_PADRAO = {"input": 1.00, "output": 3.00}
NOVO = "__novo__"
AJUDA_PROMPT = (
    "Explique como faria para um analista novo:\n"
    "• qual área/regra de negócio este agente atende e quem vai perguntar;\n"
    "• o que existe em cada tabela e o significado das colunas importantes;\n"
    "• como o negócio calcula as métricas (ex.: 'inadimplente = atraso > 30 dias');\n"
    "• filtros padrão, armadilhas e termos que os usuários usam."
)
ss = st.session_state


# =================================================================== estado
def servicos() -> Servicos:
    """Recria os clientes quando o token do usuário muda (ele é renovado a cada requisição)."""
    headers = dict(st.context.headers or {})
    token = {k.lower(): v for k, v in headers.items()}.get("x-forwarded-access-token")
    atual = ss.get("servicos")
    if atual is None or (token and getattr(atual, "token_hash", None) != hash(token)):
        ss.servicos = Servicos.do_contexto(headers, w_app=getattr(atual, "w_app", None))
    return ss.servicos


def caminho(nome: str) -> str:
    return f"{Servicos.pasta_config(ss.catalogo, ss.schema)}/{nome}"


def carregar_projeto():
    sel, agentes, extra = estado_de_json(servicos().ler(caminho("assistente_estado.json")))
    sel.catalogo, sel.schema = ss.catalogo, ss.schema
    ss.sel, ss.agentes, ss.extra = sel, agentes, extra
    ss.carregado_de = (ss.catalogo, ss.schema)


def salvar(msg: str | None = None) -> bool:
    try:
        servicos().gravar(caminho("assistente_estado.json"), estado_para_json(ss.sel, ss.agentes, ss.extra))
        if msg:
            st.toast(msg, icon="✅")
        return True
    except Exception as e:  # noqa: BLE001
        st.error(
            f"Não consegui salvar em `{caminho('assistente_estado.json')}`: {mensagem_erro(e)}\n\n"
            "Na aba **1 · Modelos e Genies**, escolha um SQL Warehouse e clique em **Salvar escolhas** "
            "para criar o schema e os volumes do projeto."
        )
        return False


def recursos(forcar: bool = False):
    if forcar or "recursos" not in ss:
        with st.spinner("Listando IAs, Genie Spaces e SQL Warehouses do workspace…"):
            ss.recursos = servicos().recursos()
    return ss.recursos


def cache(chave: str, func, *args):
    ss.setdefault("_cache", {})
    k = (chave, *args)
    if k not in ss._cache:
        try:
            ss._cache[k] = func(*args)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Não foi possível listar {chave}: {mensagem_erro(e)}")
            ss._cache[k] = []
    return ss._cache[k]


def agente(id_: str | None) -> AgenteRascunho | None:
    return next((a for a in ss.agentes if a.id == id_), None)


def rotulo_agente(id_: str) -> str:
    if id_ == NOVO:
        return "➕ Novo agente"
    a = agente(id_)
    return f"{'✅' if a.estudado else '⏳'} {a.nome} · {a.id}" if a else id_


def preco_de(modelo: str, campo: str) -> float:
    p = ss.sel.precos.get(modelo) or (PRECO_LLAMA_70B if "llama-3-3-70b" in (modelo or "") else PRECO_PADRAO)
    return float(p[campo])


# ================================================================ inicialização
ss.setdefault("catalogo", os.environ.get("PROJETO_CATALOGO", "main"))
ss.setdefault("schema", os.environ.get("PROJETO_SCHEMA", "chat_especialistas"))
if ss.get("carregado_de") != (ss.catalogo, ss.schema):
    carregar_projeto()
ss.setdefault("chat", [])
ss.setdefault("conversa_id", str(uuid.uuid4()))
ss.setdefault("gasto_chat", {"tokens": 0, "usd": 0.0})

# ===================================================================== sidebar
with st.sidebar:
    st.markdown("## 🤖 Chat Multi-Especialistas")
    sv = servicos()
    if sv.identidade == "usuario":
        st.success(f"Conectado como **{sv.usuario or 'você'}**. A app usa as suas permissões.")
    else:
        st.warning(
            (sv.motivo or "A app está usando a **própria conta de serviço**.")
            + f"\n\nEla só enxerga o que foi concedido a ela (application id `{sv.id_conta_servico}`)."
        )
    with st.expander("🩺 Diagnóstico de acesso"):
        host = getattr(getattr(sv.w_app, "config", None), "host", "?")
        st.caption(f"Host: `{host}` · identidade: **{sv.identidade}**")
        if st.button("Testar acesso à API", use_container_width=True):
            st.dataframe(pd.DataFrame(sv.diagnostico()), hide_index=True, use_container_width=True)
            st.caption("200 = ok · 401/403 = token sem permissão/escopo · 30x = redirecionamento para login.")
    st.markdown("#### Projeto")
    st.caption("Onde ficam a configuração, os documentos, o LLMOps e o modelo registrado.")
    novo_cat = st.text_input("Catálogo", ss.catalogo)
    novo_sch = st.text_input("Schema", ss.schema)
    if st.button("Abrir projeto", use_container_width=True) and (novo_cat, novo_sch) != (ss.catalogo, ss.schema):
        ss.catalogo, ss.schema = novo_cat.strip(), novo_sch.strip()
        st.rerun()
    st.caption(f"Config: `{caminho('agents.yaml')}`")
    st.divider()
    estudados = sum(a.estudado for a in ss.agentes)
    st.metric("Agentes", len(ss.agentes))
    st.caption(f"{estudados} estudado(s) · {len(ss.agentes) - estudados} pendente(s)")
    if ss.gasto_chat["tokens"]:
        st.metric("Gasto no teste do chat", f"US$ {ss.gasto_chat['usd']:.4f}")
        st.caption(f"{ss.gasto_chat['tokens']} tokens")

aba1, aba2, aba3, aba4, aba5 = st.tabs(
    ["1 · Modelos e Genies", "2 · Agentes", "3 · Estudar tabelas", "4 · Agente geral", "5 · Conversar"]
)

# ======================================================================= aba 1
with aba1:
    r = recursos()
    c1, c2 = st.columns([4, 1])
    c1.markdown(
        f"Encontrei **{len(r.llms_chat)}** LLMs de chat, **{len(r.embeddings)}** modelos de embeddings, "
        f"**{len(r.genies)}** Genie Spaces e **{len(r.warehouses)}** SQL Warehouses que você pode usar."
    )
    if c2.button("🔄 Atualizar listas"):
        ss.pop("_cache", None)
        recursos(forcar=True)
        st.rerun()
    for aviso in r.avisos:
        st.warning(aviso)

    sel: Selecao = ss.sel
    chat = {m.nome: m.rotulo for m in r.llms_chat}
    nomes_chat = list(chat)
    padrao = next((n for n in nomes_chat if "llama-3-3-70b" in n), nomes_chat[0] if nomes_chat else None)
    emb = {"": "(nenhum — agentes sem documentos)", **{m.nome: m.rotulo for m in r.embeddings}}
    whs = {"": "(nenhum)", **{x.id: x.rotulo for x in r.warehouses}}
    genies = {g.space_id: g for g in r.genies}

    def idx(opcoes, valor, alternativa=None):
        lista = list(opcoes)
        if valor in lista and valor:
            return lista.index(valor)
        return lista.index(alternativa) if alternativa in lista else 0

    if not nomes_chat:
        st.error("Nenhum endpoint de LLM de chat disponível para você. Verifique as permissões em Serving.")
    with st.form("form_modelos"):
        st.markdown("#### IAs")
        roteador = st.selectbox("IA do Agente 01 (detector de intenção)", nomes_chat, index=idx(nomes_chat, sel.llm_roteador, padrao),
                                format_func=chat.get, help="Conversa com o usuário e escolhe o especialista. Um modelo rápido funciona bem.")
        especialistas = st.selectbox("IA dos agentes especialistas", nomes_chat, index=idx(nomes_chat, sel.llm_especialistas, padrao),
                                     format_func=chat.get, help="Analisa os dados e escreve SQL. Precisa suportar tool calling. Também é usada no estudo.")
        embedding = st.selectbox("Modelo de embeddings (documentos)", list(emb), format_func=emb.get,
                                 index=idx(emb, sel.embedding, next((k for k in emb if "gte" in k or "bge" in k), "")))
        st.markdown("#### Dados")
        warehouse = st.selectbox("SQL Warehouse", list(whs), format_func=whs.get,
                                 index=idx(whs, sel.warehouse_id, next(iter(list(whs)[1:]), "")),
                                 help="Usado pelos agentes para consultar as tabelas e pela app para estudá-las.")
        auth = st.radio("Acesso aos dados no endpoint publicado", ["servico", "usuario"], horizontal=True,
                        index=0 if sel.autenticacao == "servico" else 1,
                        format_func={"servico": "Credencial do endpoint", "usuario": "Permissão de cada usuário"}.get)
        st.markdown("#### Genie Spaces")
        genies_sel = st.multiselect("Genies que os agentes poderão usar", list(genies), default=[g for g in sel.genies if g in genies],
                                    format_func=lambda g: genies[g].rotulo)
        st.markdown("#### Preço dos modelos (LLMOps) — US$ por 1 milhão de tokens")
        p1, p2, p3, p4 = st.columns(4)
        rot_in = p1.number_input("Agente 01 · entrada", value=preco_de(roteador, "input"), format="%.4f")
        rot_out = p2.number_input("Agente 01 · saída", value=preco_de(roteador, "output"), format="%.4f")
        esp_in = p3.number_input("Especialistas · entrada", value=preco_de(especialistas, "input"), format="%.4f")
        esp_out = p4.number_input("Especialistas · saída", value=preco_de(especialistas, "output"), format="%.4f")
        st.caption("Os valores sugeridos são de tabela pública; confirme no seu contrato Databricks.")
        enviar = st.form_submit_button("💾 Salvar escolhas", type="primary")

    if enviar:
        sel.llm_roteador, sel.llm_especialistas = roteador or "", especialistas or ""
        sel.embedding, sel.warehouse_id, sel.autenticacao = embedding, warehouse, auth
        sel.genies = list(genies_sel)
        sel.precos = {roteador: {"input": rot_in, "output": rot_out}, especialistas: {"input": esp_in, "output": esp_out}}
        if warehouse:
            with st.spinner(f"Preparando {ss.catalogo}.{ss.schema} (schema e volumes)…"):
                erros = servicos().preparar_projeto(ss.catalogo, ss.schema, warehouse)
            for e in erros:
                st.error(e)
        salvar("Escolhas salvas")

    if sel.genies and st.button("➕ Criar um agente para cada Genie selecionado"):
        criados = 0
        for gid in sel.genies:
            if gid in genies and not any(a.genie_space_id == gid and not a.tabelas for a in ss.agentes):
                g = genies[gid]
                ss.agentes.append(AgenteRascunho(nome=g.titulo, prompt_negocio=g.descricao, genie_space_id=gid))
                criados += 1
        salvar(f"{criados} agente(s) criado(s). Revise o prompt na aba 2.")


# ======================================================================= aba 2
def _carregar_edicao():
    a = agente(ss.agente_em_edicao)
    ss.ed_nome = a.nome if a else ""
    ss.ed_prompt = a.prompt_negocio if a else ""
    ss.ed_tabelas = list(a.tabelas) if a else []
    ss.ed_genie = (a.genie_space_id or "") if a else ""
    ss.ed_docs = a.documentos if a else False


def _adicionar_tabelas():
    ss.ed_tabelas = sorted(set(ss.ed_tabelas) | set(ss.get("pick_tabs", [])))
    ss.pick_tabs = []


def _salvar_agente():
    nome = ss.ed_nome.strip()
    if not nome:
        ss.msg_agente = ("erro", "Dê um nome ao agente.")
        return
    if not (ss.ed_tabelas or ss.ed_genie or ss.ed_docs):
        ss.msg_agente = ("erro", "O agente precisa de tabelas, um Genie ou documentação.")
        return
    a = agente(ss.agente_em_edicao)
    if a is None:
        a = AgenteRascunho(nome=nome)
        while agente(a.id):
            a.id = f"{a.id}_{len(ss.agentes) + 1}"
        ss.agentes.append(a)
    tabelas = sorted({t.lower() for t in ss.ed_tabelas})
    mudou = tabelas != a.tabelas or (ss.ed_genie or None) != a.genie_space_id or ss.ed_prompt != a.prompt_negocio
    a.nome, a.prompt_negocio, a.tabelas = nome, ss.ed_prompt, tabelas
    a.genie_space_id, a.documentos = ss.ed_genie or None, bool(ss.ed_docs)
    if mudou and a.estudo:
        a.estudo = None  # mudou o que ele sabe: precisa estudar de novo
    ss.agente_em_edicao = a.id
    ss.msg_agente = ("ok", f"Agente **{a.nome}** (`{a.id}`) salvo." + (" Estude-o de novo na aba 3." if mudou else ""))
    ss.salvar_pendente = True


def _remover_agente():
    a = agente(ss.agente_em_edicao)
    if a:
        ss.agentes.remove(a)
        ss.agente_em_edicao = NOVO
        _carregar_edicao()
        ss.msg_agente = ("ok", f"{a.nome} removido.")
        ss.salvar_pendente = True


with aba2:
    st.markdown("Cada agente é um especialista numa regra de negócio. Dê a ele **tabelas** e/ou um **Genie** e "
                "explique o negócio no **prompt**: é isso que ele vai estudar.")
    opcoes = [NOVO] + [a.id for a in ss.agentes]
    if ss.get("agente_em_edicao") not in opcoes:
        ss.agente_em_edicao = opcoes[1] if len(opcoes) > 1 else NOVO
        _carregar_edicao()
    ss.setdefault("ed_nome", "")
    rotulos = {o: rotulo_agente(o) for o in opcoes}  # dicionário fixo: o format_func não lê o estado
    escolhido = st.selectbox("Agente", opcoes, index=opcoes.index(ss.agente_em_edicao), format_func=rotulos.get)
    if escolhido != ss.agente_em_edicao:
        ss.agente_em_edicao = escolhido
        _carregar_edicao()

    st.text_input("Nome do agente", key="ed_nome", placeholder="Ex.: Especialista em Cobrança")
    st.text_area("Prompt de negócio", key="ed_prompt", height=180, placeholder=AJUDA_PROMPT)

    st.markdown("**Tabelas** — navegue no Unity Catalog e adicione ao agente")
    c1, c2, c3 = st.columns([1, 1, 2])
    catalogos = cache("catálogos", servicos().catalogos)
    cat = c1.selectbox("Catálogo", [""] + catalogos, key="pick_cat")
    schemas = cache("schemas", servicos().schemas, cat) if cat else []
    sch = c2.selectbox("Schema", [""] + schemas, key="pick_sch")
    tabs = cache("tabelas", servicos().tabelas, cat, sch) if cat and sch else []
    c3.multiselect("Tabelas", tabs, key="pick_tabs", format_func=lambda t: t.split(".")[-1])
    st.button("➕ Adicionar ao agente", on_click=_adicionar_tabelas, disabled=not ss.get("pick_tabs"))
    ss.setdefault("ed_tabelas", [])
    st.multiselect("Tabelas do agente (clique no × para tirar)", ss.ed_tabelas, key="ed_tabelas")

    genies_liberados = {"": "(nenhum)", **{g.space_id: g.rotulo for g in recursos().genies if g.space_id in ss.sel.genies}}
    if ss.get("ed_genie") and ss.ed_genie not in genies_liberados:
        genies_liberados[ss.ed_genie] = ss.ed_genie
    st.selectbox("Genie do agente", list(genies_liberados), key="ed_genie", format_func=genies_liberados.get,
                 help="Libere Genies na aba 1.")
    st.checkbox("Também usa documentação de regras (PDF/DOCX/MD)", key="ed_docs")
    if ss.get("ed_docs"):
        ident = ss.agente_em_edicao if ss.agente_em_edicao != NOVO else slug(ss.ed_nome)
        pasta = f"/Volumes/{ss.catalogo}/{ss.schema}/documentos/{ident}"
        n = servicos().contar_documentos(ss.catalogo, ss.schema, ident)
        st.info(f"📁 Suba os arquivos em `{pasta}` (Catalog Explorer → Volume → Upload). Encontrados agora: **{n}**.")

    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("💾 Salvar agente", type="primary", on_click=_salvar_agente)
    b2.button("🗑️ Remover", on_click=_remover_agente, disabled=ss.agente_em_edicao == NOVO)
    if ss.get("msg_agente"):
        tipo, texto = ss.pop("msg_agente")
        (st.success if tipo == "ok" else st.error)(texto)
    if ss.pop("salvar_pendente", False):
        salvar()


# ======================================================================= aba 3
def estudar(alvos):
    sel = ss.sel
    if not sel.llm_especialistas:
        st.error("Escolha a IA dos especialistas na aba 1.")
        return
    if any(a.tabelas for a in alvos) and not sel.warehouse_id:
        st.error("Escolha um SQL Warehouse na aba 1 (há agentes com tabelas).")
        return
    sv = servicos()
    llm, fonte = sv.llm(sel.llm_especialistas), sv.fonte_estudo(sel.warehouse_id)
    for a in alvos:
        with st.status(f"Estudando **{a.nome}**…", expanded=True) as status:
            try:
                e = estudar_agente(llm, fonte, nome=a.nome, prompt_negocio=a.prompt_negocio, tabelas=a.tabelas,
                                   genie_space_id=a.genie_space_id, usa_documentos=a.documentos, log=st.write)
                a.estudo = e.to_dict()
                for aviso in e.avisos:
                    st.warning(aviso)
                salvar()
                status.update(label=f"✅ {a.nome}: {len(e.exemplos_sql)} SQL(s) validado(s)", state="complete", expanded=False)
            except Exception as ex:  # noqa: BLE001
                status.update(label=f"❌ {a.nome}: {ex}", state="error")


def _ler_blocos(texto: str, tabelas):
    blocos, atual = {}, None
    for linha in texto.splitlines():
        if linha.startswith("### "):
            atual = linha[4:].strip().lower()
            blocos[atual] = []
        elif atual:
            blocos[atual].append(linha)
    return {t: "\n".join(blocos.get(t, [])).strip() for t in tabelas}


with aba3:
    st.markdown("O sistema lê o esquema e um **perfil dos dados** de cada tabela (nulos, distintos, mínimo/máximo, "
                "valores mais comuns e amostra), junta com o seu prompt e escreve o conhecimento do agente. "
                "Cada SQL de exemplo é **executado**; só os que rodam entram no agente.")
    if not ss.agentes:
        st.info("Crie agentes na aba 2.")
    else:
        pendentes = [a for a in ss.agentes if not a.estudado]
        c1, c2, c3 = st.columns([2, 2, 3])
        if c1.button(f"🔎 Estudar agentes pendentes ({len(pendentes)})", type="primary", disabled=not pendentes):
            estudar(pendentes)
        rotulos_rev = {a.id: rotulo_agente(a.id) for a in ss.agentes}
        rev_id = c3.selectbox("Agente para revisar", list(rotulos_rev), format_func=rotulos_rev.get)
        if c2.button("🔁 Estudar de novo o agente escolhido"):
            estudar([agente(rev_id)])

        a = agente(rev_id)
        st.divider()
        if a and a.estudo:
            e = EstudoAgente.from_dict(a.estudo)
            with st.form(f"rev_{a.id}"):
                st.markdown(f"#### O que **{a.nome}** aprendeu")
                desc = st.text_area("Descrição (é o que o Agente 01 usa para escolher este especialista)", e.descricao, height=90)
                c1, c2 = st.columns(2)
                exemplos = c1.text_area("Perguntas de exemplo (1 por linha)", "\n".join(e.exemplos), height=130)
                testes = c2.text_area("Perguntas de teste do roteamento (1 por linha)", "\n".join(e.perguntas_teste), height=130)
                desc_tabs = st.text_area("O que há em cada tabela", "\n\n".join(f"### {t}\n{d}" for t, d in e.descricao_tabelas.items()),
                                         height=240, disabled=not a.tabelas)
                instr = st.text_area("Instruções e definições de métricas", e.instrucoes, height=150, disabled=not a.tabelas)
                if st.form_submit_button("💾 Salvar revisão", type="primary"):
                    e.descricao = desc.strip()
                    e.exemplos = [l.strip() for l in exemplos.splitlines() if l.strip()]
                    e.perguntas_teste = [l.strip() for l in testes.splitlines() if l.strip()]
                    e.descricao_tabelas = _ler_blocos(desc_tabs, a.tabelas)
                    e.instrucoes = instr.strip()
                    a.estudo = e.to_dict()
                    salvar("Revisão salva")
            if e.validacao_sql:
                st.markdown("#### SQLs de exemplo (executados no estudo)")
                for v in e.validacao_sql:
                    with st.expander(f"{'✅' if v['ok'] else '❌'} {v['pergunta']}"
                                     + (f" · {v.get('linhas', 0)} linha(s)" if v["ok"] else "")):
                        st.code(v["sql"], language="sql")
                        if not v["ok"]:
                            st.error(v.get("erro"))
            for p in e.perfis:
                with st.expander(f"📊 Perfil de {p['tabela']} · {p.get('total_linhas') or '?'} linhas"):
                    if p.get("comentario"):
                        st.caption(p["comentario"])
                    st.dataframe(pd.DataFrame([
                        {"coluna": c["nome"], "tipo": c["tipo"], "nulos %": c.get("nulos_pct"), "distintos": c.get("distintos"),
                         "mín": str(c.get("minimo")), "máx": str(c.get("maximo")),
                         "valores mais comuns": ", ".join(f"{x} ({n})" for x, n in c.get("top_valores") or [])}
                        for c in p["colunas"]
                    ]), use_container_width=True, hide_index=True)
        elif a:
            st.info("⏳ Este agente ainda não foi estudado.")


# ======================================================================= aba 4
with aba4:
    st.markdown("O **agente geral** é o Agente 01: conversa com o usuário, detecta a intenção e aciona o especialista certo. "
                "Aqui ele é montado com tudo o que você escolheu e estudou.")
    sel = ss.sel
    nome_assistente = st.text_input("Nome do assistente", sel.nome_assistente)
    prompt_geral = st.text_area("Prompt geral (opcional)", sel.prompt_geral, height=110,
                                placeholder="Contexto da empresa, tom de voz, regras gerais. Em branco = padrão.")
    c1, c2, c3 = st.columns(3)
    indexar = c1.checkbox("Indexar documentos", True)
    avaliar = c2.checkbox("Avaliar o roteamento", True)
    publicar = c3.checkbox("Registrar e publicar o endpoint", True)
    pendentes = [a.nome for a in ss.agentes if not a.estudado]
    if pendentes:
        st.warning(f"Agentes ainda não estudados (funcionam, mas com menos conhecimento): {', '.join(pendentes)}")

    if st.button("🚀 Gerar agente geral", type="primary", disabled=not ss.agentes):
        sel.nome_assistente, sel.prompt_geral = nome_assistente.strip() or "Assistente", prompt_geral
        sel.catalogo, sel.schema = ss.catalogo, ss.schema
        try:
            dados = gerar_config(sel, ss.agentes)
        except Exception as e:  # noqa: BLE001
            st.error(f"Configuração incompleta: {e}")
            dados = None
        if dados:
            sv = servicos()
            yaml_txt = config_para_yaml(dados)
            anterior = sv.ler(caminho("agents.yaml"))
            if anterior:
                sv.gravar(caminho(f"agents.yaml.{datetime.now():%Y%m%d_%H%M%S}.bak"), anterior)
            sv.gravar(caminho("agents.yaml"), yaml_txt)
            ss.extra["yaml"] = yaml_txt
            st.success(f"Configuração gravada em `{caminho('agents.yaml')}`.")
            if indexar or avaliar or publicar:
                params = {"config_path": caminho("agents.yaml"), "indexar": str(indexar).lower(),
                          "avaliar": str(avaliar).lower(), "publicar": str(publicar).lower()}
                try:
                    job = sv.iniciar_publicacao(params)
                    ss.extra["publicacao"] = {**job, "inicio": time.time(), "status": None}
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
                    st.info("Alternativa: abra `notebooks/04_publicar_job`, informe o parâmetro "
                            f"`config_path = {caminho('agents.yaml')}` e clique em Run all.")
            salvar()

    pub = ss.extra.get("publicacao")
    if pub:
        st.divider()
        st.markdown(f"#### Publicação · [abrir execução do job]({pub.get('url')})")
        st.caption(f"Disparado como: {'você' if pub.get('como') == 'usuario' else 'conta de serviço da app'}")
        if st.button("🔄 Atualizar status") or pub.get("status") is None:
            try:
                pub["status"] = servicos().status_publicacao(pub["run_id"], pub.get("como", "usuario"))
                salvar()
            except Exception as e:  # noqa: BLE001
                st.error(f"Não consegui ler o status: {e}")
        stt = pub.get("status") or {}
        if not stt.get("terminado"):
            minutos = int((time.time() - pub.get("inicio", time.time())) / 60)
            st.info(f"⏳ {stt.get('estado', 'iniciando')} · {minutos} min. Indexar + avaliar + publicar costuma levar de 10 a 25 min.")
        elif stt.get("sucesso"):
            ret = stt.get("retorno") or {}
            st.success("🎉 Agente geral pronto!")
            av = ret.get("avaliacao")
            if av:
                st.metric("Acurácia do roteamento", f"{av['acuracia']:.0%}", f"{av['acertos']}/{av['total']} perguntas", delta_color="off")
                if av.get("erros"):
                    st.markdown("Perguntas que foram para o lugar errado (ajuste a descrição dos agentes na aba 3):")
                    st.dataframe(pd.DataFrame(av["erros"])[["pergunta", "esperado", "obtido", "intencao"]], hide_index=True)
            pubret = ret.get("publicacao") or {}
            if pubret.get("endpoint"):
                ss.extra["endpoint"] = pubret["endpoint"]
                st.markdown(f"**Endpoint:** `{pubret['endpoint']}` · modelo `{pubret.get('nome_modelo')}` v{pubret.get('versao')}")
            if pubret.get("review_app"):
                st.markdown(f"**Review App** (para o time de negócio testar): {pubret['review_app']}")
            st.caption("O endpoint leva alguns minutos para ficar *Ready* depois do job. Depois teste na aba 5.")
        else:
            st.error(f"O job terminou com {stt.get('resultado')}: {stt.get('erro') or stt.get('mensagem')}")
    if ss.extra.get("yaml"):
        st.download_button("⬇️ Baixar agents.yaml", ss.extra["yaml"], file_name="agents.yaml")


# ======================================================================= aba 5
with aba5:
    endpoint = st.text_input("Endpoint do agente", ss.extra.get("endpoint", ""),
                             help="Preenchido automaticamente depois da publicação. Precisa estar Ready.")
    if st.button("🧹 Nova conversa"):
        ss.chat, ss.conversa_id = [], str(uuid.uuid4())
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("detalhes"):
                d = m["detalhes"]
                with st.expander("Detalhes"):
                    st.caption(f"Intenção: {d.get('intencao')} · agentes: {', '.join(d.get('especialistas') or []) or '—'} · "
                               f"{d.get('tokens', 0)} tokens · US$ {d.get('custo_usd', 0):.5f}")
                    for c in d.get("consultas_sql") or []:
                        st.code(c.get("sql") or "", language="sql")
                    for esp, dados in (d.get("dados") or {}).items():
                        if dados.get("linhas"):
                            st.dataframe(pd.DataFrame(dados["linhas"], columns=dados["colunas"]), hide_index=True)
    pergunta = st.chat_input("Pergunte algo ao agente geral…", disabled=not endpoint)
    if pergunta:
        ss.chat.append({"role": "user", "content": pergunta})
        historico = [{"role": m["role"], "content": m["content"]} for m in ss.chat]
        with st.spinner("Pensando…"):
            try:
                texto, detalhes = extrair_resposta(servicos().conversar(endpoint, historico, ss.conversa_id))
            except Exception as e:  # noqa: BLE001
                texto, detalhes = f"⚠️ Erro ao chamar o endpoint: {e}", {}
        ss.chat.append({"role": "assistant", "content": texto, "detalhes": detalhes})
        ss.gasto_chat["tokens"] += detalhes.get("tokens", 0)
        ss.gasto_chat["usd"] += detalhes.get("custo_usd", 0.0)
        st.rerun()
