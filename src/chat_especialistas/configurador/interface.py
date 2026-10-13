"""Interface do assistente de configuração (ipywidgets) usada em notebooks/00_COMECE_AQUI.

Fluxo em 4 abas:
  1. Modelos e Genies  – escolhe LLMs, embeddings, warehouse, Genie Spaces e onde gravar o projeto
  2. Agentes           – cria os agentes: nome, prompt de negócio, tabelas, Genie, documentação
  3. Estudar           – o sistema estuda as tabelas de cada agente; você revisa e ajusta
  4. Agente geral      – gera o agents.yaml, indexa documentos, avalia o roteamento e publica
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import ipywidgets as W

from .descoberta import Recursos, descobrir, listar_catalogos, listar_schemas, listar_tabelas
from .estudo import EstudoAgente, estudar_agente
from .gerador import (
    PRECO_LLAMA_70B,
    AgenteRascunho,
    Selecao,
    carregar_estado,
    gerar_config,
    salvar_config,
    salvar_estado,
    slug,
)

LARG = W.Layout(width="760px")
LARG_CURTA = W.Layout(width="370px")
ESTILO = {"description_width": "190px"}
PRECO_PADRAO = {"input": 1.00, "output": 3.00}

AJUDA_PROMPT = (
    "Explique para o agente, como faria para um analista novo:\n"
    "• qual área/regra de negócio ele atende e quem vai perguntar;\n"
    "• o que existe em cada tabela e o significado das colunas importantes;\n"
    "• como o negócio calcula as métricas (ex.: 'inadimplente = atraso > 30 dias');\n"
    "• filtros padrão, armadilhas, termos que os usuários usam."
)


def _html(texto: str) -> W.HTML:
    return W.HTML(texto)


class Assistente:
    def __init__(
        self,
        w,
        fonte_estudo,
        criar_llm: Callable[[str], Any],
        raiz: Path,
        *,
        ingerir: Optional[Callable] = None,
        avaliar: Optional[Callable] = None,
        publicar: Optional[Callable] = None,
        contar_documentos: Optional[Callable[[str], int]] = None,
        recursos: Optional[Recursos] = None,
    ):
        self.w = w
        self.fonte = fonte_estudo
        self.criar_llm = criar_llm
        self.raiz = Path(raiz)
        self.caminho_config = self.raiz / "config" / "agents.yaml"
        self.caminho_estado = self.raiz / "config" / "assistente_estado.json"
        self.ingerir, self.avaliar, self.publicar = ingerir, avaliar, publicar
        self.contar_documentos = contar_documentos or (lambda _: 0)
        self.sel, self.agentes = carregar_estado(self.caminho_estado)
        self.recursos = recursos or descobrir(w)
        self._cache_catalogos: Optional[List[str]] = None
        self.ultimo_resultado: Dict[str, Any] = {}
        self._montar()

    # ================================================================ util
    def _salvar_estado(self):
        try:
            salvar_estado(self.caminho_estado, self.sel, self.agentes)
        except Exception as e:  # noqa: BLE001
            print(f"(não foi possível salvar o estado do assistente: {e})")

    def _catalogos(self) -> List[str]:
        if self._cache_catalogos is None:
            try:
                self._cache_catalogos = listar_catalogos(self.w)
            except Exception as e:  # noqa: BLE001
                self.recursos.avisos.append(f"Não foi possível listar catálogos: {e}")
                self._cache_catalogos = []
        return self._cache_catalogos

    def _agente(self, id_: Optional[str]) -> Optional[AgenteRascunho]:
        return next((a for a in self.agentes if a.id == id_), None)

    def _opcoes_agentes(self):
        return [(f"{'✅' if a.estudado else '⏳'} {a.nome}  ·  {a.id}", a.id) for a in self.agentes]

    def _atualizar_listas_agentes(self, selecionar: Optional[str] = None):
        ops = self._opcoes_agentes()
        for dd in (self.lista_agentes, self.dd_revisar):
            atual = selecionar or dd.value
            dd.options = ops
            ids = [v for _, v in ops]
            dd.value = atual if atual in ids else (ids[0] if ids else None)

    # ============================================================ montagem
    def _montar(self):
        self.aba1 = self._montar_aba_modelos()
        self.aba2 = self._montar_aba_agentes()
        self.aba3 = self._montar_aba_estudo()
        self.aba4 = self._montar_aba_geral()
        self.tabs = W.Tab(children=[self.aba1, self.aba2, self.aba3, self.aba4])
        for i, t in enumerate(["1 · Modelos e Genies", "2 · Agentes", "3 · Estudar tabelas", "4 · Agente geral"]):
            self.tabs.set_title(i, t)
        self.ui = W.VBox([
            _html("<h2 style='margin:0'>🤖 Assistente de configuração · Chat Multi-Especialistas</h2>"
                  "<p>Escolha as IAs e Genies, monte os agentes com suas tabelas, deixe o sistema estudar os "
                  "dados e gere o agente geral.</p>"),
            self.tabs,
        ])
        self._atualizar_listas_agentes()

    # --------------------------------------------------------------- aba 1
    def _montar_aba_modelos(self):
        r, s = self.recursos, self.sel
        chat = [(m.rotulo, m.nome) for m in r.llms_chat]
        nomes_chat = [v for _, v in chat]
        padrao = next((n for n in nomes_chat if "llama-3-3-70b" in n or "llama-4" in n), nomes_chat[0] if nomes_chat else None)

        self.dd_roteador = W.Dropdown(options=chat, value=s.llm_roteador if s.llm_roteador in nomes_chat else padrao,
                                      description="LLM do Agente 01 (roteador)", style=ESTILO, layout=LARG)
        self.dd_especialistas = W.Dropdown(options=chat, value=s.llm_especialistas if s.llm_especialistas in nomes_chat else padrao,
                                           description="LLM dos especialistas", style=ESTILO, layout=LARG)
        emb = [("(nenhum — agentes sem documentos)", "")] + [(m.rotulo, m.nome) for m in r.embeddings]
        nomes_emb = [v for _, v in emb]
        self.dd_embedding = W.Dropdown(
            options=emb,
            value=s.embedding if s.embedding and s.embedding in nomes_emb else next((v for v in nomes_emb if "gte" in v or "bge" in v), ""),
            description="Embeddings (documentos)", style=ESTILO, layout=LARG)
        whs = [("(nenhum)", "")] + [(x.rotulo, x.id) for x in r.warehouses]
        ids_wh = [v for _, v in whs]
        self.dd_warehouse = W.Dropdown(options=whs, value=s.warehouse_id if s.warehouse_id and s.warehouse_id in ids_wh else (ids_wh[1] if len(ids_wh) > 1 else ""),
                                       description="SQL Warehouse", style=ESTILO, layout=LARG)
        genies = [(g.rotulo, g.space_id) for g in r.genies]
        self.sm_genies = W.SelectMultiple(options=genies, value=tuple(g for g in s.genies if g in dict((v, k) for k, v in genies)),
                                          rows=min(10, max(3, len(genies))), description="Genie Spaces liberados",
                                          style=ESTILO, layout=LARG)
        self.btn_agente_por_genie = W.Button(description="Criar um agente para cada Genie selecionado", icon="plus",
                                             layout=W.Layout(width="380px"))
        self.btn_agente_por_genie.on_click(self._criar_agentes_genie)

        self.cb_catalogo = W.Combobox(options=self._catalogos(), value=s.catalogo, ensure_option=False,
                                      description="Catálogo do projeto", style=ESTILO, layout=LARG_CURTA)
        self.txt_schema = W.Text(value=s.schema, description="Schema do projeto", style=ESTILO, layout=LARG_CURTA)
        self.txt_vs = W.Text(value=s.vector_search_endpoint, description="Endpoint Vector Search", style=ESTILO, layout=LARG_CURTA)
        self.tb_auth = W.ToggleButtons(options=[("Credencial do endpoint", "servico"), ("Permissão de cada usuário", "usuario")],
                                       value=s.autenticacao, description="Acesso aos dados", style=ESTILO)

        def preco_inicial(modelo, campo):
            p = s.precos.get(modelo) or (PRECO_LLAMA_70B if "llama-3-3-70b" in (modelo or "") else PRECO_PADRAO)
            return float(p[campo])

        self.fp_rot_in = W.FloatText(value=preco_inicial(self.dd_roteador.value, "input"), description="Roteador: US$/1M entrada", style=ESTILO, layout=LARG_CURTA)
        self.fp_rot_out = W.FloatText(value=preco_inicial(self.dd_roteador.value, "output"), description="Roteador: US$/1M saída", style=ESTILO, layout=LARG_CURTA)
        self.fp_esp_in = W.FloatText(value=preco_inicial(self.dd_especialistas.value, "input"), description="Especialistas: US$/1M entrada", style=ESTILO, layout=LARG_CURTA)
        self.fp_esp_out = W.FloatText(value=preco_inicial(self.dd_especialistas.value, "output"), description="Especialistas: US$/1M saída", style=ESTILO, layout=LARG_CURTA)

        def ao_trocar_modelo(campos):
            def f(mudanca):
                campos[0].value = preco_inicial(mudanca["new"], "input")
                campos[1].value = preco_inicial(mudanca["new"], "output")
            return f

        self.dd_roteador.observe(ao_trocar_modelo((self.fp_rot_in, self.fp_rot_out)), names="value")
        self.dd_especialistas.observe(ao_trocar_modelo((self.fp_esp_in, self.fp_esp_out)), names="value")

        for wdg in (self.dd_roteador, self.dd_especialistas, self.dd_embedding, self.dd_warehouse, self.sm_genies,
                    self.cb_catalogo, self.txt_schema, self.txt_vs, self.tb_auth,
                    self.fp_rot_in, self.fp_rot_out, self.fp_esp_in, self.fp_esp_out):
            wdg.observe(lambda _: self._coletar_selecao(), names="value")

        avisos = "".join(f"<li>⚠️ {a}</li>" for a in r.avisos)
        resumo = _html(
            f"<p>Encontrei no workspace: <b>{len(r.llms_chat)}</b> LLMs de chat, <b>{len(r.embeddings)}</b> modelos de "
            f"embeddings, <b>{len(r.genies)}</b> Genie Spaces e <b>{len(r.warehouses)}</b> SQL Warehouses "
            f"(apenas o que você tem permissão para usar).</p>" + (f"<ul>{avisos}</ul>" if avisos else "")
        )
        return W.VBox([
            resumo,
            _html("<h4>IAs</h4>"), self.dd_roteador, self.dd_especialistas, self.dd_embedding,
            _html("<h4>Dados</h4>"), self.dd_warehouse, self.tb_auth,
            _html("<h4>Genie Spaces</h4><p>Selecione os Genies que os agentes poderão usar (Ctrl/Cmd para vários).</p>"),
            self.sm_genies, self.btn_agente_por_genie,
            _html("<h4>Onde gravar o projeto</h4><p>Catálogo/schema para documentos, índice, LLMOps e o modelo registrado.</p>"),
            W.HBox([self.cb_catalogo, self.txt_schema]), self.txt_vs,
            _html("<h4>Preço dos modelos (LLMOps)</h4><p>US$ por 1 milhão de tokens. Confirme no seu contrato Databricks.</p>"),
            W.HBox([self.fp_rot_in, self.fp_rot_out]), W.HBox([self.fp_esp_in, self.fp_esp_out]),
        ])

    def _coletar_selecao(self) -> Selecao:
        s = self.sel
        s.llm_roteador = self.dd_roteador.value or ""
        s.llm_especialistas = self.dd_especialistas.value or ""
        s.embedding = self.dd_embedding.value or ""
        s.warehouse_id = self.dd_warehouse.value or ""
        s.genies = list(self.sm_genies.value)
        s.catalogo = (self.cb_catalogo.value or "main").strip()
        s.schema = (self.txt_schema.value or "chat_especialistas").strip()
        s.vector_search_endpoint = (self.txt_vs.value or "vs_chat_especialistas").strip()
        s.autenticacao = self.tb_auth.value
        precos = {}
        if s.llm_roteador:
            precos[s.llm_roteador] = {"input": float(self.fp_rot_in.value), "output": float(self.fp_rot_out.value)}
        if s.llm_especialistas:
            precos[s.llm_especialistas] = {"input": float(self.fp_esp_in.value), "output": float(self.fp_esp_out.value)}
        s.precos = precos
        if hasattr(self, "txt_nome_assistente"):
            s.nome_assistente = self.txt_nome_assistente.value.strip() or "Assistente"
            s.prompt_geral = self.ta_prompt_geral.value
        if hasattr(self, "dd_genie"):
            self._atualizar_opcoes_genie()
        self._salvar_estado()
        return s

    def _criar_agentes_genie(self, _=None):
        por_id = {g.space_id: g for g in self.recursos.genies}
        criados = 0
        for gid in self.sm_genies.value:
            if any(a.genie_space_id == gid and not a.tabelas for a in self.agentes):
                continue
            g = por_id[gid]
            self.agentes.append(AgenteRascunho(nome=g.titulo, prompt_negocio=g.descricao, genie_space_id=gid))
            criados += 1
        self._salvar_estado()
        self._atualizar_listas_agentes()
        self.msg_agente.value = f"<p>✅ {criados} agente(s) criado(s) a partir de Genies. Revise o prompt na aba 2.</p>"

    # --------------------------------------------------------------- aba 2
    def _montar_aba_agentes(self):
        self.lista_agentes = W.Select(options=[], rows=10, description="Agentes", style={"description_width": "70px"},
                                      layout=W.Layout(width="360px"))
        self.btn_novo = W.Button(description="Novo agente", icon="plus", button_style="primary")
        self.btn_remover = W.Button(description="Remover", icon="trash", button_style="danger")
        self.btn_novo.on_click(self._novo_agente)
        self.btn_remover.on_click(self._remover_agente)

        self.txt_nome = W.Text(description="Nome do agente", placeholder="Ex.: Especialista em Cobrança", style=ESTILO, layout=LARG)
        self.ta_prompt = W.Textarea(description="Prompt de negócio", placeholder=AJUDA_PROMPT, style=ESTILO,
                                    layout=W.Layout(width="760px", height="170px"))

        self.dd_cat = W.Dropdown(options=[""] + self._catalogos(), description="Catálogo", style={"description_width": "70px"}, layout=W.Layout(width="250px"))
        self.dd_sch = W.Dropdown(options=[""], description="Schema", style={"description_width": "60px"}, layout=W.Layout(width="250px"))
        self.sm_tab = W.SelectMultiple(options=[], rows=7, description="Tabelas", style={"description_width": "70px"}, layout=W.Layout(width="510px"))
        self.btn_add_tab = W.Button(description="Adicionar ao agente ➜", icon="plus")
        self.sm_tab_agente = W.SelectMultiple(options=[], rows=7, description="Do agente", style={"description_width": "70px"}, layout=W.Layout(width="420px"))
        self.btn_rem_tab = W.Button(description="Tirar do agente", icon="minus")
        self.dd_cat.observe(self._ao_mudar_catalogo, names="value")
        self.dd_sch.observe(self._ao_mudar_schema, names="value")
        self.btn_add_tab.on_click(self._adicionar_tabelas)
        self.btn_rem_tab.on_click(self._remover_tabelas)

        self.dd_genie = W.Dropdown(options=[("(nenhum)", "")], description="Genie do agente", style=ESTILO, layout=LARG)
        self.cb_docs = W.Checkbox(description="Também usa documentação de regras (PDF/DOCX/MD)", indent=False)
        self.html_docs = _html("")
        self.cb_docs.observe(lambda _: self._mostrar_pasta_docs(), names="value")
        self.btn_salvar_agente = W.Button(description="Salvar agente", icon="save", button_style="success")
        self.btn_salvar_agente.on_click(self._salvar_agente)
        self.msg_agente = _html("")
        self.lista_agentes.observe(lambda _: self._carregar_agente(), names="value")
        self._atualizar_opcoes_genie()

        return W.VBox([
            _html("<p>Cada agente é um especialista em uma regra de negócio. Dê a ele tabelas e/ou um Genie e "
                  "explique o negócio no prompt — é isso que ele vai estudar.</p>"),
            W.HBox([self.lista_agentes, W.VBox([self.btn_novo, self.btn_remover])]),
            _html("<hr><h4>Editar agente</h4>"),
            self.txt_nome, self.ta_prompt,
            _html("<b>Tabelas</b> (navegue no Unity Catalog e adicione):"),
            W.HBox([self.dd_cat, self.dd_sch]),
            W.HBox([self.sm_tab, W.VBox([self.btn_add_tab, self.btn_rem_tab]), self.sm_tab_agente]),
            self.dd_genie, self.cb_docs, self.html_docs,
            W.HBox([self.btn_salvar_agente, self.msg_agente]),
        ])

    def _atualizar_opcoes_genie(self):
        por_id = {g.space_id: g for g in self.recursos.genies}
        ids = list(self.sm_genies.value) if hasattr(self, "sm_genies") else []
        atual = self.dd_genie.value
        self.dd_genie.options = [("(nenhum)", "")] + [(por_id[i].rotulo, i) for i in ids if i in por_id]
        self.dd_genie.value = atual if atual in [v for _, v in self.dd_genie.options] else ""

    def _ao_mudar_catalogo(self, _=None):
        cat = self.dd_cat.value
        try:
            self.dd_sch.options = [""] + (listar_schemas(self.w, cat) if cat else [])
        except Exception as e:  # noqa: BLE001
            self.msg_agente.value = f"<p>⚠️ {e}</p>"

    def _ao_mudar_schema(self, _=None):
        cat, sch = self.dd_cat.value, self.dd_sch.value
        try:
            self.sm_tab.options = listar_tabelas(self.w, cat, sch) if cat and sch else []
        except Exception as e:  # noqa: BLE001
            self.msg_agente.value = f"<p>⚠️ {e}</p>"

    def _adicionar_tabelas(self, _=None):
        self.sm_tab_agente.options = sorted(set(self.sm_tab_agente.options) | set(self.sm_tab.value))

    def _remover_tabelas(self, _=None):
        self.sm_tab_agente.options = [t for t in self.sm_tab_agente.options if t not in self.sm_tab_agente.value]

    def _mostrar_pasta_docs(self):
        if not self.cb_docs.value:
            self.html_docs.value = ""
            return
        ident = self._agente(self.lista_agentes.value).id if self._agente(self.lista_agentes.value) else slug(self.txt_nome.value)
        pasta = f"/Volumes/{self.sel.catalogo}/{self.sel.schema}/documentos/{ident}"
        n = self.contar_documentos(pasta)
        self.html_docs.value = (
            f"<p>📁 Suba os arquivos em <code>{pasta}</code> (Catalog Explorer → Volume → Upload). "
            f"Encontrados agora: <b>{n}</b>.</p>"
        )

    def _novo_agente(self, _=None):
        self.lista_agentes.value = None
        self.txt_nome.value, self.ta_prompt.value = "", ""
        self.sm_tab_agente.options = []
        self.dd_genie.value, self.cb_docs.value = "", False
        self.msg_agente.value = "<p>Preencha e clique em <b>Salvar agente</b>.</p>"

    def _carregar_agente(self):
        a = self._agente(self.lista_agentes.value)
        if a is None:
            return
        self.txt_nome.value, self.ta_prompt.value = a.nome, a.prompt_negocio
        self.sm_tab_agente.options = list(a.tabelas)
        if a.genie_space_id and a.genie_space_id not in self.sm_genies.value:
            self.sm_genies.value = tuple(self.sm_genies.value) + (a.genie_space_id,)
        self._atualizar_opcoes_genie()
        self.dd_genie.value = a.genie_space_id or ""
        self.cb_docs.value = a.documentos
        self._mostrar_pasta_docs()

    def _salvar_agente(self, _=None):
        nome = self.txt_nome.value.strip()
        tabelas = list(self.sm_tab_agente.options)
        genie = self.dd_genie.value or None
        if not nome:
            self.msg_agente.value = "<p>⚠️ Dê um nome ao agente.</p>"
            return
        if not (tabelas or genie or self.cb_docs.value):
            self.msg_agente.value = "<p>⚠️ O agente precisa de tabelas, um Genie ou documentação.</p>"
            return
        a = self._agente(self.lista_agentes.value)
        if a is None:
            a = AgenteRascunho(nome=nome)
            if self._agente(a.id):
                a.id = f"{a.id}_{len(self.agentes) + 1}"
            self.agentes.append(a)
        mudou_fonte = sorted(tabelas) != a.tabelas or genie != a.genie_space_id or self.ta_prompt.value != a.prompt_negocio
        a.nome, a.prompt_negocio = nome, self.ta_prompt.value
        a.tabelas = sorted({t.lower() for t in tabelas})
        a.genie_space_id, a.documentos = genie, bool(self.cb_docs.value)
        if mudou_fonte and a.estudo:
            a.estudo = None  # tabelas/prompt mudaram: precisa estudar de novo
        self._salvar_estado()
        self._atualizar_listas_agentes(selecionar=a.id)
        self.msg_agente.value = f"<p>✅ Agente <b>{a.nome}</b> (<code>{a.id}</code>) salvo.</p>"

    def _remover_agente(self, _=None):
        a = self._agente(self.lista_agentes.value)
        if a:
            self.agentes.remove(a)
            self._salvar_estado()
            self._atualizar_listas_agentes()
            self.msg_agente.value = f"<p>🗑️ {a.nome} removido.</p>"

    # --------------------------------------------------------------- aba 3
    def _montar_aba_estudo(self):
        self.btn_estudar_todos = W.Button(description="Estudar agentes pendentes", icon="search", button_style="primary",
                                          layout=W.Layout(width="260px"))
        self.btn_estudar_sel = W.Button(description="Estudar (de novo) o agente abaixo", icon="refresh", layout=W.Layout(width="290px"))
        self.btn_estudar_todos.on_click(lambda _: self._estudar(apenas_pendentes=True))
        self.btn_estudar_sel.on_click(lambda _: self._estudar(ids=[self.dd_revisar.value]))
        self.out_estudo = W.Output(layout=W.Layout(border="1px solid #ddd", max_height="300px", overflow="auto"))

        self.dd_revisar = W.Dropdown(options=[], description="Revisar agente", style=ESTILO, layout=LARG)
        self.ta_descricao = W.Textarea(description="Descrição (para o roteador)", style=ESTILO, layout=W.Layout(width="760px", height="80px"))
        self.ta_exemplos = W.Textarea(description="Perguntas de exemplo (1/linha)", style=ESTILO, layout=W.Layout(width="760px", height="90px"))
        self.ta_testes = W.Textarea(description="Perguntas de teste (1/linha)", style=ESTILO, layout=W.Layout(width="760px", height="90px"))
        self.ta_desc_tabelas = W.Textarea(description="O que há nas tabelas", style=ESTILO, layout=W.Layout(width="760px", height="220px"))
        self.ta_instrucoes = W.Textarea(description="Instruções / métricas", style=ESTILO, layout=W.Layout(width="760px", height="140px"))
        self.html_sql = _html("")
        self.btn_salvar_revisao = W.Button(description="Salvar revisão", icon="save", button_style="success")
        self.btn_salvar_revisao.on_click(self._salvar_revisao)
        self.msg_revisao = _html("")
        self.dd_revisar.observe(lambda _: self._carregar_revisao(), names="value")

        return W.VBox([
            _html("<p>O sistema lê o esquema e um perfil dos dados de cada tabela (nulos, distintos, mínimos/máximos, "
                  "valores mais comuns, amostra), junta com o seu prompt de negócio e escreve o conhecimento do agente. "
                  "Cada SQL de exemplo é <b>executado</b>; só os que rodam entram no agente.</p>"),
            W.HBox([self.btn_estudar_todos, self.btn_estudar_sel]),
            self.out_estudo,
            _html("<hr><h4>Revisar o que o agente aprendeu</h4>"),
            self.dd_revisar, self.ta_descricao, self.ta_exemplos, self.ta_testes,
            self.ta_desc_tabelas, self.ta_instrucoes, self.html_sql,
            W.HBox([self.btn_salvar_revisao, self.msg_revisao]),
        ])

    def _estudar(self, ids: Optional[List[str]] = None, apenas_pendentes: bool = False):
        self._coletar_selecao()
        alvo = [a for a in self.agentes if (ids is None or a.id in ids) and not (apenas_pendentes and a.estudado)]
        with self.out_estudo:
            if not self.sel.llm_especialistas:
                print("⚠️ Escolha o LLM dos especialistas na aba 1.")
                return
            if not alvo:
                print("Nada para estudar (todos os agentes já foram estudados).")
                return
            llm = self.criar_llm(self.sel.llm_especialistas)
            for a in alvo:
                try:
                    estudo = estudar_agente(
                        llm, self.fonte, nome=a.nome, prompt_negocio=a.prompt_negocio, tabelas=a.tabelas,
                        genie_space_id=a.genie_space_id, usa_documentos=a.documentos,
                    )
                    a.estudo = estudo.to_dict()
                    for aviso in estudo.avisos:
                        print(f"   ⚠️ {aviso}")
                except Exception as e:  # noqa: BLE001
                    print(f"❌ Falha ao estudar '{a.nome}': {e}")
                    traceback.print_exc()
                self._salvar_estado()
        self._atualizar_listas_agentes(selecionar=alvo[-1].id if alvo else None)
        self._carregar_revisao()

    def _carregar_revisao(self):
        a = self._agente(self.dd_revisar.value)
        e = EstudoAgente.from_dict(a.estudo) if a and a.estudo else None
        if e is None:
            for t in (self.ta_descricao, self.ta_exemplos, self.ta_testes, self.ta_desc_tabelas, self.ta_instrucoes):
                t.value = ""
            self.html_sql.value = "<p>⏳ Este agente ainda não foi estudado.</p>" if a else ""
            return
        self.ta_descricao.value = e.descricao
        self.ta_exemplos.value = "\n".join(e.exemplos)
        self.ta_testes.value = "\n".join(e.perguntas_teste)
        self.ta_desc_tabelas.value = "\n\n".join(f"### {t}\n{d}" for t, d in e.descricao_tabelas.items())
        self.ta_instrucoes.value = e.instrucoes
        linhas = "".join(
            f"<tr><td>{'✅' if v['ok'] else '❌'}</td><td>{v['pergunta']}</td>"
            f"<td><pre style='margin:0;white-space:pre-wrap'>{v['sql']}</pre></td>"
            f"<td>{v.get('linhas', 0) if v['ok'] else (v.get('erro') or '')}</td></tr>"
            for v in e.validacao_sql
        )
        self.html_sql.value = (
            "<b>SQLs de exemplo (executados no estudo)</b>"
            "<table style='font-size:12px'><tr><th></th><th>Pergunta</th><th>SQL</th><th>Linhas / erro</th></tr>"
            f"{linhas}</table>" if linhas else "<p>Sem SQL de exemplo (agente sem tabelas).</p>"
        )

    def _salvar_revisao(self, _=None):
        a = self._agente(self.dd_revisar.value)
        if not a or not a.estudo:
            self.msg_revisao.value = "<p>⚠️ Estude o agente antes.</p>"
            return
        e = EstudoAgente.from_dict(a.estudo)
        e.descricao = self.ta_descricao.value.strip()
        e.exemplos = [l.strip() for l in self.ta_exemplos.value.splitlines() if l.strip()]
        e.perguntas_teste = [l.strip() for l in self.ta_testes.value.splitlines() if l.strip()]
        e.instrucoes = self.ta_instrucoes.value.strip()
        e.descricao_tabelas = _ler_blocos(self.ta_desc_tabelas.value, a.tabelas)
        a.estudo = e.to_dict()
        self._salvar_estado()
        self.msg_revisao.value = "<p>✅ Revisão salva.</p>"

    # --------------------------------------------------------------- aba 4
    def _montar_aba_geral(self):
        self.txt_nome_assistente = W.Text(value=self.sel.nome_assistente, description="Nome do assistente", style=ESTILO, layout=LARG)
        self.ta_prompt_geral = W.Textarea(value=self.sel.prompt_geral, description="Prompt geral (opcional)",
                                          placeholder="Contexto da empresa, tom de voz, regras gerais. Em branco = padrão.",
                                          style=ESTILO, layout=W.Layout(width="760px", height="110px"))
        self.cb_indexar = W.Checkbox(value=True, description="Indexar documentos (agentes com documentação)", indent=False)
        self.cb_avaliar = W.Checkbox(value=True, description="Avaliar o roteamento com as perguntas de teste", indent=False)
        self.cb_publicar = W.Checkbox(value=True, description="Registrar e publicar o endpoint (Model Serving + Review App)", indent=False)
        self.btn_gerar = W.Button(description="Gerar agente geral", icon="rocket", button_style="primary", layout=W.Layout(width="260px"))
        self.btn_gerar.on_click(self._gerar)
        self.out_gerar = W.Output(layout=W.Layout(border="1px solid #ddd", max_height="420px", overflow="auto"))
        return W.VBox([
            _html("<p>O agente geral é o <b>Agente 01</b>: conversa com o usuário, detecta a intenção e aciona o "
                  "especialista certo. Aqui o projeto é montado com tudo o que você escolheu e estudou.</p>"),
            self.txt_nome_assistente, self.ta_prompt_geral,
            self.cb_indexar, self.cb_avaliar, self.cb_publicar,
            self.btn_gerar, self.out_gerar,
        ])

    def _gerar(self, _=None):
        from ..config import carregar_config

        sel = self._coletar_selecao()
        with self.out_gerar:
            self.out_gerar.clear_output()
            pendentes = [a.nome for a in self.agentes if not a.estudado]
            if pendentes:
                print(f"⚠️ Agentes ainda não estudados (vão funcionar, mas com menos conhecimento): {pendentes}")
            try:
                dados = gerar_config(sel, self.agentes)
            except Exception as e:  # noqa: BLE001
                print(f"❌ Configuração incompleta: {e}")
                return
            backup = salvar_config(dados, self.caminho_config)
            self._salvar_estado()
            print(f"💾 Configuração gravada em {self.caminho_config}" + (f" (anterior: {backup.name})" if backup else ""))
            cfg = carregar_config(self.caminho_config)
            resultado: Dict[str, Any] = {"config": str(self.caminho_config)}
            try:
                if self.cb_indexar.value and cfg.usa_documentos and self.ingerir:
                    print("\n📚 Indexando documentos…")
                    resultado["documentos"] = self.ingerir(cfg)
                if self.cb_avaliar.value and self.avaliar:
                    print("\n🧭 Avaliando o detector de intenção…")
                    resultado["avaliacao"] = self.avaliar(cfg)
                if self.cb_publicar.value and self.publicar:
                    print("\n📦 Registrando e publicando…")
                    resultado["publicacao"] = self.publicar(cfg)
                print("\n🎉 Agente geral pronto." if self.cb_publicar.value else "\n✅ Configuração gerada.")
            except Exception as e:  # noqa: BLE001
                print(f"❌ Falhou: {e}")
                traceback.print_exc()
            self.ultimo_resultado = resultado

    # ============================================================== exibir
    def mostrar(self):
        from IPython.display import display

        display(self.ui)
        return self


def _ler_blocos(texto: str, tabelas: List[str]) -> Dict[str, str]:
    """Converte '### tabela\\ntexto' em {tabela: texto}."""
    blocos: Dict[str, List[str]] = {}
    atual = None
    for linha in texto.splitlines():
        if linha.startswith("### "):
            atual = linha[4:].strip().lower()
            blocos[atual] = []
        elif atual:
            blocos[atual].append(linha)
    return {t: "\n".join(blocos.get(t, [])).strip() for t in tabelas}
