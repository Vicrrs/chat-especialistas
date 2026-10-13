-- projeto (configuração, documentos, LLMOps, modelo)
CREATE SCHEMA IF NOT EXISTS main.chat_especialistas;
CREATE VOLUME IF NOT EXISTS main.chat_especialistas.config;
CREATE VOLUME IF NOT EXISTS main.chat_especialistas.documentos;
CREATE VOLUME IF NOT EXISTS main.chat_especialistas.llmops;
GRANT USE CATALOG ON CATALOG main TO `3507dcde-cb1a-490a-bcf6-b09261623384`;
GRANT ALL PRIVILEGES ON SCHEMA main.chat_especialistas TO `3507dcde-cb1a-490a-bcf6-b09261623384`;

-- tabelas que os agentes vão estudar e consultar
GRANT USE CATALOG ON CATALOG <catalogo> TO `3507dcde-cb1a-490a-bcf6-b09261623384`;
GRANT USE SCHEMA, SELECT ON SCHEMA <catalogo>.<schema_das_tabelas> TO `3507dcde-cb1a-490a-bcf6-b09261623384`;

