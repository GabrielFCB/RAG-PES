# RAG-PES

Este repositório contém o modelo de uma aplicação baseada no conceito de Retriever-Augmented Generation (RAG), especializada na metodologia PES (Planejamento Estratégico Situacional), composta por duas partes principais:

- **Cliente - TestClientRAG**: Desenvolvido em Node.js.
- **Servidor - TestServerRAG**: Desenvolvido em Python, utilizando as bibliotecas LangChain, OpenAI e Pinecone.

## Requisitos

Certifique-se de ter as seguintes ferramentas instaladas no seu sistema:

- [Node.js](https://nodejs.org/)
- [Python](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/)

## Configuração

### Variáveis de Ambiente

O servidor utiliza chaves privadas para acessar os serviços da LangChain, OpenAI e Pinecone. Estas chaves devem ser configuradas como variáveis de ambiente no arquivo `.env`.

Um exemplo de arquivo `.env` é fornecido como `.env.example`. Para começar, copie este arquivo e insira as chaves apropriadas:

```bash
cp TestServerRAG/.env.example TestServerRAG/.env
```

Abra o arquivo .env e insira os valores das suas chaves:

LANGCHAIN_API_KEY=<sua_chave_langchain>
OPENAI_API_KEY=<sua_chave_openai>
PINECONE_API_KEY=<sua_chave_pinecone>
⚠️ Nota: O arquivo .env está listado no .gitignore e não será versionado no repositório.

### Configuração do Cliente

Navegue até a pasta do cliente e instale as dependências::

```bash
cd client
npm install
```

Inicie o cliente:

```bash
npm start
```

Configuração do Servidor
Navegue até a pasta do servidor e instale as dependências:

```bash
cd server
pip install -r requirements.txt
```

Execute o servidor:

```bash
python /app/server.py
```

⚠️ O servidor verificará se as variáveis de ambiente necessárias estão configuradas. Caso contrário, um erro será gerado:
"As chaves 'LANGCHAIN_API_KEY', 'OPENAI_API_KEY' e 'PINECONE_API_KEY' precisam ser definidas como variáveis de ambiente."

## Rodando com Docker

A aplicação pode ser executada utilizando **Docker Compose** para facilitar a configuração e gerenciamento de contêineres.

### Passos para Iniciar a Aplicação

1. Certifique-se de que o arquivo `.env` esteja configurado corretamente na pasta `server/`.
2. Execute o comando abaixo para criar e iniciar os contêineres:
   docker compose up -d --build
3. Para fechar a aplicação, rode
   docker compose down
