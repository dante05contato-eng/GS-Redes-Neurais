# Segurança Pública Inteligente em São Paulo: Classificação de Ocorrências com MLP

Projeto desenvolvido para a disciplina de Aprendizado Profundo com Redes Neurais Convolucionais. A solução utiliza uma rede neural do tipo Multilayer Perceptron (MLP) para classificar registros de ocorrências policiais da capital paulista entre Roubo e Furto, com base em atributos contextuais de espaço e tempo.

Acesse a aplicação web online: [COLE_AQUI_O_LINK_DO_STREAMLIT_CLOUD]



# Fonte e Preparação dos Dados
* Fonte Pública: Portal de Estatísticas da Secretaria da Segurança Pública do Estado de São Paulo (**SSP-SP**) — https://www.ssp.sp.gov.br/estatistica/consultas
* Recorte: Ocorrências registradas no município de **São Paulo (Capital)** (amostra estruturada de 15.000 registros).
* Atributos de Entrada ($X$): Região de São Paulo, Período do Dia, Dia da Semana e Tipo de Local.
* Variável-Alvo ($y$): Classificação binária (`0` = Furto | `1` = Roubo).



# Estrutura do Repositório
* `GS_Redes_Neurais.ipynb` — Notebook desenvolvido no Google Colab com a análise exploratória, pré-processamento, treinamento e comparação das 3 arquiteturas MLP.
* `app.py` — Código-fonte da interface web desenvolvida em Streamlit.
* `best_mlp_model.keras` — Modelo MLP selecionado (arquitetura intermediária com Dropout) já treinado.
* `preprocessor.pkl` — Pipeline de codificação (*One-Hot Encoder*) ajustado aos dados de treino.
* `requirements.txt` — Lista de dependências do projeto.



# Como Executar a Aplicação Localmente

1. **Clone este repositório** ou baixe os arquivos para uma pasta local:
   ```bash
   git clone https://github.com/dante05contato-eng/GS-Redes-Neurais.git
   cd GS-Redes-Neurais
   ```

2. **Instale as dependências necessárias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute a aplicação web:**
   ```bash
   streamlit run app.py
   ```