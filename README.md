#1- abra o terminal no vscode
verifique em qual branch está, com o comando: git branch
munda de branch com o comando: git checkout Denisebranch

confirme se esta na nova branch: git branch

# 2. Crie o ambiente virtual
python -m venv .venv

# 3. Ative o ambiente virtual
- No Windows:
.venv\Scripts\activate


# 4. Instale as dependências
pip install -r requirements.txt

# 5. Rode a aplicação
python app.py
