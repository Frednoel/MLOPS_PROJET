# Utilisation d'une image officielle Python
FROM python:3.9.6

# Définition du répertoire de travail
WORKDIR /app

RUN pip install --upgrade pip

# Copie du fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste de l'application
COPY . .

# Définition du port exposé (ajuste selon ton besoin)
EXPOSE 8000

# Commande de démarrage (à adapter si nécessaire)
CMD ["streamlit", "run", "appfred.py"]

