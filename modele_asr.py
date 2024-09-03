
# Installer les bibliothèques nécessaires
!pip install pydub SpeechRecognition pandas transformers openpyxl
!pip install transformers torchaudio

#Importation des bibliothèques nécessaires

import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
from pydub import AudioSegment


# Fonction pour les fichiers audios

def read_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    return audio

# Charger le modèle et le tokenizer de Hugging Face

model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Fonction pour convertir l'audio en texte

def audio_to_text(file_path):

    # Convertir le fichier audio en WAV si ce n'est pas déjà

    if not file_path.endswith('.wav'):
        audio_segment = AudioSegment.from_file(file_path)
        wav_path = file_path.rsplit('.', 1)[0] + '.wav'
        audio_segment.export(wav_path, format='wav')
        file_path = wav_path

    # Charger l'audio et effectuer la reconnaissance vocale

    waveform, sample_rate = torchaudio.load(file_path)
    input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt", padding="longest").input_values

    # Effectuer la prédiction

    with torch.no_grad():
        logits = model(input_values).logits

    # Obtenir les indices des tokens

    predicted_ids = logits.argmax(dim=-1)

    # Convertir les indices en texte

    text = tokenizer.batch_decode(predicted_ids)[0]
    return text

# Fonction pour analyser le sentiment du texte

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentiment = sentiment_pipeline(text)
    return sentiment[0]['label'], sentiment[0]['score']

# Dossier contenant les fichiers audio

audio_folder = 'dataset_audio'
results = []

# Lire les fichiers audio dans le dossier

for index, filename in enumerate(os.listdir(audio_folder)):
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        file_path = os.path.join(audio_folder, filename)
        try:
            text = audio_to_text(file_path)
            sentiment_label, sentiment_score = analyze_sentiment(text)

            results.append({
                'Numéro': index + 1,
                'Texte': text,
                'Sentiment': sentiment_label,
                'Score': sentiment_score
            })
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {filename}: {e}")

# Créer un DataFrame et l'exporter en Excel

df = pd.DataFrame(results)
df.to_excel('resultats_analyse_sentiment.xlsx', index=False)

# Affichier le resultat

print(df)