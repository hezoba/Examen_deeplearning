
# Installer les bibliothèques nécessaires

!pip install pydub SpeechRecognition pandas transformers openpyxl
!pip install transformers torchaudio

#Imortation des bibliothèques necessaires

import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
from pydub import AudioSegment


# Fonction de lecture fichier audio

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

# Dossier contenant les fichiers audio

audio_folder = 'dataset_audio'
results = []

# Lire les fichiers audio dans le dossier

for index, filename in enumerate(os.listdir(audio_folder)):
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        file_path = os.path.join(audio_folder, filename)
        try:
            text = audio_to_text(file_path)

            results.append({
                'Numero': index + 1,
                'Texte': text,
            })
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {filename}: {e}")

# Créer un DataFrame et l'exporter en Excel

df = pd.DataFrame(results)
df.to_excel('Transforamtion_audio_texte.xlsx', index=False)

# Charger le modèle entraîné

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load('classification.pth'))
model.eval()

# Initialiser le tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Charger le fichier Excel

input_file_path = 'Transforamtion_audio_texte.xlsx'  # Remplacez par le chemin de votre fichier Excel
output_file_path = 'Analyse_sentiments.xlsx' # Chemin pour le fichier de sortie

df = pd.read_excel(input_file_path)
texts = df['Texte']
order_numbers = df['Numero']

# Préparer les prédictions

predictions = []

with torch.no_grad():
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()  # Obtenir la prédiction
        predictions.append("Positif" if prediction == 1 else "Négatif")  # 1 = Positif et 0 = Négatif

# Créer un DataFrame avec les résultats

df_results = pd.DataFrame({
    'Numero': order_numbers,
    'Texte': texts,
    'Sentiment prédit': predictions
})

# Sauvegarder les résultats dans un nouveau fichier Excel

df_results.to_excel(output_file_path, index=False)
print("Les résultats ont été sauvegardés avec succès dans le fichier Analyse_sentiments.xlsx")

print(df_results)