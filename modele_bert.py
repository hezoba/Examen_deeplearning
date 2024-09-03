
# Etape 1 Importation des bibliothèque necessaires

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from sklearn.metrics import classification_report, accuracy_score
from google.colab import drive

# Monter Google Drive
drive.mount('/content/drive')

# Étape 2 : Charger les données à partir des fichiers CSV téléchargé (voir questionnaire de l'examen)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_val = pd.read_csv('valid.csv')

# Étape 3 : Créer un Dataset personnalisé

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialiser le tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Créer les datasets

MAX_LENGTH = 128
BATCH_SIZE = 16

train_dataset = SentimentDataset(df_train['review'], df_train['polarity'], tokenizer, MAX_LENGTH)
test_dataset = SentimentDataset(df_test['review'], df_test['polarity'], tokenizer, MAX_LENGTH)
val_dataset = SentimentDataset(df_val['review'], df_val['polarity'], tokenizer, MAX_LENGTH)

# Créer les DataLoaders

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Étape 4 : Créer le modèle BERT

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Configurer le modèle pour l'entraînement

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Étape 5 : Entraîner le modèle

EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Afficher l'itération et la perte
        if (i + 1) % 1000 == 0:  # Afficher toutes les 1000 itérations
            print(f'  Iteration {i + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}')

# Étape 6 :Sauvegarde du modèle entraîné

torch.save(model.state_dict(), 'classification.pth')
print("Modèle sauvegardé avec succès.")

# Étape 7 : Évaluer le modèle sur l'ensemble de test

model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

# Calculer l'accuracy pour l'ensemble de test

test_accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy sur l'ensemble de test : {test_accuracy:.4f}")

# Afficher le rapport de classification pour l'ensemble de test

print("Rapport de classification sur l'ensemble de test :")
print(classification_report(true_labels, predictions, target_names=["Négatif", "Positif"]))

# Étape 8 : Évaluer le modèle sur l'ensemble de validation

val_predictions, val_true_labels = [], []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        val_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_true_labels.extend(batch['labels'].cpu().numpy())

# Calculer l'accuracy pour l'ensemble de validation

val_accuracy = accuracy_score(val_true_labels, val_predictions)
print(f"Accuracy sur l'ensemble de validation : {val_accuracy:.4f}")

# Afficher le rapport de classification pour l'ensemble de validation

print("Rapport de classification sur l'ensemble de validation :")
print(classification_report(val_true_labels, val_predictions, target_names=["Négatif", "Positif"]))