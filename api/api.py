from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# === Настройки ===
MODEL_PATH = "model"
LABEL_LIST = ["O", "B-PRF", "I-PRF"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# === Загрузка модели и токенизатора ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

# === FastAPI ===
app = FastAPI(title="Profanity Censorship API")

# === Входная модель ===
class TextInput(BaseModel):
    text: str

# === Функции цензуры ===
def predict_labels(text: str):
    tokens = text.strip().split()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    offset_mapping = encoding.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=2).squeeze().tolist()
    word_ids = encoding.word_ids()

    word_labels = []
    current_word = None

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != current_word:
            current_word = word_id
            word_labels.append(ID2LABEL[preds[i]])

    return tokens, word_labels

def censor_word(word: str, mask_char: str = "*") -> str:
    if len(word) <= 2:
        return mask_char * len(word)
    return word[0] + mask_char * (len(word) - 2) + word[-1]

def censor_text(text: str, censor_mode: str = "smart", mask_char: str = "*") -> str:
    tokens, labels = predict_labels(text)
    censored_tokens = []

    for token, label in zip(tokens, labels):
        if label in ["B-PRF", "I-PRF"]:
            if censor_mode == "smart":
                censored = censor_word(token, mask_char)
            elif censor_mode == "stars":
                censored = mask_char * len(token)
            else:
                censored = "[CENSORED]"
            censored_tokens.append(censored)
        else:
            censored_tokens.append(token)

    return " ".join(censored_tokens)

# === Эндпоинт ===
@app.post("/censor")
def censor_endpoint(data: TextInput):
    censored = censor_text(data.text)
    return {
        "original": data.text,
        "censored": censored
    }