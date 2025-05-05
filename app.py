from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle
import torch
from transformers import MarianMTModel, MarianTokenizer
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ==== Load mô hình image caption ====
model = load_model("model1.keras")
with open("image_features1.pkl", "rb") as f:
    tokenizer_caption = pickle.load(f)

vocab_size = len(tokenizer_caption.word_index) + 1
max_length = 74
img_size = 224

# Trích đặc trưng ảnh
base_model = DenseNet201(include_top=True, weights='imagenet')
fe_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# ==== Load mô hình dịch ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translate_model_name = "Helsinki-NLP/opus-mt-en-vi"
translate_tokenizer = MarianTokenizer.from_pretrained(translate_model_name)
translate_model = MarianMTModel.from_pretrained(translate_model_name)
translate_model.load_state_dict(torch.load("translate.pt", map_location=device))
translate_model.to(device)
translate_model.eval()

# === Dịch tiếng Anh -> Việt ===
def translate_en_to_vi(text):
    inputs = translate_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    translated = translate_model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    return translate_tokenizer.decode(translated[0], skip_special_tokens=True)

# === Tạo caption ===
def generate_caption(photo):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer_caption.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer_caption.index_word.get(yhat)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq ", "").replace(" endseq", "")

# ==== ROUTES ====
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route('/generate', methods=["POST"])
def generate():
    caption = ""
    image_url = ""
    bleu_scores = {}

    file = request.files['image']
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_url = f"/{file_path}"

        # Tiền xử lý ảnh
        img = load_img(file_path, target_size=(img_size, img_size))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        photo = fe_model.predict(img, verbose=0)

        # Caption tiếng Anh
        caption = generate_caption(photo)

        # Tính BLEU nếu có tham chiếu
        ref_caption = request.form.get("reference", "").strip()
        if ref_caption:
            reference = [[w.lower() for w in ref_caption.split()]]
            candidate = caption.split()
            smoothie = SmoothingFunction().method1

            # Tính BLEU cho các n-gram (1-gram đến 4-gram)
            for n in range(1, 5):
                weight = tuple((1. / n for _ in range(n))) + tuple(0. for _ in range(4 - n))
                score = sentence_bleu(reference, candidate, weights=weight, smoothing_function=smoothie)
                bleu_scores[f"BLEU-{n}"] = round(score, 4)
        else:
            bleu_scores = None

    return jsonify({"caption": caption, "image_url": image_url, "bleu": bleu_scores})

@app.route('/translate', methods=["POST"])
def translate():
    data = request.get_json()
    caption = data.get("caption")
    translated_caption = translate_en_to_vi(caption)
    return jsonify({"translated": translated_caption})

if __name__ == '__main__':
    app.run(debug=True)
