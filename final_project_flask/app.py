from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model and class names
model = load_model("models/model.h5")
loaded_data = np.load("models/model_and_classes_english.npy", allow_pickle=True).item()
class_names = loaded_data["class_names"]

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi untuk memprediksi gambar
def predict_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Sesuaikan ukuran dengan model
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    
    # Prediksi menggunakan model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Index kelas dengan probabilitas tertinggi
    confidence = predictions[0][predicted_class_index]  # Probabilitas kelas tertinggi
    return class_names[predicted_class_index], confidence

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Halaman analyze untuk mengunggah gambar
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Prediksi gambar
            predicted_label, confidence = predict_image(filepath)
            
            description = f"This is {predicted_label} with the confidence of {confidence:.2f}"
            
            # Menentukan link berdasarkan label
            if predicted_label.lower() == "cat":  # Sesuaikan dengan nama label dari model Anda
                redirect_url = url_for('articles_about_cat')
                
            elif predicted_label.lower() == "dog":  # Sesuaikan dengan nama label dari model Anda
                redirect_url = url_for('articles_about_dog')
                
            elif predicted_label.lower() == "butterfly":
                redirect_url = url_for('articles_about_butterfly')
                
            elif predicted_label.lower() == "chicken":
                redirect_url = url_for('articles_about_chicken')
                
            elif predicted_label.lower() == "cow":
                redirect_url = url_for('articles_about_cow')
                
            elif predicted_label.lower() == "elephant":
                redirect_url = url_for('articles_about_elephant')
                
            elif predicted_label.lower() == "horse":
                redirect_url = url_for('articles_about_horse')
                
            elif predicted_label.lower() == "sheep":
                redirect_url = url_for('articles_about_sheep')
                
            elif predicted_label.lower() == "spider":
                redirect_url = url_for('articles_about_spider')
                
            elif predicted_label.lower() == "squirrel":
                redirect_url = url_for('articles_about_squirrel')
                
            else:
                redirect_url = None  # Tidak ada tombol jika label tidak dikenali            

            # # Teks deskripsi
            # if predicted_label.lower() == 'cat':
            #     description = f"Ini adalah kucing dengan tingkat kepercayaan {confidence:.2f}."
            # else:
            #     description = f"Ini bukan kucing. Ini adalah {predicted_label} dengan tingkat kepercayaan {confidence:.2f}."

            return render_template('analyze.html', analyzed=True, image_url=url_for('static', filename=f'uploads/{filename}'), description=description, redirect_url=redirect_url)


    # GET request untuk menampilkan halaman analyze
    return render_template('analyze.html', analyzed=False)

@app.route('/articles_about_butterfly')
def articles_about_butterfly():
    return render_template('articles_about_butterfly.html')

@app.route('/articles_about_cat')
def articles_about_cat():
    return render_template('articles_about_cat.html')

@app.route('/articles_about_chicken')
def articles_about_chicken():
    return render_template('articles_about_chicken.html')

@app.route('/articles_about_cow')
def articles_about_cow():
    return render_template('articles_about_cow.html')

@app.route('/articles_about_dog')
def articles_about_dog():
    return render_template('articles_about_dog.html')

@app.route('/articles_about_elephant')
def articles_about_elephant():
    return render_template('articles_about_elephant.html')

@app.route('/articles_about_horse')
def articles_about_horse():
    return render_template('articles_about_horse.html')

@app.route('/articles_about_sheep')
def articles_about_sheep():
    return render_template('articles_about_sheep.html')

@app.route('/articles_about_spider')
def articles_about_spider():
    return render_template('articles_about_spider.html')

@app.route('/articles_about_squirrel')
def articles_about_squirrel():
    return render_template('articles_about_squirrel.html')

@app.route('/articles')
def articles_main():
    return render_template('articles_main.html')

@app.route('/articles/taking_your_dog_for_a_walk')
def articles_taking_your_dog():
    return render_template('articles_taking_your_dog.html')

@app.route('/articles/cat_101')
def articles_cat_101():
    return render_template('articles_cat_101.html')

@app.route('/articles/feeding_your_cat')
def articles_feeding_your_cat():
    return render_template('articles_feeding_your_cat.html')

@app.route('/articles/livestock_management')
def articles_livestock():
    return render_template('articles_livestock_management.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)