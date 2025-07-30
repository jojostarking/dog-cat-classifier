from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# 載入訓練好的模型
model = tf.keras.models.load_model("model.h5")

# 設定圖片上傳資料夾
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 預測圖片是否為貓或狗
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        return "狗 🐶"
    else:
        return "貓 🐱"

# 主頁面
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="請選擇圖片！")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="沒有選擇圖片！")
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        return render_template("index.html", result=result, img_path=filepath)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)