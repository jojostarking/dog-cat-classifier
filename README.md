---
title: Dog vs Cat Classifier
emoji: 🐶🐱
colorFrom: green
colorTo: indigo
sdk: docker
app_file: app.py 
python_version: 3.9 
# model: your-username/your-model-on-hub 
---

# 🐱🐶 Dog vs Cat Classifier App

這是一個簡單的貓狗圖像分類器，使用 **TensorFlow/Keras** 訓練，並透過 **Flask** 部署在 Hugging Face Spaces 上。

## 如何使用

上傳一張貓或狗的圖片，模型將會預測圖片中是貓還是狗。

## 專案結構
├── app.py             # Flask 應用主檔案
├── model.h5           # 預訓練的貓狗分類模型 (由Git LFS管理)
├── requirements.txt   # Python 依賴列表
├── Procfile           # Hugging Face Spaces 啟動命令
├── README.md          # 專案說明與 Space 配置
├── static/            # 靜態檔案 (CSS, JS, 圖片等)
└── templates/         # HTML 模板

## 本地運行

1.  **克隆儲存庫：**
    ```bash
    git clone [https://github.com/jojostarking/dog-cat-classifier.git](https://github.com/jojostarking/dog-cat-classifier.git)
    cd dog-cat-classifier
    ```
2.  **建立並啟動 Anaconda 環境：**
    ```bash
    conda create -n dogcat_env python=3.9
    conda activate dogcat_env
    ```
3.  **安裝依賴：**
    ```bash
    pip install -r requirements.txt
    ```
4.  **運行應用程式：**
    ```bash
    python app.py
    ```
    然後在瀏覽器中訪問 `http://127.0.0.1:5000` (或控制台顯示的地址)。

## 部署到 Hugging Face Spaces

。