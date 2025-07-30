import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os # 導入os模組，以便處理端口

# 載入模型 (確保 model.h5 在應用程式的根目錄下)
try:
    model = load_model('model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # 如果模型載入失敗，可以在這裡退出或提供一個錯誤處理機制

# 定義用於 Gradio 的預測函數
# Gradio 會自動將上傳的圖片轉換為 PIL.Image.Image 物件
def predict_cat_dog(image_input):
    if image_input is None:
        return "請上傳一張圖片。" # 如果沒有圖片輸入，返回提示

    try:
        # 確保圖片被正確處理成模型期望的格式
        img = image_input.resize((128, 128)) # 調整圖片大小至模型輸入大小
        img_array = np.array(img) / 255.0 # 標準化像素值到 0-1 範圍
        img_array = np.expand_dims(img_array, axis=0) # 增加批次維度 (batch dimension)

        predictions = model.predict(img_array)

        # 假設您的模型輸出是二分類（貓/狗），並且 output[0][0] 是狗的機率
        # 您可能需要根據您模型的實際輸出調整這裡的邏輯和閾值
        if predictions[0][0] > 0.5:
            result = "這是一隻狗！🐶"
        else:
            result = "這是一隻貓！🐱"

        return result

    except Exception as e:
        return f"預測時發生錯誤：{e}" # 處理預測過程中的錯誤

# 創建 Gradio 界面
iface = gr.Interface(
    fn=predict_cat_dog, # 指定預測函數
    inputs=gr.Image(type="pil", label="上傳貓或狗的圖片"), # 輸入元件：圖片，接收為 PIL Image 物件
    outputs="text", # 輸出元件：文本
    title="🐈🐕‍🦺 貓狗分類器", # 應用程式標題
    description="這是一個簡單的應用程式，可以判斷您上傳的圖片是貓還是狗。", # 應用程式描述
    allow_flagging="auto", # 允許使用者對結果進行標記反饋
    # 如果您有範例圖片，可以在這裡添加，方便使用者測試
    # examples=[["path/to/cat_example.jpg"], ["path/to/dog_example.jpg"]]
)

# 啟動 Gradio 應用程式
# 在 Hugging Face Spaces 中，Gradio 會自動處理端口和主機
if __name__ == "__main__":
    # Gradio 在Hugging Face Spaces上會自動處理 PORT 和 HOST
    # 但本地運行時，通常使用 iface.launch() 即可
    iface.launch(
        server_name="0.0.0.0", # 讓它在所有網絡接口上可用
        server_port=int(os.environ.get("PORT", 7860)) # 使用環境變數或預設端口
    )