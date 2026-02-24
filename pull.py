import requests
import numpy as np
from PIL import Image

TF_URL = "http://localhost:8501/v1/models/mnist_model:predict"

img = Image.open("digit_3_3.png").convert("L").resize((28, 28))
img_arr = np.array(img).astype("float32") / 255.0
img_arr = img_arr.reshape(1, 28, 28, 1)

payload = {
    "instances": img_arr.tolist()
}

response = requests.post(TF_URL, json=payload)
predictions = response.json()["predictions"]
predicted_digit = np.argmax(predictions[0])

print("Predicted Digit:", predicted_digit)
