from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

app = Flask(__name__)


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


model_path = r"C:\\Users\\V1CTUS\\Desktop\\dataset\\training_results\\weights\\best.pt"
model = YOLO(model_path)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    results = model(image)

    plates = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_img = gray[y1:y2, x1:x2]  

            
            text = pytesseract.image_to_string(plate_img, config='--psm 8')
            plates.append(text.strip())

    return jsonify({"plates": plates})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

