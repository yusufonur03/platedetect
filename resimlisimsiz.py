from flask import Flask, request, send_file
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
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = model(image)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            plate_img = image[y1:y2, x1:x2]

            # OCR işlemi yapılacaksa buraya eklenebilir ancak sadece kare içine alacağız
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    output_path = "detected_plate.jpg"
    cv2.imwrite(output_path, image)

    return send_file(output_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
