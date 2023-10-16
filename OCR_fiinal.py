from flask import Flask, request, jsonify
import cv2
import pytesseract
import json,os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    video_file = request.files['video']

    # Validate file
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Process video and extract frames
    cap = cv2.VideoCapture(video_file)
    ocr_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # OCR processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        options = "outputbase digits"
        text = pytesseract.image_to_string(rgb, config=options)
        
        ocr_results.append(text)

        json_file_path = os.path.join(os.getcwd(), 'ocr_results.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(ocr_results, json_file)
    
    return jsonify({'ocr_results': ocr_results})

@app.route('/')
def index():
    return "Hello, this is my Flask server!"


if __name__ == '__main__':
    app.run(debug=True)
