from flask import Flask, jsonify, request
import base64
from processing import ImageProcessingManager

app = Flask(__name__)

ipm = ImageProcessingManager()

@app.route('/process', methods=['POST'])
def basic_image_processing():
    base64_image = request.form.get("image")
    return jsonify(
        color_locations=ipm.processForColor(base64_image)
    )

if __name__ == "__main__":
    app.run()
