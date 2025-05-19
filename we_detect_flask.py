from flask import Flask, request, jsonify
from google.cloud import vision
import os

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r''

app = Flask(__name__)

def annotate(path: str) -> vision.WebDetection:
    """
    Returns web annotations given the path to an image.

    Args:
        path: path to the input image.

    Returns:
        An WebDetection object with relevant information of the
        image from the internet (i.e., the annotations).
    """
    client = vision.ImageAnnotatorClient()

    if path.startswith("http") or path.startswith("gs:"):
        image = vision.Image()
        image.source.image_uri = path
    else:
        with open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

    web_detection = client.web_detection(image=image).web_detection
    return web_detection

def parse_annotations(annotations: vision.WebDetection) -> dict:
    """
    Parses detected features in the provided web annotations.

    Args:
        annotations: The web annotations (WebDetection object) to parse.

    Returns:
        A dictionary containing parsed web detection information.
    """
    result = {
        "pages_with_matching_images": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "web_entities": [],
    }

    if annotations.pages_with_matching_images:
        result["pages_with_matching_images"] = [
            {"url": page.url} for page in annotations.pages_with_matching_images
        ]

    if annotations.full_matching_images:
        result["full_matching_images"] = [
            {"url": image.url} for image in annotations.full_matching_images
        ]

    if annotations.partial_matching_images:
        result["partial_matching_images"] = [
            {"url": image.url} for image in annotations.partial_matching_images
        ]

    if annotations.web_entities:
        result["web_entities"] = [
            {"score": entity.score, "description": entity.description}
            for entity in annotations.web_entities
        ]

    return result

@app.route('/annotate', methods=['POST'])
def annotate_image():
    """
    Flask route to handle image annotation requests.

    Expects a JSON payload with an 'image_path' key containing either a URL,
    a Google Cloud Storage path, or a local file path.
    """
    try:
        data = request.json
        image_path = data.get('image_path')

        if not image_path:
            return jsonify({"error": "Image path is required."}), 400

        annotations = annotate(image_path)
        response = parse_annotations(annotations)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
