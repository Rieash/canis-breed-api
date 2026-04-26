"""
Simple Dog Breed Classifier Backend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("=" * 50)
print("Flask app starting...")
print("=" * 50)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/test', methods=['GET'])
def test():
    print("TEST endpoint called")
    return jsonify({'message': 'API is working'})

@app.route('/debug', methods=['GET'])
def debug():
    import sys
    return jsonify({
        'python_version': sys.version,
        'flask_version': '3.0.0',
        'working': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    return jsonify({
        'success': True,
        'top_prediction': {
            'breed': 'Shih Tzu',
            'confidence': 0.75,
            'confidence_percentage': '75.0%',
            'breed_info': {
                'name': 'Shih Tzu',
                'origin': 'China',
                'temperament': 'Affectionate, playful, outgoing',
                'life_span': '10-18 years',
                'weight': '4-7 kg',
                'height': '20-28 cm',
                'description': 'The Shih Tzu is a toy dog breed known for its long, flowing coat and friendly disposition.',
                'image_url': '',
            }
        },
        'all_probabilities': {'Shih Tzu': 0.75, 'Golden Retriever': 0.15, 'Pomeranian': 0.10}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
