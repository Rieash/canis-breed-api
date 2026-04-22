"""
Improved Dog Breed Classifier Backend
Uses better TensorFlow Hub model for dog breed classification
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
CORS(app)

# TheDogAPI Key
THEDOGAPI_KEY = "live_FUHeepNv7WI2yi2qeVFG8GhgCjOQpQhz3O2DcLErAYAxptsCqbyCMQ5XQKsqLTlB"
THEDOGAPI_URL = "https://api.thedogapi.com/v1"

# Cache breed info
breed_cache = {}

def get_breed_info(breed_name):
    """Get breed info from TheDogAPI"""
    try:
        response = requests.get(
            f"{THEDOGAPI_URL}/breeds/search",
            headers={"x-api-key": THEDOGAPI_KEY},
            params={"q": breed_name},
            timeout=10
        )
        
        if response.status_code == 200:
            breeds = response.json()
            if breeds:
                return breeds[0]
        return None
    except Exception as e:
        print(f"TheDogAPI error: {e}")
        return None

def classify_with_ml(image_bytes):
    """Use better TensorFlow Hub model for dog breed classification"""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        
        # Use a better model - ResNet V2 trained on ImageNet (more accurate)
        model = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5")
        
        # Load ImageNet labels
        labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
        )
        with open(labels_path, 'r') as f:
            imagenet_labels = f.read().splitlines()
        
        # Preprocess
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model(img_array)
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            label = imagenet_labels[idx + 1]  # +1 because first label is background
            breed = map_label_to_breed(label)
            if breed:
                results.append({
                    'breed': breed,
                    'confidence': confidence,
                    'confidence_percentage': f"{confidence:.1%}"
                })
        
        return results[0] if results else None
        
    except Exception as e:
        print(f"ML classification error: {e}")
        return None

def map_label_to_breed(label):
    """Map ImageNet label to common dog breed name"""
    # Common dog breed mappings
    breed_mapping = {
        'Japanese spaniel': 'Japanese Chin',
        'Maltese dog': 'Maltese',
        'Pekinese': 'Pekingese',
        'Shih-Tzu': 'Shih Tzu',
        'Blenheim spaniel': 'Cavalier King Charles Spaniel',
        'papillon': 'Papillon',
        'toy terrier': 'Toy Terrier',
        'Rhodesian ridgeback': 'Rhodesian Ridgeback',
        'Afghan hound': 'Afghan Hound',
        'basset': 'Basset Hound',
        'beagle': 'Beagle',
        'bloodhound': 'Bloodhound',
        'bluetick': 'Bluetick Coonhound',
        'black-and-tan coonhound': 'Black and Tan Coonhound',
        'Walker hound': 'Treeing Walker Coonhound',
        'English foxhound': 'English Foxhound',
        'redbone': 'Redbone Coonhound',
        'borzoi': 'Borzoi',
        'Irish wolfhound': 'Irish Wolfhound',
        'Italian greyhound': 'Italian Greyhound',
        'whippet': 'Whippet',
        'Ibizan hound': 'Ibizan Hound',
        'Norwegian elkhound': 'Norwegian Elkhound',
        'otterhound': 'Otterhound',
        'Saluki': 'Saluki',
        'Scottish deerhound': 'Scottish Deerhound',
        'Weimaraner': 'Weimaraner',
        'Staffordshire bullterrier': 'Staffordshire Bull Terrier',
        'American Staffordshire terrier': 'American Staffordshire Terrier',
        'Bedlington terrier': 'Bedlington Terrier',
        'border terrier': 'Border Terrier',
        'Kerry blue terrier': 'Kerry Blue Terrier',
        'Irish terrier': 'Irish Terrier',
        'Norfolk terrier': 'Norfolk Terrier',
        'Norwich terrier': 'Norwich Terrier',
        'Yorkshire terrier': 'Yorkshire Terrier',
        'wire-haired fox terrier': 'Wire Fox Terrier',
        'Lakeland terrier': 'Lakeland Terrier',
        'Sealyham terrier': 'Sealyham Terrier',
        'Airedale': 'Airedale Terrier',
        'cairn': 'Cairn Terrier',
        'Australian terrier': 'Australian Terrier',
        'Dandie Dinmont': 'Dandie Dinmont Terrier',
        'Boston bull': 'Boston Terrier',
        'miniature schnauzer': 'Miniature Schnauzer',
        'giant schnauzer': 'Giant Schnauzer',
        'standard schnauzer': 'Standard Schnauzer',
        'Scotch terrier': 'Scottish Terrier',
        'Tibetan terrier': 'Tibetan Terrier',
        'silky terrier': 'Silky Terrier',
        'soft-coated wheaten terrier': 'Soft Coated Wheaten Terrier',
        'West Highland white terrier': 'West Highland White Terrier',
        'Lhasa': 'Lhasa Apso',
        'flat-coated retriever': 'Flat-Coated Retriever',
        'curly-coated retriever': 'Curly-Coated Retriever',
        'golden retriever': 'Golden Retriever',
        'Labrador retriever': 'Labrador Retriever',
        'Chesapeake Bay retriever': 'Chesapeake Bay Retriever',
        'German short-haired pointer': 'German Shorthaired Pointer',
        'vizsla': 'Vizsla',
        'English setter': 'English Setter',
        'Irish setter': 'Irish Setter',
        'Gordon setter': 'Gordon Setter',
        'Brittany spaniel': 'Brittany',
        'clumber': 'Clumber Spaniel',
        'English springer': 'English Springer Spaniel',
        'Welsh springer spaniel': 'Welsh Springer Spaniel',
        'cocker spaniel': 'Cocker Spaniel',
        'Sussex spaniel': 'Sussex Spaniel',
        'Irish water spaniel': 'Irish Water Spaniel',
        'kuvasz': 'Kuvasz',
        'schipperke': 'Schipperke',
        'groenendael': 'Belgian Sheepdog',
        'malinois': 'Belgian Malinois',
        'briard': 'Briard',
        'kelpie': 'Australian Kelpie',
        'komondor': 'Komondor',
        'Old English sheepdog': 'Old English Sheepdog',
        'Shetland sheepdog': 'Shetland Sheepdog',
        'collie': 'Rough Collie',
        'Border collie': 'Border Collie',
        'Bouvier des Flandres': 'Bouvier des Flandres',
        'Rottweiler': 'Rottweiler',
        'German shepherd': 'German Shepherd',
        'Doberman': 'Doberman Pinscher',
        'miniature pinscher': 'Miniature Pinscher',
        'Greater Swiss Mountain dog': 'Greater Swiss Mountain Dog',
        'Bernese mountain dog': 'Bernese Mountain Dog',
        'Appenzeller': 'Appenzeller Sennenhund',
        'EntleBucher': 'Entlebucher Mountain Dog',
        'boxer': 'Boxer',
        'bull mastiff': 'Bullmastiff',
        'Tibetan mastiff': 'Tibetan Mastiff',
        'French bulldog': 'French Bulldog',
        'Great Dane': 'Great Dane',
        'Saint Bernard': 'Saint Bernard',
        'Eskimo dog': 'American Eskimo Dog',
        'malamute': 'Alaskan Malamute',
        'Siberian husky': 'Siberian Husky',
        'Dalmatian': 'Dalmatian',
        'affenpinscher': 'Affenpinscher',
        'basenji': 'Basenji',
        'pug': 'Pug',
        'Leonberg': 'Leonberger',
        'Newfoundland': 'Newfoundland',
        'Great Pyrenees': 'Great Pyrenees',
        'Samoyed': 'Samoyed',
        'Pomeranian': 'Pomeranian',
        'chow': 'Chow Chow',
        'keeshond': 'Keeshond',
        'Brabancon griffon': 'Brussels Griffon',
        'Pembroke': 'Pembroke Welsh Corgi',
        'Cardigan': 'Cardigan Welsh Corgi',
        'toy poodle': 'Toy Poodle',
        'miniature poodle': 'Miniature Poodle',
        'standard poodle': 'Standard Poodle',
        'Mexican hairless': 'Xoloitzcuintli',
        'timber wolf': 'Gray Wolf',
        'white wolf': 'Arctic Wolf',
        'red wolf': 'Red Wolf',
        'coyote': 'Coyote',
        'dingo': 'Dingo',
        'dhole': 'Dhole',
        'African hunting dog': 'African Wild Dog',
        'hyena': 'Hyena',
        'red fox': 'Red Fox',
        'kit fox': 'Kit Fox',
        'Arctic fox': 'Arctic Fox',
        'grey fox': 'Gray Fox',
        'tabby': 'Tabby Cat',
        'Persian cat': 'Persian Cat',
        'Siamese cat': 'Siamese Cat',
        'Egyptian cat': 'Egyptian Mau',
        'cougar': 'Cougar',
        'lynx': 'Lynx',
        'leopard': 'Leopard',
        'snow leopard': 'Snow Leopard',
        'tiger': 'Tiger',
        'lion': 'Lion',
        'tiger cat': 'Tiger Cat',
        'sphinx': 'Sphynx Cat',
        'Egyptian mau': 'Egyptian Mau',
        'boxer': 'Boxer',
    }
    
    return breed_mapping.get(label.lower(), label.title() if 'dog' in label.lower() or 'hound' in label.lower() else None)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Canis Breed Classifier v2',
        'features': ['Improved ML classification', 'TheDogAPI breed info']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict dog breed from image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        print(f"Received image: {image_file.filename}, size: {len(image_bytes)} bytes")
        
        # Classify with ML
        ml_result = classify_with_ml(image_bytes)
        
        if ml_result:
            breed_name = ml_result['breed']
            confidence = ml_result['confidence']
            confidence_pct = ml_result['confidence_percentage']
            
            print(f"ML classification: {breed_name} ({confidence_pct})")
            
            # Get breed info from TheDogAPI
            breed_info = get_breed_info(breed_name)
            
            if breed_info:
                response = {
                    'success': True,
                    'top_prediction': {
                        'breed': breed_name,
                        'confidence': confidence,
                        'confidence_percentage': confidence_pct,
                        'breed_info': {
                            'name': breed_info.get('name', breed_name),
                            'origin': breed_info.get('origin', 'Unknown'),
                            'temperament': breed_info.get('temperament', 'Unknown'),
                            'life_span': breed_info.get('life_span', 'Unknown'),
                            'weight': breed_info.get('weight', {}).get('metric', 'Unknown'),
                            'height': breed_info.get('height', {}).get('metric', 'Unknown'),
                            'description': breed_info.get('description', f'{breed_name} is a wonderful dog breed.'),
                            'image_url': breed_info.get('image', {}).get('url', ''),
                        }
                    },
                    'alternative_predictions': []
                }
            else:
                # Fallback if TheDogAPI fails
                response = {
                    'success': True,
                    'top_prediction': {
                        'breed': breed_name,
                        'confidence': confidence,
                        'confidence_percentage': confidence_pct,
                        'breed_info': {
                            'name': breed_name,
                            'origin': 'Unknown',
                            'temperament': 'Friendly, loyal',
                            'life_span': '10-15 years',
                            'weight': 'Medium',
                            'height': 'Medium',
                            'description': f'{breed_name} is a wonderful dog breed.',
                            'image_url': '',
                        }
                    },
                    'alternative_predictions': []
                }
            
            return jsonify(response)
        else:
            return jsonify({'error': 'Classification failed - could not identify dog breed'}), 500
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Canis Breed Classifier API - V2 (Improved)")
    print("=" * 60)
    print("TheDogAPI: ✓ Configured")
    print("Features:")
    print("  - ResNet V2 classification (more accurate)")
    print("  - TheDogAPI breed database")
    print("  - 120+ dog breeds supported")
    print("=" * 60)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
