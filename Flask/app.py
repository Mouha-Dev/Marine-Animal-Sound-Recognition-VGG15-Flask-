import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import numpy as np
from skimage.transform import resize

# Configuration de l'application Flask
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

logging.basicConfig(level=logging.INFO)

# Paramètres modèle
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
IMG_SIZE = (224, 224)

# Labels 
labels = ['Atlantic_Spotted_Dolphin', 'Bearded_Seal', 'Beluga_White_Whale', 'Bottlenose_Dolphin', 'Bowhead_Whale', 'Clymene_Dolphin', 'Common_Dolphin', 'False_Killer_Whale', 'Finback_Whale', 'Fraser_s_Dolphin', 'Grampus_Risso_s_Dolphin', 'Harp_Seal', 'Humpback_Whale', 'Killer_Whale', 'Leopard_Seal', 'Long_Finned_Pilot_Whale', 'Melon_Headed_Whale', 'Minke_Whale', 'Narwhal', 'Northern_Right_Whale', 'Pantropical_Spotted_Dolphin', 'Ross_Seal', 'Rough_Toothed_Dolphin', 'Short_Finned_Pacific_Pilot_Whale', 'Southern_Right_Whale', 'Sperm_Whale', 'Spinner_Dolphin_Stenella_longirostris', 'Striped_Dolphin', 'Walrus', 'Weddell_Seal', 'White_beaked_Dolphin', 'White_sided_Dolphin'] 

french_translations = {
    'Atlantic_Spotted_Dolphin': 'Dauphin tacheté de l\'Atlantique',
    'Bearded_Seal': 'Phoque barbu',
    'Beluga_White_Whale': 'Béluga baleine blanche',
    'Bottlenose_Dolphin': 'Grand dauphin',
    'Bowhead_Whale': 'Baleine boréale',
    'Clymene_Dolphin': 'Dauphin de Clymène',
    'Common_Dolphin': 'Dauphin commun',
    'False_Killer_Whale': 'Fausse orque',
    'Finback_Whale': 'Baleine à dos commun',
    'Fraser_s_Dolphin': 'Dauphin de Fraser',
    'Grampus_Risso_s_Dolphin': 'Dauphin de Risso',
    'Harp_Seal': 'Phoque du Groenland',
    'Humpback_Whale': 'Baleine à bosse',
    'Killer_Whale': 'Orque',
    'Leopard_Seal': 'Léopard de mer',
    'Long_Finned_Pilot_Whale': 'Globicéphale commun',
    'Melon_Headed_Whale': 'Baleine à tête de melon',
    'Minke_Whale': 'Petit rorqual',
    'Narwhal': 'Narval',
    'Northern_Right_Whale': 'Baleine franche',
    'Pantropical_Spotted_Dolphin': 'Dauphin tacheté pantropical',
    'Ross_Seal': 'Phoque de Ross',
    'Rough_Toothed_Dolphin': 'Dauphin à dents rugueuses',
    'Short_Finned_Pacific_Pilot_Whale': 'Globicéphale tropical',
    'Southern_Right_Whale': 'Baleine franche australe',
    'Sperm_Whale': 'Cachalot',
    'Spinner_Dolphin_Stenella_longirostris': 'Dauphin longirostre ou Dauphin à long bec',
    'Striped_Dolphin': 'Dauphin bleu et blanc',
    'Walrus': 'Morse',
    'Weddell_Seal': 'Phoque de Weddell',
    'White_beaked_Dolphin': 'Dauphin à bec blanc',
    'White_sided_Dolphin': 'Dauphin à flancs blancs'
}    

# Chargement du modele
def load_trained_model():
    try:
        model_path = "modele/marine_sounds15_vgg19.h5"
        model = load_model(model_path)
        app.logger.info(f"Modèle chargé avec succès depuis {model_path}")
        app.logger.info(f"Nombre de classes: {len(labels)}")
        
        return model
        
    except Exception as e:
        app.logger.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Chargement du modèle au démarrage
model = load_trained_model()

# Fonctions utilitaires
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonction pour convertir WAV en Mel spectrogramme
def wav_to_melspectrogram(path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS):
    try:
       
        y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        log_mel_norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())

        img = tf.image.resize(log_mel_norm[..., np.newaxis], IMG_SIZE)
        img = np.repeat(img.numpy(), 3, axis=-1)  # spectrogramme → RGB
        
        return img
        
    except Exception as e:
        app.logger.error(f"Erreur lors de la conversion audio: {e}")
        return None

# Fonction pour nettoyer le nom de l'animal
def clean_animal_name(name):
    return name.strip()


# Logique de prédiction du modèle 
def predict_marine_animal(audio_path):
    try:
        if model is None:
            return {"error": "Erreur: Modèle non chargé"}
        
        # Convertir le fichier en spectrogramme
        img = wav_to_melspectrogram(audio_path)
        
        if img is None:
            return {"error": "Erreur lors du traitement audio"}
        
        # Préparer les données pour la prédiction 
        img = np.expand_dims(img, axis=0)  
        
        # Prédiction
        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        animal_name = clean_animal_name(labels[pred_class])
        animal_name_fr = french_translations.get(animal_name, animal_name)
        
        
        result_string = f"{animal_name_fr} \n (Confiance: {confidence:.2%})"
        
        app.logger.info(f"Prédiction: {result_string}")
        
        return result_string

    except Exception as e:
        app.logger.error(f"Erreur lors de la prédiction : {e}")
        import traceback
        return {'error': f"Erreur lors de l'analyse : {str(e)}"}

# Route pour tester avec un fichier spécifique 
@app.route('/test')
def test_model():
    try:
        
        file_path = "marine_sounds_test/Finback_Whale/6107900F.wav"
        
        if not os.path.exists(file_path):
            return "Fichier de test non trouvé"
        
        result = predict_marine_animal(file_path)
        
        if 'error' in result:
            return f"Erreur: {result['error']}"
        
        return render_template('index.html', 
                             prediction=result, 
                             filename=os.path.basename(file_path))
        
    except Exception as e:
        return f"Erreur lors du test: {str(e)}"

# Route principale 
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            prediction_result = predict_marine_animal(filepath)
            
            # Nettoyer le fichier temporaire
            try:
                os.remove(filepath)
            except OSError as e:
                app.logger.error(f"Erreur lors de la suppression du fichier {filepath}: {e}")

            return render_template('index.html', 
                                 prediction=prediction_result, 
                                 filename=filename)
            
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    if model is not None:
        app.logger.info(" Application Flask démarrée avec le modèle chargé")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        app.logger.error("Impossible de démarrer: modèle non chargé")