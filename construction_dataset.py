import cv2
import mediapipe as mp
import os
from pathlib import Path

# Initialisation de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_hand_landmarks(image_path):
    """
    Extrait les coordonnées des points de la main depuis une image.
    Retourne une liste de coordonnées [x1, y1, z1, x2, y2, z2, ...] ou None si pas de main détectée.
    """
    # Lecture de l'image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Erreur: Impossible de lire {image_path}")
        return None
    
    # Conversion BGR vers RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Détection de la main
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Extraction des coordonnées du premier ensemble de points détecté
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Création d'une liste plate de coordonnées [x, y, z] pour chaque point
        coords = []
        for landmark in hand_landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        
        return coords
    
    return None

def process_gesture_folder(gesture_folder, output_folder):
    """
    Traite tous les images d'un dossier de geste et sauvegarde les coordonnées.
    """
    gesture_folder = Path(gesture_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    gesture_name = gesture_folder.name
    output_file = output_folder / f"{gesture_name}.txt"
    
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Liste toutes les images dans le dossier
    image_files = [f for f in gesture_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"Aucune image trouvée dans {gesture_folder}")
        return
    
    print(f"\nTraitement du geste: {gesture_name}")
    print(f"Nombre d'images: {len(image_files)}")
    
    successful_extractions = 0
    
    # Ouvre le fichier de sortie
    with open(output_file, 'w') as f:
        for image_file in image_files:
            coords = extract_hand_landmarks(image_file)
            
            if coords is not None:
                # Écrit les coordonnées séparées par des espaces
                coords_str = ' '.join(map(str, coords))
                f.write(coords_str + '\n')
                successful_extractions += 1
            else:
                print(f"  Aucune main détectée dans: {image_file.name}")
    
    print(f"Extraction terminée: {successful_extractions}/{len(image_files)} images traitées")
    print(f"Fichier sauvegardé: {output_file}")

def process_all_gestures(dataset_folder, output_folder):
    """
    Traite tous les dossiers de gestes dans le dataset.
    """
    dataset_folder = Path(dataset_folder)
    
    if not dataset_folder.exists():
        print(f"Erreur: Le dossier {dataset_folder} n'existe pas")
        return
    
    # Liste tous les sous-dossiers (chaque dossier = un geste)
    gesture_folders = [f for f in dataset_folder.iterdir() if f.is_dir()]
    
    if not gesture_folders:
        print(f"Aucun dossier de geste trouvé dans {dataset_folder}")
        return
    
    print(f"Nombre de gestes à traiter: {len(gesture_folders)}")
    
    for gesture_folder in gesture_folders:
        process_gesture_folder(gesture_folder, output_folder)
    
    print("\n=== Extraction terminée pour tous les gestes ===")

# Utilisation du script
if __name__ == "__main__":
    # Spécifiez vos chemins ici
    DATASET_FOLDER = "/Users/valentindaveau/Documents/UE_Vision/DATASET"  # Dossier contenant les sous-dossiers de gestes
    OUTPUT_FOLDER = "/Users/valentindaveau/Documents/UE_Vision/Coordonnees"  # Dossier où sauvegarder les fichiers .txt
    
    process_all_gestures(DATASET_FOLDER, OUTPUT_FOLDER)
    
    # Fermeture de MediaPipe
    hands.close()