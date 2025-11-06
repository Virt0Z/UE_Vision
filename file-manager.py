import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

tab = []

with open("/Users/valentindaveau/Documents/UE_Vision/training_set/sequences/2.txt", "r", encoding="utf-8") as f:
    lignes = f.readlines()

lignes = [ligne.strip() for ligne in lignes]




class Coords():
    def __init__(self):
        self.terms = [('palmpos',3),('thumbApos',3),('thumBpos',3),('thumbEndpos',3),('indexApos',3),('indexBpos',3),
                      ('indexCpos',3),('indexEndpos',3),('middleApos',3),('middleBpos',3),('middleCpos',3),
                      ('middleEndpos',3),('ringApos',3),('ringBpos',3),('ringCpos',3),('ringEndpos',3),
                      ('pinkyApos',3),('pinkyBpos',3),('pinkyCpos',3),('pinkyEndpos',3)]
        for term, count in self.terms:
            setattr(self, term, [])
        


position = Coords()


for ligne in lignes:
    ligne = ligne.strip()
    if not ligne:
        continue  
    values = [float(v) for v in ligne.split(";") if v.strip() != ""]
    i = 0
    for name, count in position.terms:
        subset = tuple(values[i:i+count])  
        getattr(position, name).append(subset)
        i += count + 4 #On augmente de 4 car on ne prend pas en compte les quads pour le moment


with open("/Users/valentindaveau/Documents/UE_Vision/training_set/annotations_revised_training.txt", "r", encoding="utf-8") as f:
    lignes2 = f.readlines()

# Parser les annotations - chaque ligne correspond à un fichier
# Format: sequence_id;label1;start1;end1;label2;start2;end2;...
# Parser les annotations - nouvelle logique
annotations_by_sequence = {}


for ligne in lignes2:
    ligne = ligne.strip()
    if not ligne:
        continue
    
    parts = [v.strip() for v in ligne.split(";") if v.strip() != ""]
    
    if len(parts) < 1:
        continue
    
    sequence_id = int(parts[0])
    annotations_by_sequence[sequence_id] = []
    
    # Parser les quadruplets (label, end1, start2, ...)
    # Où end1 = fin du label, start2 = début du prochain label
    i = 1
    current_start = 0
    
    while i + 1 < len(parts):
        label = parts[i]
        try:
            end_frame = int(parts[i + 1])
            annotations_by_sequence[sequence_id].append({
                'label': label,
                'start': current_start,
                'end': end_frame
            })
            
            # Le prochain label commence à la frame suivante après le gap
            if i + 2 < len(parts):
                # Il y a un nombre suivant qui indique où commence le prochain label
                if i + 2 < len(parts) and not parts[i + 2].isalpha():
                    current_start = int(parts[i + 2])
                    i += 3  # On saute label, end, next_start
                else:
                    current_start = end_frame + 1
                    i += 2
            else:
                i += 2
        except (ValueError, IndexError):
            i += 1

sequence_number = 2  # Correspond à 2.txt
frame_labels = {}

if sequence_number in annotations_by_sequence:
    annotations = annotations_by_sequence[sequence_number]
    
    # Construire les labels frame par frame
    for ann in annotations:
        label = ann['label']
        start_frame = ann['start']
        end_frame = ann['end']
        
        # Assigner le label de start_frame à end_frame
        for frame in range(start_frame, end_frame + 1):
            frame_labels[frame] = label
    
    print(f"Séquence {sequence_number}:")
    print(f"  - Total frames: {len(getattr(position, position.terms[0][0]))}")
    print(f"  - Frames avec labels: {len(frame_labels)}")
    print(f"  - Mouvements annotés: {len(annotations)}")
    
    # Afficher les intervalles
    for ann in annotations:
        print(f"    * {ann['label']}: frames {ann['start']}-{ann['end']}")
else:
    print(f"Aucune annotation trouvée pour la séquence {sequence_number}")


# ============ VISUALISATION INTERACTIVE ============

class HandVisualizer:
    def __init__(self, position_data, frame_labels_dict, sequence_num):
        self.position = position_data
        self.frame_labels = frame_labels_dict
        self.sequence_num = sequence_num
        self.current_frame = 0
        self.total_frames = len(getattr(position_data, position_data.terms[0][0]))
        
        # Stocker les angles de vue
        self.view1_elev = 20
        self.view1_azim = 45
        self.view2_elev = 20
        self.view2_azim = 135
        
        # Définir les connexions entre les articulations pour dessiner les doigts
        self.finger_chains = {
            'Pouce': ['palmpos', 'thumbApos', 'thumBpos', 'thumbEndpos'],
            'Index': ['palmpos', 'indexApos', 'indexBpos', 'indexCpos', 'indexEndpos'],
            'Majeur': ['palmpos', 'middleApos', 'middleBpos', 'middleCpos', 'middleEndpos'],
            'Annulaire': ['palmpos', 'ringApos', 'ringBpos', 'ringCpos', 'ringEndpos'],
            'Auriculaire': ['palmpos', 'pinkyApos', 'pinkyBpos', 'pinkyCpos', 'pinkyEndpos']
        }
        
        self.colors = {
            'Pouce': 'red',
            'Index': 'blue',
            'Majeur': 'green',
            'Annulaire': 'orange',
            'Auriculaire': 'purple'
        }
        
        # Créer la figure
        self.fig = plt.figure(figsize=(16, 8))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
        # Ajuster l'espace pour les boutons
        self.fig.subplots_adjust(bottom=0.2, top=0.85)
        
        # Créer les boutons
        ax_prev = plt.axes([0.15, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.3, 0.05, 0.1, 0.075])
        ax_jump_back = plt.axes([0.45, 0.05, 0.12, 0.075])
        ax_jump = plt.axes([0.6, 0.05, 0.12, 0.075])
        ax_next_label = plt.axes([0.75, 0.05, 0.12, 0.075])
        
        self.btn_prev = Button(ax_prev, '← Frame')
        self.btn_next = Button(ax_next, 'Frame →')
        self.btn_jump_back = Button(ax_jump_back, '← 10 frames')
        self.btn_jump = Button(ax_jump, '10 frames →')
        self.btn_next_label = Button(ax_next_label, 'Prochain label')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_jump_back.on_clicked(self.jump_back)
        self.btn_jump.on_clicked(self.jump_frames)
        self.btn_next_label.on_clicked(self.next_labeled_frame)
        
        # Connecter les touches du clavier
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.update_plot()
        plt.show()
    
    def save_view_angles(self):
        """Sauvegarde les angles de vue actuels avant de redessiner"""
        self.view1_elev = self.ax1.elev
        self.view1_azim = self.ax1.azim
        self.view2_elev = self.ax2.elev
        self.view2_azim = self.ax2.azim
    
    def restore_view_angles(self):
        """Restaure les angles de vue sauvegardés"""
        self.ax1.view_init(elev=self.view1_elev, azim=self.view1_azim)
        self.ax2.view_init(elev=self.view2_elev, azim=self.view2_azim)
    
    def get_current_label(self):
        """Retourne le label du mouvement pour la frame actuelle"""
        return self.frame_labels.get(self.current_frame, "Pas de label")
    
    def next_labeled_frame(self, event):
        """Saute à la prochaine frame qui a un label"""
        start_frame = self.current_frame + 1
        for i in range(start_frame, self.total_frames):
            if i in self.frame_labels:
                self.current_frame = i
                self.update_plot()
                return
        # Si on n'a rien trouvé, chercher depuis le début
        for i in range(0, start_frame):
            if i in self.frame_labels:
                self.current_frame = i
                self.update_plot()
                return
    
    def update_plot(self):
        # Sauvegarder les angles de vue actuels
        self.save_view_angles()
        
        # Effacer les graphiques
        self.ax1.cla()
        self.ax2.cla()
        
        # Pour chaque doigt, récupérer les points et tracer
        for finger_name, joints in self.finger_chains.items():
            points = []
            for joint in joints:
                joint_data = getattr(self.position, joint)
                if self.current_frame < len(joint_data):
                    points.append(joint_data[self.current_frame])
            
            if len(points) > 0:
                points = np.array(points)
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
                
                # Graphique 1 : Vue 3D avec lignes connectées
                self.ax1.plot(x, y, z, 'o-', color=self.colors[finger_name], 
                             label=finger_name, linewidth=2, markersize=8)
                
                # Graphique 2 : Même chose avec une vue différente
                self.ax2.plot(x, y, z, 'o-', color=self.colors[finger_name], 
                             label=finger_name, linewidth=2, markersize=8)
        
        # Configuration des axes
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            # Définir les mêmes limites pour tous les axes pour garder les proportions
            all_points = []
            for joint_name, _ in self.position.terms:
                joint_data = getattr(self.position, joint_name)
                if self.current_frame < len(joint_data):
                    all_points.append(joint_data[self.current_frame])
            
            if len(all_points) > 0:
                all_points = np.array(all_points)
                margin = 0.1
                ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
                ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
                ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
        
        # Restaurer les angles de vue sauvegardés
        self.restore_view_angles()
        
        self.ax1.set_title('Vue 1')
        self.ax2.set_title('Vue 2')
        
        # Titre principal avec label du mouvement
        current_label = self.get_current_label()
        label_color = 'green' if current_label != "Pas de label" else 'gray'
        
        self.fig.suptitle(f'Séquence {self.sequence_num} - Frame {self.current_frame}/{self.total_frames - 1}\n'
                         f'Mouvement: {current_label}', 
                         fontsize=16, fontweight='bold', color=label_color)
        
        plt.draw()
    
    def next_frame(self, event):
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.update_plot()
    
    def prev_frame(self, event):
        self.current_frame = (self.current_frame - 1) % self.total_frames
        self.update_plot()
    
    def jump_frames(self, event):
        self.current_frame = (self.current_frame + 10) % self.total_frames
        self.update_plot()
    
    def jump_back(self, event):
        self.current_frame = (self.current_frame - 10) % self.total_frames
        self.update_plot()
    
    def on_key(self, event):
        if event.key == 'right':
            self.next_frame(None)
        elif event.key == 'left':
            self.prev_frame(None)
        elif event.key == 'up':
            self.current_frame = (self.current_frame + 10) % self.total_frames
            self.update_plot()
        elif event.key == 'down':
            self.current_frame = (self.current_frame - 10) % self.total_frames
            self.update_plot()
        elif event.key == 'n':
            self.next_labeled_frame(None)
        elif event.key == 'q' or event.key == 'escape':
            plt.close()


# Lancer la visualisation
print("\n=== VISUALISATION DE LA MAIN AVEC LABELS ===")
print("Utilisez les boutons ou les flèches du clavier pour naviguer")
print("Flèche droite : frame suivante")
print("Flèche gauche : frame précédente")
print("Flèche haut : +10 frames")
print("Flèche bas : -10 frames")
print("N : aller au prochain label")
print("Q ou Echap : quitter")
print("\nAstuce : Utilisez la souris pour faire pivoter les vues 3D")
print("         L'angle de vue sera conservé quand vous changez de frame !\n")

visualizer = HandVisualizer(position, frame_labels, sequence_number)