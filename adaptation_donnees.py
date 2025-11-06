import numpy as np
import matplotlib.pyplot as plt
# Indices à garder dans l’ordre de TON dataset (20 points):
# wrist (0), thumb A(1), thumb B(2), thumb End(4),
# index A(5), B(6), C(7), End(8),
# middle A(9), B(10), C(11), End(12),
# ring A(13), B(14), C(15), End(16),
# pinky A(17), B(18), C(19), End(20)
KEEP_IDX_20 = [0, 1, 2, 4,  5,6,7,8,  9,10,11,12,  13,14,15,16,  17,18,19,20]

def mp21_to_dataset20(mp21_xyz: np.ndarray) -> np.ndarray:
    """
    mp21_xyz: array (21, 3) en ordre MediaPipe (channels_last)
    retourne: array (20, 3) dans l’ordre de ton dataset
    """
    assert mp21_xyz.shape == (21, 3), f"attendu (21,3), reçu {mp21_xyz.shape}"
    return mp21_xyz[KEEP_IDX_20, :]


def normalize_landmarks(X20: np.ndarray) -> np.ndarray:
    """
    X20: (20,3) ordre dataset.
    - centre sur le poignet (point 0)
    - mise à l’échelle par une distance de référence stable (ex: WRIST→MIDDLE_MCP)
    """
    wrist = X20[0]
    Xc = X20 - wrist
    # distance de ref: WRIST (0) -> MIDDLE MCP (index MediaPipe 9, devenu X20[8] après mapping)
    scale = np.linalg.norm(Xc[8]) + 1e-8
    return Xc / scale

def load_sequence_txt(path):
    """
    Lit un .txt où chaque ligne est une frame : "v1;v2;...;vN;"
    -> retourne un numpy array (T, D) où T = nb de frames, D = nb de features par frame
    """
    feats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p for p in line.strip().split(";") if p.strip() != ""]
            if not parts:
                continue
            vec = list(map(float, parts))
            feats.append(vec)
    X = np.array(feats, dtype=np.float32)  # (T, D)
    return X

def parse_annotations(path):
    """
    Fichier d’annotations, format (extraits) :
      1;DENY;670;751;CIRCLE;1610;1655; ...
    => Pour chaque ID, on récupère une liste [(label, start, end), ...]
    """
    ann = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = [p for p in raw.split(";") if p != ""]
            # parts = [id, L1, s1, e1, L2, s2, e2, ...]
            seq_id = parts[0]
            triples = parts[1:]
            # lire par paquets de 3 (label, start, end)
            for i in range(0, len(triples), 3):
                label = triples[i]
                start = int(triples[i+1])
                end   = int(triples[i+2])
                ann[seq_id].append((label, start, end))
    return ann

# =========================
# 2) Prétraitements
# =========================

def normalize_framewise(X):
    """
    Normalisation simple frame-par-frame :
    - centrage par la moyenne
    - mise à l’échelle par l’écart-type
    Remarque : si tes features sont des coordonnées articulaires (x,y,z) concaténées,
    tu peux remplacer par un centrage/scale géométrique (soustraire le 'poignet', etc.).
    """
    Xn = X.copy()
    mu = Xn.mean(axis=1, keepdims=True)      # (T,1)
    sigma = Xn.std(axis=1, keepdims=True) + 1e-8
    Xn = (Xn - mu) / sigma
    return Xn

def slice_segment(X, start, end):
    """
    Découpe X (T,D) sur l’intervalle [start, end] inclusif (ou semi-ouvert)
    On borne pour éviter les dépassements.
    """
    T = len(X)
    s = max(0, start)
    e = min(T, end)
    if e <= s:
        return None
    return X[s:e]

# =========================
# 3) Construction dataset
# =========================

def build_dataset_from_folder(seq_folder, ann_path, max_len=200):
    """
    - lit toutes les séquences *.txt du dossier
    - récupère les segments annotés pour chaque ID
    - normalise, pad/tronque à max_len
    Retourne X (N, max_len, D), y (N,), class_names
    """
    annotations = parse_annotations(ann_path)
    sequences = sorted(glob.glob(os.path.join(seq_folder, "*.txt")))
    X_list, y_list = [], []
    label_to_id = {}
    next_id = 0

    for path in sequences:
        # ID = nom de fichier sans extension
        seq_id = os.path.splitext(os.path.basename(path))[0]
        if seq_id not in annotations:
            continue
        X = load_sequence_txt(path)          # (T, D)
        X = normalize_framewise(X)

        for (label, start, end) in annotations[seq_id]:
            seg = slice_segment(X, start, end)
            if seg is None: 
                continue
            # padding/tronquage à max_len
            if len(seg) >= max_len:
                seg = seg[:max_len]
            else:
                pad = np.zeros((max_len - len(seg), seg.shape[1]), dtype=seg.dtype)
                seg = np.vstack([seg, pad])
            # map label -> id
            if label not in label_to_id:
                label_to_id[label] = next_id
                next_id += 1
            y = label_to_id[label]
            X_list.append(seg)
            y_list.append(y)

    if not X_list:
        raise RuntimeError("Aucun segment construit (vérifie les chemins et les IDs).")

    X = np.stack(X_list, axis=0)          # (N, max_len, D)
    y = np.array(y_list, dtype=np.int64)  # (N,)
    # classes triées par id
    id_to_label = {v:k for k,v in label_to_id.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    return X, y, class_names