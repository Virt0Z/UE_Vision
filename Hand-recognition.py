import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- IMPORTANT: backend AVFoundation (macOS) + tester index 0 puis 1 ---
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Caméra non ouverte (backend AVFoundation). Vérifie permissions et index.")

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ cap.read() a échoué — autre app utilisant la caméra ? mauvais index ?")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    out = frame.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(out, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Detection', out)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
