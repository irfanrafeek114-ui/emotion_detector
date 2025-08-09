import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def detect_expression(landmarks):
    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]
    mouth_left = landmarks[78]
    mouth_right = landmarks[308]
    eye_left_top = landmarks[159]
    eye_left_bottom = landmarks[145]

    mouth_height = np.linalg.norm([mouth_top.x - mouth_bottom.x, mouth_top.y - mouth_bottom.y])
    eye_openness = np.linalg.norm([eye_left_top.x - eye_left_bottom.x, eye_left_top.y - eye_left_bottom.y])

    if mouth_height > 0.05 and eye_openness > 0.03:
        return "Surprised"
    elif mouth_height > 0.03:
        return "Happy"
    elif eye_openness < 0.015:
        return "Sleepy"
    elif mouth_height < 0.02:
        return "Sad"
    else:
        return "Neutral"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    expression = "No face detected"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        expression = detect_expression(landmarks)

        h, w, _ = frame.shape
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.putText(frame, f"Expression: {expression}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Facial Expression Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
