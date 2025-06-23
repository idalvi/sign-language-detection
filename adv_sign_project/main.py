import numpy as np
import cv2 as cv
import mediapipe as mp
import pyttsx3
from function import *

# Initialize TTS engine
engine = pyttsx3.init()

# Mediapipe hands
holy_hands = mp.solutions.hands
cap = cv.VideoCapture(0)

# Buffers & settings
recent_predictions = []
sentence_buffer = []
cooldown_frames = 0
COOLDOWN_LIMIT = 10
no_hand_frames = 0
NO_HAND_RESET = 5

# Two-hand letter detection
def detect_two_hand_letter(left_hand, right_hand):
    def is_index_up(hand): return hand[8][2] < hand[6][2]
    def is_thumb_extended(hand): return hand[4][1] > hand[3][1]

    if left_hand is not None and right_hand is not None:
        if is_index_up(left_hand) and is_thumb_extended(right_hand):
            return "L"  # Example two-hand letter
    return ""

with holy_hands.Hands(max_num_hands=2) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        imgH, imgW = image.shape[:2]

        left_hand, right_hand = None, None

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                coords = np.array(
                    [[i, int(lm.x * imgW), int(lm.y * imgH)] for i, lm in enumerate(hand_landmarks.landmark)]
                )
                handedness = results.multi_handedness[idx].classification[0].label
                if handedness == "Left":
                    left_hand = coords
                elif handedness == "Right":
                    right_hand = coords
                image = get_fram(image, coords, persons_input(coords))

        # Detect from left or right hand
        one_hand_letter = ""
        if left_hand is not None:
            one_hand_letter = persons_input(left_hand)
        elif right_hand is not None:
            one_hand_letter = persons_input(right_hand)

        # Prefer two-hand letter
        detected_letter = detect_two_hand_letter(left_hand, right_hand) or one_hand_letter

        # Letter detection
        if detected_letter:
            no_hand_frames = 0
            recent_predictions.append(detected_letter)
        else:
            no_hand_frames += 1
            if no_hand_frames >= NO_HAND_RESET:
                cooldown_frames = 0

        if len(recent_predictions) > 10:
            recent_predictions.pop(0)

        most_common_letter = (
            max(set(recent_predictions), key=recent_predictions.count)
            if recent_predictions else ""
        )

        if cooldown_frames > 0:
            cooldown_frames -= 1

        if most_common_letter and cooldown_frames == 0:
            sentence_buffer.append(most_common_letter)
            cooldown_frames = COOLDOWN_LIMIT
            recent_predictions.clear()

        # Display
        sentence_text = "".join(sentence_buffer)
        cv.putText(
            image, f"Sentence: {sentence_text}",
            (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            engine.say(sentence_text)
            engine.runAndWait()
        elif key == ord('c'):
            sentence_buffer.clear()
        elif key == ord('x'):
            break

        cv.imshow('Sign Language detection', cv.flip(image, 1))

cap.release()
cv.destroyAllWindows()
