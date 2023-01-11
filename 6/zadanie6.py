#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: jan 2023
# version ='0.1'
# Algorytm przetwarza strumień z kamery i
# wykrywa twarz, jeżeli twarz się rusza rysuje calownik na środku
# Required: face_recognition, cv2
# ---------------------------------------------------------------------------
import face_recognition
import cv2
import math


def face_detect():
    """Przetwarzanie strumienia z kamery i wykrywanie
    położenia ruszajacej się twarzy.
    Po wykryciu, rysujemy celownik na środku twarzy.
    Wciśnięcie 'q' powoduje zamknięcie okna.
    """
    # Uruchamiamy kamerę
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    wait = 0
    # W pętli while sprawdzamy zmiany położenia twarzy
    while True:
        ret, frame = video_capture.read()
        if wait > 0:
            wait -= 1
        else:
            changed = False

        # Przeskalowujemy obraz
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        face_locations_old = face_locations
        face_locations = face_recognition.face_locations(
            small_frame, model="hog")

        # Sprawdzamy, czy położenie twarzy uległo zmianie
        if face_locations_old != face_locations:
            changed = True
            wait = 5

        for top, right, bottom, left in face_locations:
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            center_x = left + math.floor((right - left) / 2)
            center_y = top + math.floor((bottom - top) / 2)

            if changed:
                cv2.circle(frame, (center_x, center_y), 27, (0, 0, 255), 2)
                cv2.line(frame, (center_x, center_y),
                         (center_x, center_y - 50), (0, 0, 255), 2)
                cv2.line(frame, (center_x, center_y),
                         (center_x - 50, center_y), (0, 0, 255), 2)
                cv2.line(frame, (center_x, center_y),
                         (center_x + 50, center_y), (0, 0, 255), 2)
                cv2.line(frame, (center_x, center_y),
                         (center_x, center_y + 50), (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detect()
