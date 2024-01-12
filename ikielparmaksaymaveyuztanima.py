import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHand = mp.solutions.hands
eller = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

mpYuz = mp.solutions.face_detection
yuz_tespit = mpYuz.FaceDetection()

parmakUcuIdListesi = [4, 8, 12, 16, 20]
oncekiPozisyonlar = None
esik_deger = 30

while True:
    basarili, img = cap.read()

    if not basarili:
        print("Kamera açma hatası!")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eller_sonuclari = eller.process(imgRGB)
    yuz_sonuclari = yuz_tespit.process(img)

    sagElListesi = []  # Sağ elin parmak pozisyonları
    solElListesi = []  # Sol elin parmak pozisyonları

    if eller_sonuclari.multi_hand_landmarks:
        for elPozisyonlari in eller_sonuclari.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, elPozisyonlari, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(elPozisyonlari.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                elListesi = sagElListesi if elPozisyonlari.landmark[mpHand.HandLandmark.WRIST].x > 0.5 else solElListesi
                elListesi.append([id, cx, cy])

                renk = (0, 255, 0)
                if id in parmakUcuIdListesi:
                    renk = (255, 0, 0)
                cv2.circle(img, (cx, cy), 9, renk, cv2.FILLED)

        sagParmaklar = []  # Sağ el parmakları
        solParmaklar = []  # Sol el parmakları

        for id in range(5):
            # Sağ el parmakları
            if len(sagElListesi) > parmakUcuIdListesi[id] and len(sagElListesi) > parmakUcuIdListesi[id] - 2 and sagElListesi[parmakUcuIdListesi[id]][2] < sagElListesi[parmakUcuIdListesi[id] - 2][2]:
                sagParmaklar.append(1)
            else:
                sagParmaklar.append(0)

            # Sol el parmakları
            if len(solElListesi) > parmakUcuIdListesi[id] and len(solElListesi) > parmakUcuIdListesi[id] - 2 and solElListesi[parmakUcuIdListesi[id]][2] < solElListesi[parmakUcuIdListesi[id] - 2][2]:
                solParmaklar.append(1)
            else:
                solParmaklar.append(0)

        sagToplamParmakSayisi = sagParmaklar.count(1)  # Sağ el parmak sayısı
        solToplamParmakSayisi = solParmaklar.count(1)  # Sol el parmak sayısı

        # Toplam parmak sayılarını ekrana yazdırma
        cv2.putText(img, f"Sag El Parmak Sayisi: {sagToplamParmakSayisi}", (30, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f"Sol El Parmak Sayisi: {solToplamParmakSayisi}", (30, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Yüz tanıma
    if yuz_sonuclari.detections:
        for tespit in yuz_sonuclari.detections:
            bboxC = tespit.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, "Yuz Tespit Edildi!", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == 27:  # 27, Esc tuşunun ASCII kodudur
        break

cap.release()
cv2.destroyAllWindows()
