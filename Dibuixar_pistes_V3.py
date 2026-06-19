import cv2      # OpenCV: per capturar la càmera i dibuixar a la imatge
import numpy as np  # Càlcul matemàtic amb llistes de punts (matrius)
import os         # Per comprovar fitxers/carpetes del sistema
import json       # Per desar la configuració en un fitxer de text estructurat

# ==========================================
# QUÈ FA AQUEST PROGRAMA?
# ==========================================
# És l'eina de CALIBRATGE. Es fa servir UNA vegada, abans d'engegar la visió.
# Fa una foto del circuit amb la càmera i deixa que l'usuari dibuixi a mà,
# amb el ratolí:
#   - Les PISTES (per on passen els cotxes).
#   - Les ZONES DE PERILL (corbes/cruïlles on poden xocar).
# Tot això es guarda al fitxer 'config_pistes.json', que després llegeixen
# els programes de visió per saber la geometria del circuit.

# ==========================================
# 1. CAPTURAR LA FOTO (CONFIGURACIÓ HARDWARE IDÈNTICA A YOLO)
# ==========================================
print("Iniciant la càmera...")
cap = cv2.VideoCapture(0)  # El 0 vol dir "la primera càmera connectada a l'ordinador"

# FORCEM LA MATEIXA RESOLUCIÓ QUE EL PROGRAMA DE VISIÓ (640x480).
# És fonamental: si calibrem a una mida i analitzem a una altra, les
# coordenades dibuixades no quadrarien amb la realitat.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Error: No s'ha pogut obrir la càmera.")
    exit()

# Comprovem a quina resolució REAL s'ha obert la càmera (algunes no admeten 640x480)
w_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Càmera configurada a: {w_real}x{h_real} (Idèntic a YOLO)")

print("\n--- PREPARACIÓ DEL CALIBRATGE ---")
print("1. Assegura't que el circuit es veu sencer.")
print("2. Prem la tecla 'ESPAI' per fer la foto i començar a dibuixar.")
print("3. Prem 'q' per cancel·lar i sortir.")

imatge_original = None  # Aquí guardarem la foto congelada del circuit

# Bucle de previsualització: mostrem el vídeo en directe fins que l'usuari faci la foto
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error llegint el vídeo de la càmera.")
        break

    cv2.imshow("Enquadra el circuit i prem ESPAI per fer la foto", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord(' '):              # Barra espaiadora -> congelem la imatge
        imatge_original = frame.copy()
        break
    elif tecla == ord('q'):            # 'q' -> cancel·lem
        break

# Ja tenim la foto: alliberem la càmera i tanquem la finestra de previsualització
cap.release()
cv2.destroyWindow("Enquadra el circuit i prem ESPAI per fer la foto")

if imatge_original is None:
    print("❌ Calibratge cancel·lat per l'usuari.")
    exit()

nom_fitxer_config = 'config_pistes.json'
# Estructura buida on anirem acumulant tot el que dibuixi l'usuari
config_final = {"pistes": {}, "zones_perill": []}

# ==========================================
# FASE 1: DIBUIXAR LES PISTES (Ruta del cotxe)
# ==========================================
try:
    num_pistes = int(input("\nQuantes pistes diferents vols calibrar al teu circuit? (ex: 2): "))
except ValueError:
    print("❌ Has d'introduir un número vàlid.")
    exit()

# Funció lligada a la barra lliscant: quan l'usuari mou el control de gruix,
# actualitzem la variable global i tornem a redibuixar la pista.
def actualitzar_gruix(val):
    global gruix_pista
    gruix_pista = val
    dibuixar_pista()

# Repetim el procés de dibuix una vegada per cada pista que vulgui l'usuari
for num_pista in range(1, num_pistes + 1):
    print(f"\n--- CALIBRANT LA PISTA {num_pista} ---")
    print("🖱️  Botó ESQUERRE: Afegir punt a la línia actual.")
    print("✂️  Botó DRET: Tallar la línia per saltar-te un tros (ponts/túnels).")
    print("🎚️  Barra superior: Ajusta el gruix a temps real.")
    print("✅  Prem 'q' o 'Enter' quan acabis.")

    # Una pista és una llista de "segments"; cada segment és una llista de punts.
    # Tallar amb el botó dret crea un nou segment buit (per fer salts/ponts).
    segments_pista = [[]]
    gruix_pista = 40  # Gruix inicial raonable per a una imatge de 640px d'ample

    # Què passa quan l'usuari clica amb el ratolí sobre la imatge de la pista
    def fer_clic_pista(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Botó esquerre: afegim el punt clicat al segment actual
            segments_pista[-1].append((int(x), int(y)))
            dibuixar_pista()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Botó dret: tanquem el segment actual i n'obrim un de nou
            if len(segments_pista[-1]) > 0:
                segments_pista.append([])
                print("   -> Tram tallat! Pots seguir dibuixant.")

    # Funció que redibuixa la pista sencera sobre una còpia neta de la foto
    def dibuixar_pista():
        img_temp = imatge_original.copy()
        capa_gruix = np.zeros_like(img_temp)  # Capa negra a part per pintar-hi el gruix

        # 1r) Pintem la "carretera" grossa (verda) a la capa transparent
        for segment in segments_pista:
            if len(segment) > 1:
                cv2.polylines(capa_gruix, [np.array(segment)], False, (0, 255, 0), gruix_pista)

        # Barregem la capa verda amb la foto al 50% per veure-hi a través (transparència)
        cv2.addWeighted(capa_gruix, 0.5, img_temp, 1.0, 0, img_temp)

        # 2n) Pintem a sobre la línia fina vermella i els punts clicats (per precisió)
        for segment in segments_pista:
            if len(segment) > 1:
                cv2.polylines(img_temp, [np.array(segment)], False, (0, 0, 255), 2)
            for p in segment:
                cv2.circle(img_temp, p, 4, (0, 0, 255), -1)

        cv2.imshow(f"Calibrador - Pista {num_pista}", img_temp)

    # Creem la finestra, la barra de gruix i activem l'escolta del ratolí
    cv2.namedWindow(f"Calibrador - Pista {num_pista}")
    cv2.createTrackbar("Gruix", f"Calibrador - Pista {num_pista}", gruix_pista, 150, actualitzar_gruix)
    cv2.setMouseCallback(f"Calibrador - Pista {num_pista}", fer_clic_pista)

    dibuixar_pista()
    # Esperem fins que l'usuari premi 'q' o 'Enter' (codi 13) per acabar aquesta pista
    while True:
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 13:
            break
    cv2.destroyWindow(f"Calibrador - Pista {num_pista}")

    # Descartem segments buits i guardem la pista (els seus punts i el gruix) al diccionari
    segments_valids = [seg for seg in segments_pista if len(seg) > 0]
    if len(segments_valids) > 0:
        config_final["pistes"][f"pista_{num_pista}"] = {"segments": segments_valids, "gruix": gruix_pista}

# ==========================================
# FASE 2: DIBUIXAR ZONES DE PERILL (Corbes)
# ==========================================
# Una zona de perill és un polígon tancat (no una línia) que envolta un punt
# delicat del circuit, com una corba o una cruïlla on els cotxes poden xocar.
print("\n--- CALIBRANT ZONES DE PERILL (CORBES) ---")
print("🖱️  Botó ESQUERRE: Afegir vèrtex del polígon (envolta la corba).")
print("✂️  Botó DRET: Tancar el polígon actual i començar una nova zona de perill.")
print("✅  Prem 'q' o 'Enter' quan tinguis totes les corbes marcades.")

poligons_perill = [[]]  # Llista de polígons; cada polígon és una llista de vèrtexs

# Què passa quan l'usuari clica sobre la imatge de zones de perill
def fer_clic_perill(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Botó esquerre: afegim un vèrtex al polígon actual
        poligons_perill[-1].append((int(x), int(y)))
        dibuixar_perill()
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Botó dret: tanquem aquesta zona i comencem la següent
        if len(poligons_perill[-1]) > 2:  # Un polígon necessita almenys 3 punts
            poligons_perill.append([])
            print("   -> Zona de perill tancada! Comença'n una altra.")

# Funció que redibuixa totes les zones de perill sobre la foto
def dibuixar_perill():
    img_temp = imatge_original.copy()
    capa_perill = np.zeros_like(img_temp)  # Capa a part per a l'ombrejat vermell

    for poligon in poligons_perill:
        if len(poligon) > 0:
            pts = np.array(poligon, np.int32)
            cv2.fillPoly(capa_perill, [pts], (0, 0, 255))      # Omplim la zona de vermell
            cv2.polylines(img_temp, [pts], True, (0, 0, 255), 2)  # Vora del polígon
            for p in poligon:
                cv2.circle(img_temp, p, 4, (0, 255, 255), -1)  # Vèrtexs en groc

    # Barregem l'ombrejat vermell amb la foto (transparència al 40%)
    cv2.addWeighted(capa_perill, 0.4, img_temp, 1.0, 0, img_temp)
    cv2.imshow("Zones de Perill", img_temp)

cv2.namedWindow("Zones de Perill")
cv2.setMouseCallback("Zones de Perill", fer_clic_perill)

dibuixar_perill()
# Esperem que l'usuari acabi de marcar totes les corbes
while True:
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q') or tecla == 13:
        break
cv2.destroyWindow("Zones de Perill")

# Només guardem els polígons que tinguin almenys 3 punts (els vàlids)
poligons_valids = [pol for pol in poligons_perill if len(pol) > 2]
if len(poligons_valids) > 0:
    config_final["zones_perill"] = poligons_valids
    print(f"✅ Guardades {len(poligons_valids)} zones de perill.")

# --- 3. GUARDAR EL RESULTAT FINAL ---
# Afegim la resolució de la càmera al JSON per si la necessitem en el futur
config_final["resolucio_original"] = [w_real, h_real]

# Escrivim tota la configuració al fitxer JSON (indent=4 el fa llegible per humans)
with open(nom_fitxer_config, 'w') as f:
    json.dump(config_final, f, indent=4)
print(f"\n🎉 Tota la configuració s'ha guardat al fitxer '{nom_fitxer_config}'.")
