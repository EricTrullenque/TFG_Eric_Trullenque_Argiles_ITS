import torch     # Motor de xarxes neuronals on s'executa el model YOLOv5
import os         # Per gestionar fitxers i carpetes del sistema
import sys        # Accés a parts internes de Python (per al pedaç de compatibilitat)
import pathlib    # Gestió de rutes de fitxers (necessari per al pedaç)
import cv2        # OpenCV: llegir/mostrar imatges i vídeo
import time       # Per mesurar la velocitat (FPS)

# ==========================================
# QUÈ FA AQUEST PROGRAMA?
# ==========================================
# És el DETECTOR. Passa el model d'intel·ligència artificial YOLOv5 per sobre
# d'imatges (de la càmera, d'una carpeta o d'un vídeo) i, per cada fotograma:
#   - Dibuixa caixes al voltant dels cotxes detectats i guarda la foto.
#   - Escriu un fitxer .txt amb les coordenades de cada objecte trobat.
# Aquests .txt els llegirà després el programa "Dibuixar_Vectors" per analitzar
# trajectòries i col·lisions.

# --- PARATXE DE COMPATIBILITAT ---
# Evita que un model entrenat/exportat a Windows falli quan s'executa a Linux
# (Raspberry Pi), per la manera diferent d'escriure les rutes de fitxers.
sys.modules['pathlib._local'] = pathlib
pathlib.WindowsPath = pathlib.PosixPath

# --- CONFIGURACIÓ GENERAL ---
ruta_pesos = 'best.pt'                       # Fitxer amb el "cervell" entrenat del model
carpeta_fotos_sortida = 'resultats_fotos'    # On es desaran les imatges amb les caixes
carpeta_txt_sortida = 'resultats_txt'        # On es desaran les coordenades en text

# Creem les carpetes de sortida si encara no existeixen (no falla si ja hi són)
os.makedirs(carpeta_fotos_sortida, exist_ok=True)
os.makedirs(carpeta_txt_sortida, exist_ok=True)

# --- CARREGAR EL MODEL ---
# Descarrega/carrega l'arquitectura YOLOv5 i hi posa els nostres pesos entrenats.
print("Carregant el model YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=ruta_pesos)

# ==========================================
# FUNCIÓ 1: ANALITZAR CÀMERA
# ==========================================
# Captura en directe de la càmera, analitza cada frame i va guardant els
# últims 30 segons (esborrant els més antics per no omplir el disc).
def processar_camera(compressio_guardat):
    font_video = 0  # 0 = primera càmera connectada
    cap = cv2.VideoCapture(font_video)

    # Demanem alta resolució a la càmera per a una anàlisi més precisa
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: No s'ha pogut obrir la càmera.")
        return

    # Si l'usuari ha triat comprimir, baixem la qualitat del JPEG (ocupa menys)
    qualitat_jpeg = 50 if compressio_guardat else 100
    fps_estimat = 10.0          # Suposició de velocitat per calcular quants frames són 30s
    segons_a_guardar = 30
    max_frames = int(fps_estimat * segons_a_guardar)  # Nombre màxim de frames a conservar

    print(f"\n--- INICIANT DETECCIÓ EN TEMPS REAL ---")
    print("⚠️ L'anàlisi s'està fent a MÀXIMA qualitat.")
    print(f"Es guardaran els últims {segons_a_guardar} segons.")
    print("Prem la lletra 'q' per tancar el programa.")

    contador_frames = 0

    while True:
        temps_inici = time.time()  # Marca de temps per calcular els FPS d'aquest frame

        ret, frame = cap.read()
        if not ret: break

        contador_frames += 1

        # --- MISSATGE CLAR PER PANTALLA ---
        print(f"-> Capturant i analitzant frame de càmera: {contador_frames}", flush=True)

        # YOLO treballa en format de color RGB, però OpenCV dóna BGR: cal convertir-ho
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # IA SEMPRE A MÀXIMA QUALITAT (640px): aquí és on detecta els objectes
        resultats = model(frame_rgb, size=640)
        dades = resultats.pandas().xywh[0]  # Taula amb les deteccions (centre, mida, confiança)

        # render() torna la imatge amb les caixes ja dibuixades; la tornem a BGR per OpenCV
        img_renderitzada_bgr = cv2.cvtColor(resultats.render()[0], cv2.COLOR_RGB2BGR)

        # Calculem i mostrem els FPS reals a la cantonada de la imatge
        fps_real = 1.0 / (time.time() - temps_inici)
        cv2.putText(img_renderitzada_bgr, f"FPS: {fps_real:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Deteccio YOLOv5 Camera (Prem "q" per sortir)', img_renderitzada_bgr)

        nom_base = f"frame_{contador_frames:05d}"  # Nom amb 5 xifres (frame_00001, ...)

        # GUARDAT DE LA IMATGE (amb o sense compressió segons l'opció triada)
        cv2.imwrite(os.path.join(carpeta_fotos_sortida, f"{nom_base}_detect.jpg"), img_renderitzada_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qualitat_jpeg])

        # GUARDAT DEL TXT: una fila per cada objecte detectat, amb les seves dades
        with open(os.path.join(carpeta_txt_sortida, f"{nom_base}.txt"), 'w') as fitxer:
            fitxer.write("Nom_Objecte, X_Centre(px), Y_Centre(px), Amplada(px), Alcada(px), Confianca\n")
            for _, fila in dades.iterrows():
                fitxer.write(f"{fila['name']}, {fila['xcenter']:.2f}, {fila['ycenter']:.2f}, {fila['width']:.2f}, {fila['height']:.2f}, {fila['confidence']:.2f}\n")

        # FINESTRA LLISCANT: esborrem el frame més antic per mantenir només els últims 30s
        if contador_frames > max_frames:
            antic = contador_frames - max_frames
            r_img = os.path.join(carpeta_fotos_sortida, f"frame_{antic:05d}_detect.jpg")
            r_txt = os.path.join(carpeta_txt_sortida, f"frame_{antic:05d}.txt")
            if os.path.exists(r_img): os.remove(r_img)
            if os.path.exists(r_txt): os.remove(r_txt)

        if cv2.waitKey(1) & 0xFF == ord('q'): break  # 'q' per sortir

    # Apagat net de la càmera i les finestres
    cap.release()
    cv2.destroyAllWindows()
    print("Procés finalitzat. Dades guardades.")

# ==========================================
# FUNCIÓ 2: ANALITZAR CARPETA D'IMATGES
# ==========================================
# Recorre totes les fotos d'una carpeta i les analitza una a una.
def processar_carpeta(compressio_guardat):
    carpeta_entrada = input("\nCom es diu la carpeta on tens les imatges? (Enter per defecte: 'imatges_a_analitzar'): ")
    if carpeta_entrada.strip() == "":
        carpeta_entrada = 'imatges_a_analitzar'

    os.makedirs(carpeta_entrada, exist_ok=True)
    arxius = sorted(os.listdir(carpeta_entrada))  # Ordenats per nom
    # Ens quedem només amb els fitxers d'imatge
    imatges = [f for f in arxius if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(imatges) == 0:
        print(f"\nATENCIÓ: La carpeta '{carpeta_entrada}' està buida.")
        return

    print(f"\nS'han trobat {len(imatges)} imatges per analitzar a màxima qualitat.")

    qualitat_jpeg = 50 if compressio_guardat else 100

    # Processem cada imatge igual que a la càmera: detectar -> dibuixar -> guardar foto + txt
    for i, nom_arxiu in enumerate(imatges, 1):
        ruta_imatge = os.path.join(carpeta_entrada, nom_arxiu)
        nom_base = f"frame_{i:05d}"

        # --- MISSATGE CLAR PER PANTALLA ---
        print(f"-> Processant imatge {i}/{len(imatges)}: {nom_arxiu}", flush=True)

        frame = cv2.imread(ruta_imatge)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # IA SEMPRE A MÀXIMA QUALITAT (640)
        resultats = model(frame_rgb, size=640)
        dades = resultats.pandas().xywh[0]
        img_renderitzada_bgr = cv2.cvtColor(resultats.render()[0], cv2.COLOR_RGB2BGR)

        # GUARDAT DE LA IMATGE (amb o sense compressió)
        cv2.imwrite(os.path.join(carpeta_fotos_sortida, f"{nom_base}_detect.jpg"), img_renderitzada_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qualitat_jpeg])

        # GUARDAT DEL TXT amb les coordenades de cada objecte
        with open(os.path.join(carpeta_txt_sortida, f"{nom_base}.txt"), 'w') as fitxer:
            fitxer.write("Nom_Objecte, X_Centre(px), Y_Centre(px), Amplada(px), Alcada(px), Confianca\n")
            for _, fila in dades.iterrows():
                fitxer.write(f"{fila['name']}, {fila['xcenter']:.2f}, {fila['ycenter']:.2f}, {fila['width']:.2f}, {fila['height']:.2f}, {fila['confidence']:.2f}\n")

    print(f"\nProcés completat! S'han analitzat {len(imatges)} imatges.")

# ==========================================
# FUNCIÓ 3: ANALITZAR VÍDEO NOU
# ==========================================
# Analitza un fitxer de vídeo (.mp4, .avi...). Permet saltar-se frames per
# anar més ràpid en vídeos llargs.
def processar_video(compressio_guardat):
    ruta_video = input("\nIntrodueix el nom del fitxer de vídeo (ex: 'video.mp4'): ")

    if not os.path.exists(ruta_video):
        print(f"Error: No s'ha trobat el fitxer '{ruta_video}'. Assegura't que estigui a la mateixa carpeta.")
        return

    # Opció per analitzar 1 de cada N frames i estalviar temps
    print("\nOPCIÓ D'ESTALVI DE TEMPS:")
    print("1 = Analitzar tots els frames (Lent)")
    print("2 = Analitzar 1 de cada 2 frames (Meitat de temps)")
    print("3 = Analitzar 1 de cada 3 frames (Ideal per vídeos llargs)")
    salt_input = input("Tria una opció (Prem Enter per deixar-ho en 1): ")

    try:
        salt_frames = int(salt_input)
        if salt_frames < 1: salt_frames = 1
    except ValueError:
        salt_frames = 1  # Si l'usuari no posa un número vàlid, analitzem-ho tot

    # Mostrar el vídeo mentre s'analitza alenteix molt; oferim apagar-ho
    resposta_pantalla = input("\nVols veure el vídeo mentre s'analitza? (S/N - Posa 'N' per anar molt més ràpid): ").strip().upper()
    mostrar_pantalla = False if resposta_pantalla == 'N' else True

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error: No s'ha pogut obrir el vídeo '{ruta_video}'.")
        return

    # Estimem quants frames analitzarem, per mostrar el progrés
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_estimats_final = total_frames_video // salt_frames

    print(f"\n--- INICIANT DETECCIÓ EN VÍDEO: {ruta_video} ---")
    print(f"S'analitzarà 1 de cada {salt_frames} frames a MÀXIMA qualitat.")

    if mostrar_pantalla:
        print("Prem la lletra 'q' a la finestra del vídeo si vols aturar l'anàlisi abans d'hora.\n")
    else:
        print("MODE CEC ACTIVAT: Prem Ctrl+C a la consola per aturar.\n")

    contador_frames_totals = 0  # Compta tots els frames llegits del vídeo
    frames_guardats = 0         # Compta només els que realment analitzem
    qualitat_jpeg = 50 if compressio_guardat else 100

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n\nS'ha arribat al final del vídeo.")
            break

        contador_frames_totals += 1

        # Si toca saltar aquest frame (segons l'opció N), el descartem sense analitzar
        if contador_frames_totals % salt_frames != 0:
            continue

        frames_guardats += 1

        # --- MISSATGE CLAR PER PANTALLA ---
        print(f"-> Analitzant frame: {frames_guardats} / ~{frames_estimats_final}", flush=True)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # IA SEMPRE A MÀXIMA QUALITAT (640)
        resultats = model(frame_rgb, size=640)
        dades = resultats.pandas().xywh[0]

        img_renderitzada_bgr = cv2.cvtColor(resultats.render()[0], cv2.COLOR_RGB2BGR)

        # Si l'usuari ho ha demanat, mostrem el vídeo amb les deteccions
        if mostrar_pantalla:
            cv2.imshow('Deteccio YOLOv5 Video (Prem "q" per sortir)', img_renderitzada_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nS'ha aturat l'anàlisi manualment.")
                break

        nom_base = f"frame_{frames_guardats:05d}"

        # GUARDAT DE LA IMATGE (amb o sense compressió)
        cv2.imwrite(os.path.join(carpeta_fotos_sortida, f"{nom_base}_detect.jpg"), img_renderitzada_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qualitat_jpeg])

        # GUARDAT DEL TXT amb les coordenades de cada objecte
        with open(os.path.join(carpeta_txt_sortida, f"{nom_base}.txt"), 'w') as fitxer:
            fitxer.write("Nom_Objecte, X_Centre(px), Y_Centre(px), Amplada(px), Alcada(px), Confianca\n")
            for _, fila in dades.iterrows():
                fitxer.write(f"{fila['name']}, {fila['xcenter']:.2f}, {fila['ycenter']:.2f}, {fila['width']:.2f}, {fila['height']:.2f}, {fila['confidence']:.2f}\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcés finalitzat. S'han analitzat i guardat {frames_guardats} imatges d'un total de {contador_frames_totals} frames originals.")

# ==========================================
# MENÚ PRINCIPAL
# ==========================================
# Punt d'entrada: pregunta a l'usuari què vol fer i crida la funció corresponent.
print("\n" + "="*45)
print("   SISTEMA DE DETECCIÓ YOLOv5 - TFG")
print("="*45)
print("1. Analitzar CÀMERA en temps real (últims 30s)")
print("2. Analitzar una CARPETA d'imatges estàtiques")
print("3. Analitzar un VÍDEO (.mp4, .avi, etc.)")
print("4. Sortir")

opcio = input("\nTria una opció (1, 2, 3 o 4): ")

if opcio in ['1', '2', '3']:
    # Per a qualsevol mode d'anàlisi, preguntem com vol guardar les fotos
    print("\n⚙️  CONFIGURACIÓ DE GUARDAT (L'anàlisi de la IA sempre és a màxima qualitat):")
    print("1. Ràpid (Defecte) - Guarda les fotos amb compressió (Estalvia targeta SD)")
    print("2. Qualitat Màxima - Guarda les fotos sense comprimir (Més lent)")
    opcio_comp = input("Tria una opció (1 o 2, Enter per 1): ").strip()

    compressio_activada = False if opcio_comp == '2' else True

    if opcio == '1':
        processar_camera(compressio_activada)
    elif opcio == '2':
        processar_carpeta(compressio_activada)
    elif opcio == '3':
        processar_video(compressio_activada)

elif opcio == '4':
    print("Sortint del programa...")
else:
    print("Opció no vàlida. Torna a executar el programa.")
