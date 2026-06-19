import os         # Gestió de fitxers i carpetes
import math       # Operacions matemàtiques (arrels, potències...)
import numpy as np  # Càlcul amb matrius i llistes de punts
import cv2        # OpenCV: llegir/dibuixar imatges
import itertools  # Per fer combinacions de parelles de cotxes (per als xocs)
import json       # Per llegir la configuració de les pistes
import time       # Per esperar entre lectures en mode temps real

# ==========================================
# QUÈ FA AQUEST PROGRAMA?
# ==========================================
# És l'ANALITZADOR de trajectòries. NO fa servir la càmera ni la IA directament:
# llegeix els fitxers .txt que ha generat el detector (Detectar_vectors_4) i, per
# cada fotograma, fa tota la "intel·ligència":
#   1) Segueix cada cotxe i li assigna un identificador (ID) estable (tracking).
#   2) Dibuixa un vector que prediu cap a on anirà el cotxe.
#   3) Detecta si dos vectors es creuen dins d'una zona de perill (col·lisió).
#   4) Calcula la velocitat de frenada i ho apunta a un fitxer de telemetria (CSV)
#      perquè ROS2 ho pugui llegir i controlar els cotxes.

# ==========================================
# 1. CONFIGURACIÓ GENERAL I CARPETES
# ==========================================
carpeta_txt = 'resultats_txt'                 # On hi ha les coordenades del detector
carpeta_fotos_base = 'resultats_fotos'        # On hi ha les fotos amb les caixes
carpeta_fotos_vectors = 'resultats_amb_vectors'  # On desarem les fotos amb els vectors
nom_fitxer_config = 'config_pistes.json'      # Geometria del circuit (del calibrador)
arxiu_telemetria = 'dades_pista.csv'          # Fitxer de sortida amb les ordres per a ROS2

os.makedirs(carpeta_fotos_vectors, exist_ok=True)  # Creem la carpeta de sortida si no hi és

# ==========================================
# 2. CÀRREGA AUTOMÀTICA DE PISTES (JSON)
# ==========================================
# Llegim la geometria de les pistes que es va dibuixar amb el calibrador.
dades_pistes = {}
if os.path.exists(nom_fitxer_config):
    with open(nom_fitxer_config, 'r') as f:
        dades_pistes = json.load(f)
    print(f"✅ Configuració carregada! Es monitoritzaran {len(dades_pistes)} pistes.")
else:
    print(f"⚠️ Alerta: No s'ha trobat '{nom_fitxer_config}'. Executa el calibrador si no ho has fet.")

mascares_pistes = {}   # Per a cada pista, una imatge en blanc i negre que marca per on passa
mapa_colors_pistes = {}  # Un color diferent per a cada pista (per dibuixar)
colors_pistes = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255)]

# ==========================================
# 3. CONFIGURACIÓ DE L'ALGORISME I RASTREIG
# ==========================================
distancia_maxima = 250   # Píxels màxims per considerar que un cotxe és el mateix entre frames
historial_maxim = 5      # Quants frames del passat recordem per calcular la trajectòria
paciencia_frames = 5     # Frames de cortesia abans d'esborrar un cotxe que ha desaparegut
escala_prediccio = 5     # Com d'allunyat dibuixem el vector de predicció (cap a on va)
factor_amplada = 0.6     # Amplada del vector de predicció
llindar_moviment = 1.0   # Moviment mínim perquè considerem que el cotxe està en marxa
frames_minims_alerta = 3  # Frames seguits amb risc abans de disparar l'alerta de xoc

classes_perill = ['Creuament4', 'CreuamentX', 'Estretament']  # Noms de les zones perilloses
paciencia_zones = 90     # Frames que recordem una zona de perill encara que no es vegi
memoria_zones = []       # Llista de zones de perill detectades actualment
memoria_colisions = {}   # Recorda quantes vegades seguides s'ha vist cada possible xoc

color_zona = (0, 165, 255)            # Taronja per a les zones de perill
color_prediccio_base = (255, 255, 255)  # Blanc per defecte si el cotxe no és a cap pista
color_alerta = (0, 0, 255)            # Vermell per a les alertes de xoc
gruix_linies = 2

objectes_actius = {}     # Diccionari amb tots els cotxes que estem seguint ara mateix
proper_id = 1            # Següent número d'identificador a assignar a un cotxe nou

# ==========================================
# 4. CONFIGURACIÓ DE FRENADA I TELEMETRIA (ROS2)
# ==========================================
v_maxima = 800       # Velocitat màxima (a la recta, sense perill)
D_max = 200.0        # Distància a partir de la qual comença a frenar progressivament
X_seguretat = 40.0   # Distància d'impacte: per sota d'això, frena del tot (velocitat 0)
memoria_velocitats = {}  # Recorda l'última velocitat enviada de cada cotxe (per no repetir)

# Creem el fitxer de telemetria i hi escrivim la capçalera de columnes
with open(arxiu_telemetria, 'w') as f_log:
    f_log.write("Pista,Velocitat_Objectiu,Frame,ID_Cotxe,Situacio_Xoc\n")

# --- FUNCIONS MATEMÀTIQUES AUXILIARS ---
# Distància en línia recta entre dos punts (teorema de Pitàgores)
def distancia_euclidiana(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Comprova si un punt cau dins d'un rectangle centrat en (rx, ry) i de mida rw x rh
def punt_dins_rectangle(punt, rect):
    px, py = punt
    rx, ry, rw, rh = rect
    return (rx - rw/2) <= px <= (rx + rw/2) and (ry - rh/2) <= py <= (ry + rh/2)

# ==========================================
# PREGUNTAR MODE D'EXECUCIÓ I DEFINIR GENERADOR
# ==========================================
print("Selecciona el mode de funcionament:")
print("1 - Analitzar la carpeta fins a finalitzar (Mode Normal)")
print("2 - Escoltar la carpeta contínuament (Mode Temps Real)")
opcio = input("Introdueix 1 o 2: ").strip()
mode_temps_real = (opcio == '2')

# Aquest "generador" va lliurant els noms dels .txt d'un en un. En mode temps real
# es queda escoltant la carpeta indefinidament i lliura els nous fitxers a mesura
# que el detector els va creant; en mode normal acaba quan s'han processat tots.
def generador_arxius(carpeta, en_viu):
    processats = set()  # Recorda quins fitxers ja hem tractat per no repetir-los
    while True:
        tots_els_arxius = sorted([f for f in os.listdir(carpeta) if f.endswith('.txt')])
        nous = [f for f in tots_els_arxius if f not in processats]

        if not nous:
            if en_viu:
                time.sleep(0.05)  # Esperem una mica i tornem a mirar si n'hi ha de nous
                continue
            else:
                break             # Mode normal: si no en queden, acabem

        for nom in nous:
            processats.add(nom)
            yield nom             # Lliurem el fitxer al bucle principal

# ==========================================
# INICI DEL PROCÉS (BUCLE PRINCIPAL)
# ==========================================
fitxer_log = open(arxiu_telemetria, 'a')  # Obrim el CSV en mode "afegir" (append)

# temps_t és el número de fotograma; va pujant amb cada fitxer processat
for temps_t, nom_arxiu in enumerate(generador_arxius(carpeta_txt, mode_temps_real)):
    ruta_txt = os.path.join(carpeta_txt, nom_arxiu)
    nom_base = os.path.splitext(nom_arxiu)[0]  # Nom sense l'extensió .txt

    # NETEJA: esborrem cotxes que fa estona que no es veuen i zones de perill caducades
    ids_a_esborrar = [obj_id for obj_id, dades in objectes_actius.items() if (temps_t - dades['ultim_vist']) > paciencia_frames]
    for obj_id in ids_a_esborrar: del objectes_actius[obj_id]
    memoria_zones = [z for z in memoria_zones if (temps_t - z['ultim_vist']) <= paciencia_zones]

    # Llegim el fitxer .txt (saltant la primera línia, que és la capçalera)
    with open(ruta_txt, 'r') as fitxer: linies = fitxer.readlines()[1:]
    nous_centres_cotxes = []

    # --- INTERPRETACIÓ DE LES DADES DE YOLO ---
    for linia in linies:
        dades = linia.strip().split(',')
        nom_classe = dades[0].strip()
        x, y, w, h = float(dades[1]), float(dades[2]), float(dades[3]), float(dades[4])
        if nom_classe == 'car':
            # Si és un cotxe, el guardem per fer-ne el seguiment
            nous_centres_cotxes.append((x, y, w, h))
        elif nom_classe in classes_perill:
            # Si és una zona de perill detectada per la IA, l'afegim/actualitzem a memòria.
            # Si ja teníem una zona molt a prop, suavitzem la seva posició (filtre 70/30)
            # per evitar que "balli" entre frames.
            ja_existeix = False
            for zona in memoria_zones:
                if distancia_euclidiana((x, y), (zona['x'], zona['y'])) < 100:
                    zona['x'] = (zona['x']*0.7 + x*0.3); zona['y'] = (zona['y']*0.7 + y*0.3)
                    zona['w'] = (zona['w']*0.7 + w*0.3); zona['h'] = (zona['h']*0.7 + h*0.3)
                    zona['ultim_vist'] = temps_t
                    ja_existeix = True; break
            if not ja_existeix: memoria_zones.append({'classe': nom_classe, 'x': x, 'y': y, 'w': w, 'h': h, 'ultim_vist': temps_t})

    # Carreguem la foto corresponent (la que ja porta les caixes del detector)
    imatge = cv2.imread(os.path.join(carpeta_fotos_base, f"{nom_base}_detect.jpg"))
    if imatge is None: continue  # Si no hi ha foto, saltem aquest frame

    # CREACIÓ DE LES MÀSCARES DE PISTA (només la primera vegada).
    # Convertim la geometria del JSON en imatges blanc/negre que diuen, per a cada
    # píxel, si pertany a una pista o no.
    if not mascares_pistes and dades_pistes:
        h_img, w_img = imatge.shape[:2]
        for idx, (nom_p, info_pista) in enumerate(dades_pistes.items()):
            mascara_temp = np.zeros((h_img, w_img), dtype=np.uint8)
            for segment in info_pista["segments"]:
                cv2.polylines(mascara_temp, [np.array(segment)], False, 255, info_pista["gruix"])
            mascares_pistes[nom_p] = mascara_temp
            nom_net = nom_p.replace("_", " ").capitalize()
            mapa_colors_pistes[nom_net] = colors_pistes[idx % len(colors_pistes)]

    # Dibuixem un rectangle taronja per cada zona de perill activa
    for zona in memoria_zones:
        zx, zy, zw, zh = int(zona['x']), int(zona['y']), int(zona['w']), int(zona['h'])
        cv2.rectangle(imatge, (zx - zw//2, zy - zh//2), (zx + zw//2, zy + zh//2), color_zona, 2)

    # ------------------------------------------------------
    # TRACKING: ASSIGNACIÓ D'IDS AMB PREDICCIÓ
    # ------------------------------------------------------
    # Per saber si un cotxe d'aquest frame és el mateix del frame anterior, predim
    # on hauria d'estar (segons cap a on anava) i el comparem amb les deteccions noves.
    possibles_emparellaments = []
    for index_nou, dades_cotxe in enumerate(nous_centres_cotxes):
        x_c, y_c, w_c, h_c = dades_cotxe
        centre_real = (x_c, y_c)
        for obj_id, dades_obj in objectes_actius.items():
            historial = dades_obj['historial']
            if len(historial) >= 2:
                # Estimem la velocitat (diferència entre les dues últimes posicions)
                # i predim la posició següent
                vx = historial[-1][1] - historial[-2][1]; vy = historial[-1][2] - historial[-2][2]
                pred_x = historial[-1][1] + vx; pred_y = historial[-1][2] + vy
            else:
                pred_x, pred_y = historial[-1][1], historial[-1][2]
            dist = distancia_euclidiana((pred_x, pred_y), centre_real)
            # Només considerem parelles que estiguin prou a prop
            if dist < distancia_maxima: possibles_emparellaments.append((dist, obj_id, index_nou))

    # Ordenem per distància: primer emparellem els que tenen menys dubte
    possibles_emparellaments.sort(key=lambda x: x[0])
    ids_assignats = set()
    deteccions_assignades = set()
    objectes_actualitzats_aquest_frame = []

    # Assignem cada cotxe nou a l'ID més proper, sense repetir IDs ni deteccions
    for dist, obj_id, index_nou in possibles_emparellaments:
        if obj_id not in ids_assignats and index_nou not in deteccions_assignades:
            ids_assignats.add(obj_id); deteccions_assignades.add(index_nou)
            objectes_actualitzats_aquest_frame.append(obj_id)
            x_c, y_c, w_c, h_c = nous_centres_cotxes[index_nou]
            w_ver = min(w_c, h_c)
            # Afegim la nova posició a l'historial del cotxe
            objectes_actius[obj_id]['historial'].append((temps_t, x_c, y_c, w_ver, w_c, h_c))
            objectes_actius[obj_id]['ultim_vist'] = temps_t
            if len(objectes_actius[obj_id]['historial']) > historial_maxim: objectes_actius[obj_id]['historial'].pop(0)

    # Les deteccions que no han trobat parella són cotxes NOUS: els donem un ID nou
    for index_nou, dades_cotxe in enumerate(nous_centres_cotxes):
        if index_nou not in deteccions_assignades:
            x_c, y_c, w_c, h_c = dades_cotxe
            w_ver = min(w_c, h_c)
            objectes_actius[proper_id] = {'historial': [(temps_t, x_c, y_c, w_ver, w_c, h_c)], 'ultim_vist': temps_t, 'pista_guardada': 'Fora'}
            objectes_actualitzats_aquest_frame.append(proper_id)
            proper_id += 1

    print(f"Processant {nom_base}...", end="\r")
    poligons_prediccio_cotxes = {}

    # Estat per defecte de cada cotxe: sense xoc i a velocitat màxima
    estat_ros2 = {obj_id: {'xoc': 'No', 'vel': v_maxima} for obj_id in objectes_actualitzats_aquest_frame}

    # ------------------------------------------------------
    # IDENTIFICAR CARRIL I CONSTRUIR EL VECTOR DE PREDICCIÓ
    # ------------------------------------------------------
    for obj_id in objectes_actualitzats_aquest_frame:
        historial = objectes_actius[obj_id]['historial']
        pista_recordada = objectes_actius[obj_id].get('pista_guardada', 'Fora')
        _, cx, cy, w_ver, w_orig, h_orig = historial[-1]

        # Mirem si el cotxe és dins d'una zona de perill
        dins_zona_conflictiva = False
        for zona in memoria_zones:
            if punt_dins_rectangle((cx, cy), (zona['x'], zona['y'], zona['w'], zona['h'])):
                dins_zona_conflictiva = True; break

        # Mirem sobre quina(es) pista(es) trepitja el centre del cotxe
        pistes_trepitjades = []
        for nom_p, mascara in mascares_pistes.items():
            if mascara[int(cy), int(cx)] == 255: pistes_trepitjades.append(nom_p.replace("_", " ").capitalize())

        # Decidim a quina pista pertany. Si és en una zona conflictiva, mantenim la
        # pista que ja tenia (per no confondre'ns a les cruïlles).
        nom_pista = "Fora"
        if dins_zona_conflictiva and pista_recordada != "Fora": nom_pista = pista_recordada
        elif len(pistes_trepitjades) > 0:
            if pista_recordada in pistes_trepitjades: nom_pista = pista_recordada
            else: nom_pista = pistes_trepitjades[0]
            objectes_actius[obj_id]['pista_guardada'] = nom_pista
        else: nom_pista = "Fora"

        color_pista = mapa_colors_pistes.get(nom_pista, color_prediccio_base)

        # Necessitem almenys 3 posicions per estimar una trajectòria fiable
        if len(historial) >= 3:
            t_v, x_v, y_v, _, _, _ = zip(*historial)
            # polyfit ajusta una recta a les posicions per estimar la velocitat (vx, vy)
            vx, _ = np.polyfit(t_v, x_v, 1); vy, _ = np.polyfit(t_v, y_v, 1)
            modul = math.sqrt(vx**2 + vy**2)
            if modul > llindar_moviment:
                # COTXE EN MOVIMENT: construïm un vector ("banya") cap a on es dirigeix
                nx = -vy / modul; ny = vx / modul; meitat_w = (w_ver * factor_amplada) / 2
                p1 = [cx + nx*meitat_w, cy + ny*meitat_w]; p2 = [cx - nx*meitat_w, cy - ny*meitat_w]
                xf = cx + vx*escala_prediccio; yf = cy + vy*escala_prediccio
                p3 = [xf - nx*meitat_w, yf - ny*meitat_w]; p4 = [xf + nx*meitat_w, yf + ny*meitat_w]
                pts = np.array([p1, p2, p3, p4], dtype=np.float32)
            else:
                # COTXE ATURAT: fem servir una caixa rectangular al voltant del cotxe
                mw = w_orig / 2; mh = h_orig / 2
                pts = np.array([[cx-mw, cy-mh], [cx+mw, cy-mh], [cx+mw, cy+mh], [cx-mw, cy+mh]], dtype=np.float32)

            # convexHull tanca els punts en un polígon i el dibuixem amb el color de la pista
            poligon = cv2.convexHull(pts)
            poligons_prediccio_cotxes[obj_id] = poligon
            cv2.polylines(imatge, [np.int32(poligon)], True, color_pista, gruix_linies)
            cv2.putText(imatge, f"ID:{obj_id} ({nom_pista})", (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pista, 2)

    # ------------------------------------------------------
    # COMPROVAR COL·LISIONS I APLICAR L'EQUACIÓ DE FRENADA
    # ------------------------------------------------------
    colisions_d_aquest_frame = {}
    # Comparem tots els cotxes de dos en dos per veure si els seus vectors es creuen
    for id_a, id_b in itertools.combinations(poligons_prediccio_cotxes.keys(), 2):
        poly_a = poligons_prediccio_cotxes[id_a]
        poly_b = poligons_prediccio_cotxes[id_b]
        intersecten, area_interseccio = cv2.intersectConvexConvex(poly_a, poly_b)

        if intersecten > 0:
            # Hi ha encreuament: calculem el punt central del xoc
            centre_xoc = np.mean(area_interseccio, axis=0).flatten()
            punt_xoc = (centre_xoc[0], centre_xoc[1])
            # Només ens importa si el xoc cau dins d'una zona de perill
            dins_zona = False
            for zona in memoria_zones:
                if punt_dins_rectangle(punt_xoc, (zona['x'], zona['y'], zona['w'], zona['h'])):
                    dins_zona = True; break
            if dins_zona:
                parella = (min(id_a, id_b), max(id_a, id_b))
                # Comptem quants frames seguits portem veient aquest mateix risc
                frames_abans = memoria_colisions.get(parella, {}).get('frames', 0)
                colisions_d_aquest_frame[parella] = {'frames': frames_abans + 1, 'punt': punt_xoc}

    memoria_colisions = colisions_d_aquest_frame

    # Per cada possible xoc detectat, si s'ha confirmat prou frames seguits, actuem
    for parella, dades_col in memoria_colisions.items():
        punt_xoc = dades_col['punt']
        if dades_col['frames'] >= frames_minims_alerta:
            px, py = int(punt_xoc[0]), int(punt_xoc[1])
            # Dibuixem l'alerta visual (cercle i text vermell)
            cv2.circle(imatge, (px, py), 30, color_alerta, 4)
            cv2.putText(imatge, "!!! ALERTA XOC !!!", (px - 80, py - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_alerta, 2)

            # Calculem la velocitat de frenada per als dos cotxes implicats
            for id_cotxe in parella:
                if id_cotxe in estat_ros2:
                    cx, cy = objectes_actius[id_cotxe]['historial'][-1][1:3]
                    distancia = distancia_euclidiana((cx, cy), punt_xoc)

                    # EQUACIÓ DE FRENADA (rampa lineal segons la distància al xoc):
                    if distancia <= X_seguretat:
                        v_nova = 0          # Molt a prop -> frena del tot
                    elif distancia >= D_max:
                        v_nova = v_maxima   # Lluny -> manté la velocitat màxima
                    else:
                        # Intermedi -> redueix la velocitat de manera proporcional
                        v_nova = v_maxima * ((distancia - X_seguretat) / (D_max - X_seguretat))

                    v_nova = int(max(0, min(v_maxima, v_nova)))  # Limitem entre 0 i v_maxima
                    estat_ros2[id_cotxe]['xoc'] = 'Si'
                    # Ens quedem amb la velocitat més baixa (la més prudent)
                    estat_ros2[id_cotxe]['vel'] = min(estat_ros2[id_cotxe]['vel'], v_nova)

    # =======================================================
    # ESCRIPTURA A DISC: LÒGICA D'EMISSIÓ PER A ROS2
    # =======================================================
    for obj_id in objectes_actualitzats_aquest_frame:
        pista_act = objectes_actius[obj_id].get('pista_guardada', 'Fora')
        estat_xoc = estat_ros2[obj_id]['xoc']
        vel_final = estat_ros2[obj_id]['vel']

        if pista_act == 'Fora':
            continue  # Si el cotxe no és sobre cap pista, no enviem res

        vel_anterior = memoria_velocitats.get(obj_id, None)

        # Per no saturar ROS2 amb missatges repetits: si el cotxe ja anava a velocitat
        # màxima i sense risc, i abans també, no cal tornar-ho a escriure.
        if estat_xoc == 'No' and vel_final == v_maxima:
            if vel_anterior == v_maxima:
                continue

        # Convertim el nom "Pista 1" en simplement "1" (el que espera ROS2)
        numero_pista = pista_act.replace("Pista ", "").strip()

        # Escrivim una fila al CSV: Pista, Velocitat_Objectiu, Frame, ID_Cotxe, Situacio_Xoc
        fitxer_log.write(f"{numero_pista},{vel_final},{nom_base},{obj_id},{estat_xoc}\n")

        memoria_velocitats[obj_id] = vel_final

    # Forcem que l'ordinador escrigui ja les dades al disc (no es quedin a la memòria)
    fitxer_log.flush()

    # Desem la foto amb els vectors i alertes dibuixats
    cv2.imwrite(os.path.join(carpeta_fotos_vectors, f"{nom_base}_vector.jpg"), imatge)

# Tanquem el fitxer de telemetria en acabar
fitxer_log.close()
print(f"\n\nProcés completat! Fitxer de telemetria generat: '{arxiu_telemetria}'")
