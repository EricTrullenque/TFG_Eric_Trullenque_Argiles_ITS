# ==========================================
# PONT DE COMUNICACIÓ: UDP  ->  ROS2
# ==========================================
# Aquest petit programa fa de "traductor" o pont entre dos mons:
#   1) El programa de visió (visio_udp_...) que detecta els cotxes i decideix
#      a quina velocitat ha d'anar cada pista. Aquest envia ordres per UDP.
#   2) El robot/sistema ROS2, que és qui realment controla els cotxes.
# La feina d'aquest fitxer és escoltar els missatges UDP que arriben i
# reenviar-los cap a ROS2 en el format que ROS2 entén.

import rclpy                                  # Llibreria oficial de ROS2 per a Python
from rclpy.node import Node                   # Un "Node" és cada peça del sistema ROS2
from std_msgs.msg import Int32MultiArray      # Tipus de missatge: una llista de números enters
import socket                                 # Per rebre dades per la xarxa (UDP)


# ==========================================
# 1. EL NODE DE CONTROL
# ==========================================
# Aquesta classe representa la peça de ROS2 que escolta el port UDP i
# publica les ordres dins del sistema ROS2.
class UDPControl(Node):
    def __init__(self):
        super().__init__('udp_control_node')  # Li posem nom al node dins de ROS2

        # Creem un "altaveu" (publisher) que cridarà ordres pel canal anomenat
        # 'vel_pista_escalextric'. El número 10 és la mida de la cua de missatges.
        self.publisher_ = self.create_publisher(Int32MultiArray, 'vel_pista_escalextric', 10)

        # Obrim un port de comunicació interna (UDP) per rebre dades
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # CANVI CLAU: Només escoltem a localhost (la mateixa màquina) per evitar
        # rebre "brossa" o paquets aliens que circulin per la xarxa WiFi/externa.
        self.sock.bind(("127.0.0.1", 9999))

        # No bloquejant: si no hi ha dades, el programa no es queda penjat esperant.
        self.sock.setblocking(False)

        # Posem una alarma que crida la funció check_udp() cada 0,01 segons (100 cops/seg)
        # per anar mirant si ha arribat algun missatge nou.
        self.timer = self.create_timer(0.01, self.check_udp)
        self.get_logger().info('🚀 Node UDP iniciat! Escoltant a 127.0.0.1:9999 (Mode aïllat)...')

    # ------------------------------------------------------
    # Funció que comprova si ha arribat un missatge UDP nou
    # ------------------------------------------------------
    def check_udp(self):
        try:
            # Intentem llegir fins a 1024 bytes del port. Si no hi ha res, saltarà l'excepció.
            data, _ = self.sock.recvfrom(1024)

            # Intentem convertir els bytes rebuts a text i separar-lo per la coma
            # El format esperat és "pista,velocitat" (ex: "1,500")
            dades = data.decode('utf-8').split(',')

            # Ens assegurem que el missatge tingui exactament 2 parts (Pista i Velocitat)
            if len(dades) == 2:
                pista = int(dades[0])
                velocitat = int(dades[1])

                # Empaquetem els dos números i els publiquem al canal de ROS2
                msg = Int32MultiArray()
                msg.data = [pista, velocitat]
                self.publisher_.publish(msg)

                self.get_logger().info(f'📡 Rebut UDP i enviat a ROS: Pista {pista} a {velocitat}')

        # --- GESTIÓ D'ERRORS (silenciosa per no omplir la pantalla) ---
        except BlockingIOError:
            pass  # No hi ha dades noves ara mateix: és normal, no fem res
        except UnicodeDecodeError:
            pass  # CANVI CLAU: Ignorem silenciosament la brossa binària d'altres programes
        except Exception as e:
            # Qualsevol altre error inesperat l'ignorem perquè el node no caigui
            pass


# ==========================================
# 2. ARRANCADA DEL PROGRAMA
# ==========================================
def main(args=None):
    rclpy.init(args=args)        # Engeguem el sistema ROS2
    node = UDPControl()          # Creem el nostre node de control
    try:
        rclpy.spin(node)         # El deixem funcionant en bucle infinit, escoltant
    except KeyboardInterrupt:
        pass                     # Si l'usuari prem Ctrl+C, sortim amb elegància
    finally:
        # Apagat net: tanquem el port i alliberem els recursos de ROS2
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


# Aquesta línia fa que main() només s'executi si llancem aquest fitxer directament
if __name__ == '__main__':
    main()
