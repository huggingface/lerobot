# 🤖 Guía Rápida: Teleoperación Bi-Manual Piper con SO100

Guía práctica para teleoperar 2 robots Piper usando 2 brazos SO100 modificados (7 DOF) como líderes.

---

## 📋 Hardware

- **2 Robots Piper** (followers) - USB-to-CAN
- **2 Brazos SO100 modificados** (leaders) - 7 motores/brazo (IDs 1-7)
- **2 Adaptadores USB-to-CAN** (Geschwister Schneider)
- **Cámaras** (opcional)

**Mapeo de Juntas:**
```
Motor SO100        →  Piper Joint
Motor 1 (ID 1)     →  joint_0 (shoulder_pan)
Motor 2 (ID 2)     →  joint_1 (shoulder_lift)
Motor 3 (ID 3)     →  joint_2 (elbow_flex)
Motor 4 (ID 4)     →  joint_3 (forearm_roll) ⭐ Nuevo motor
Motor 5 (ID 5)     →  joint_4 (wrist_flex)
Motor 6 (ID 6)     →  joint_5 (wrist_roll)
Motor 7 (ID 7)     →  joint_6 (gripper)
```

---

## 🚀 GUÍA PASO A PASO

### **PASO 1: Activar Entorno**

```bash
conda activate lerobot_piper
cd /home/mbrq/NONHUMAN/TELEOPERATION/lerobot
```

---

### **PASO 2: Verificar Hardware SO100 (Puertos Seriales)**

```bash
# Ver puertos seriales de los SO100
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

**Salida esperada:**
```
crw-rw---- 1 root dialout 166, 0 Nov  3 20:20 /dev/ttyACM0
crw-rw---- 1 root dialout 166, 1 Nov  3 20:20 /dev/ttyACM1
```

✅ 2 SO100 detectados

**Si no tienes permisos:**
```bash
sudo usermod -a -G dialout $USER
# Cerrar sesión y volver a entrar
```

---

### **PASO 3: Verificar Hardware Piper (Adaptadores CAN)**

```bash
# Ver adaptadores USB-to-CAN
lsusb | grep -i "CAN\|gs_usb\|Schneider\|OpenMoko"
```

**Salida esperada:**
```
Bus 001 Device 080: ID 1d50:606f OpenMoko, Inc. Geschwister Schneider CAN adapter
Bus 001 Device 086: ID 1d50:606f OpenMoko, Inc. Geschwister Schneider CAN adapter
```

✅ 2 Pipers detectados

---

### **PASO 4: Ver Interfaces CAN**

```bash
# Ver interfaces CAN creadas
ip link show type can
```

**Salida esperada:**
```
22: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT
23: can1: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT
```

- `state DOWN` → Necesita activarse ❌
- `state UP` → Listo para usar ✅

---

### **PASO 5: Identificar Direcciones USB de las Interfaces CAN**

```bash
for iface in $(ip -br link show type can | awk '{print $1}'); do
    BUS_INFO=$(sudo ethtool -i "$iface" 2>/dev/null | grep "bus-info" | awk '{print $2}')
    echo "Interfaz: $iface -> Puerto USB: $BUS_INFO"
done
```

**Salida esperada:**
```
Interfaz: can0 -> Puerto USB: 1-3.3:1.0
Interfaz: can1 -> Puerto USB: 1-1:1.0
```

✅ **Guarda estas direcciones USB** (necesarias para el siguiente paso)

---

### **PASO 6: Activar Interfaces CAN (ponerlas UP)**

**⚠️ Usa las direcciones USB del paso anterior**

```bash
# Activar can0 (ajusta la dirección USB según tu sistema)
sudo bash ~/miniconda3/envs/lerobot_piper/lib/python3.11/site-packages/piper_sdk/can_activate.sh can0 1000000 1-3.3:1.0

# Activar can1 (ajusta la dirección USB según tu sistema)
sudo bash ~/miniconda3/envs/lerobot_piper/lib/python3.11/site-packages/piper_sdk/can_activate.sh can1 1000000 1-1:1.0
```

**Verificar que estén UP:**
```bash
ip link show can0 | grep "state"
ip link show can1 | grep "state"
```

**Debe mostrar:**
```
state UP
state UP
```

✅ Interfaces CAN activadas

---

### **PASO 7: Calibrar Brazos Líderes (solo primera vez)**

**⚠️ Este paso solo se hace UNA VEZ. La calibración se guarda permanentemente.**

```bash
python -m lerobot.calibrate \
    --teleop.type=bi_so100_piper_leader \
    --teleop.left_arm_port=/dev/ttyACM0 \
    --teleop.right_arm_port=/dev/ttyACM1 \
    --teleop.id=my_bi_piper_leader
```

**Proceso interactivo:**

#### **Para BRAZO IZQUIERDO:**

1. **Mensaje:** "Mueve el brazo a la mitad de su rango y presiona ENTER..."
   - ✋ Mover brazo izquierdo a posición intermedia → **ENTER**

2. **Mensaje:** "Mueve todas las juntas (menos 'wrist_roll') a través de su rango completo. ENTER para parar..."
   - ✋ Mover cada junta desde su mínimo hasta su máximo:
     - shoulder_pan
     - shoulder_lift
     - elbow_flex
     - forearm_roll
     - wrist_flex
     - gripper
   - **ENTER** cuando termines

3. Verás una tabla con los rangos registrados

#### **Para BRAZO DERECHO:**

Repite el mismo proceso (el sistema lo pedirá automáticamente)

**Confirmación:**
```
Calibración guardada en ~/.cache/huggingface/lerobot/calibration/teleoperators/so100_piper_leader/my_bi_piper_leader_left.json
Calibración guardada en ~/.cache/huggingface/lerobot/calibration/teleoperators/so100_piper_leader/my_bi_piper_leader_right.json
```

✅ Calibración completada

---

### **PASO 8: Teleoperar (¡El momento de la verdad!)**

#### **Opción A: Sin cámaras (más simple)**

```bash
python -m lerobot.teleoperate \
    --robot.type=bi_piper_follower \
    --robot.left_port=can0 \
    --robot.right_port=can1 \
    --robot.id=my_bi_piper \
    --teleop.type=bi_so100_piper_leader \
    --teleop.left_arm_port=/dev/ttyACM0 \
    --teleop.right_arm_port=/dev/ttyACM1 \
    --teleop.id=my_bi_piper_leader \
    --fps=60
```

#### **Opción B: Con cámaras y visualización**

```bash
python -m lerobot.teleoperate \
    --robot.type=bi_piper_follower \
    --robot.left_port=can0 \
    --robot.right_port=can1 \
    --robot.id=my_bi_piper \
    --robot.cameras='{
        left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
        top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
        right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
    }' \
    --teleop.type=bi_so100_piper_leader \
    --teleop.left_arm_port=/dev/ttyACM0 \
    --teleop.right_arm_port=/dev/ttyACM1 \
    --teleop.id=my_bi_piper_leader \
    --display_data=true \
    --fps=60
```

**Para detener:** `Ctrl+C`

---

## ✅ CHECKLIST RÁPIDO

```bash
# 1. Entorno activado
conda activate lerobot_piper

# 2. Ver SO100 (2 puertos esperados)
ls -l /dev/ttyACM*

# 3. Ver Pipers (2 adaptadores esperados)
lsusb | grep -i CAN

# 4. Ver interfaces CAN (2 interfaces esperadas)
ip link show type can

# 5. Identificar USB
for iface in $(ip -br link show type can | awk '{print $1}'); do BUS_INFO=$(sudo ethtool -i "$iface" 2>/dev/null | grep "bus-info" | awk '{print $2}'); echo "$iface -> $BUS_INFO"; done

# 6. Activar CAN (usar direcciones del paso 5)
sudo bash ~/miniconda3/envs/lerobot_piper/lib/python3.11/site-packages/piper_sdk/can_activate.sh can0 1000000 1-X.X:1.0
sudo bash ~/miniconda3/envs/lerobot_piper/lib/python3.11/site-packages/piper_sdk/can_activate.sh can1 1000000 1-Y.Y:1.0

# 7. Verificar estado UP
ip link show can0 | grep "state"
ip link show can1 | grep "state"

# 8. Calibrar (solo primera vez)
python -m lerobot.calibrate --teleop.type=bi_so100_piper_leader --teleop.left_arm_port=/dev/ttyACM0 --teleop.right_arm_port=/dev/ttyACM1 --teleop.id=my_bi_piper_leader

# 9. Teleoperar
python -m lerobot.teleoperate --robot.type=bi_piper_follower --robot.left_port=can0 --robot.right_port=can1 --robot.id=my_bi_piper --teleop.type=bi_so100_piper_leader --teleop.left_arm_port=/dev/ttyACM0 --teleop.right_arm_port=/dev/ttyACM1 --teleop.id=my_bi_piper_leader --fps=60
```

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### ❌ Problema: Interface CAN en estado DOWN

**Solución:**
```bash
# Activar con dirección USB (obtener con ethtool -i canX)
sudo bash ~/miniconda3/envs/lerobot_piper/lib/python3.11/site-packages/piper_sdk/can_activate.sh can0 1000000 1-3.3:1.0
```

### ❌ Problema: Permission denied en /dev/ttyACM*

**Solución:**
```bash
sudo usermod -a -G dialout $USER
# Cerrar sesión y volver a entrar
```

### ❌ Problema: Solo 1 Piper detectado

**Verificar:**
```bash
lsusb | grep -i CAN
```
Si solo aparece 1 línea → revisar conexión física del segundo Piper

### ❌ Problema: "Failed to initialize Piper SDK"

**Verificar que interfaces estén UP:**
```bash
ip link show can0 can1 | grep state
```
Ambos deben mostrar `state UP`

### ❌ Problema: Motor ID no detectado en SO100

**Verificar IDs de motores:**
```bash
python -m lerobot.find_port
# Revisar que los 7 motores (IDs 1-7) estén presentes
```

### ❌ Problema: Puertos USB cambiaron después de desconectar

**Re-verificar puertos:**
```bash
ls -l /dev/ttyACM*  # SO100
ip link show type can  # Piper
```

---

## 📝 NOTAS IMPORTANTES

⚠️ **IMPORTANTE:**
1. **Las interfaces CAN se desactivan al reiniciar** → Activa `can0` y `can1` cada sesión
2. **La calibración se guarda permanentemente** → Solo calibrar una vez (a menos que muevas los motores o cambies el setup)
3. **Los puertos USB pueden cambiar** → Siempre verificar `/dev/ttyACM*` antes de teleoperar
4. **El bitrate debe ser 1000000** para comunicación Piper
5. **Motor ID 6 (wrist_roll)** → No se calibra manualmente, rango automático 0-4095

---

## 🎓 REFERENCIAS

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [Piper SDK](https://github.com/agilexrobotics/piper_sdk)
- [SO100 Arm](https://github.com/TheRobotStudio/SO-ARM100)

---

**✅ TELEOPERACIÓN EXITOSA**

Este documento fue validado con una teleoperación bi-manual exitosa el 4 de noviembre de 2025.

**Creado:** Noviembre 2025  
**Última actualización:** Noviembre 2025  
**Proyecto:** TELEOPERATION - NONHUMAN Lab

