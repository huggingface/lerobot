# HX-30HM Serial Bus Servo — Communication Protocol

Hiwonder HX-30HM is a serial bus servo using the **STS/SMS half-duplex UART protocol** (compatible with Feetech STS3215 register layout). All communication uses a single-wire half-duplex serial bus at TTL voltage levels.

---

## 1. Physical Interface

| Parameter | Value |
|-----------|-------|
| Interface | Half-duplex UART (single wire) |
| Voltage Level | TTL 3.3V / 5V tolerant |
| Default Baud Rate | 1,000,000 bps |
| Supported Baud Rates | 1M / 500K / 250K / 128K / 115200 / 76800 / 57600 / 38400 |
| Connector | 3-pin JST (GND / VCC / Signal) |

---

## 2. Packet Structure

### 2.1 Instruction Packet (Host → Servo)

```
[0xFF] [0xFF] [ID] [LEN] [INST] [P1] [P2] ... [Pn] [CHKSUM]
```

| Field    | Offset | Description |
|----------|--------|-------------|
| Header   | 0–1    | Always `0xFF 0xFF` |
| ID       | 2      | Servo ID (0–252); 254 = broadcast |
| LEN      | 3      | Number of remaining bytes: `N_params + 2` |
| INST     | 4      | Instruction code (see §3) |
| Params   | 5…     | Instruction-specific parameters |
| CHKSUM   | last   | `(~(ID + LEN + INST + P1 + … + Pn)) & 0xFF` |

### 2.2 Status Packet (Servo → Host)

```
[0xFF] [0xFF] [ID] [LEN] [ERROR] [D1] [D2] ... [Dn] [CHKSUM]
```

| Field    | Offset | Description |
|----------|--------|-------------|
| Header   | 0–1    | Always `0xFF 0xFF` |
| ID       | 2      | Responding servo ID |
| LEN      | 3      | `N_data + 2` |
| ERROR    | 4      | Error bitmask (see §5) |
| Data     | 5…     | Read data bytes |
| CHKSUM   | last   | Same checksum formula |

Checksum formula (both directions):
```
CHKSUM = (~(sum of bytes from ID through last param)) & 0xFF
```

---

## 3. Instruction Set

| Instruction   | Code | Description |
|---------------|------|-------------|
| PING          | 0x01 | Query servo presence; returns status packet |
| READ          | 0x02 | Read N bytes from register address |
| WRITE         | 0x03 | Write N bytes to register address (immediate) |
| REG_WRITE     | 0x04 | Write to buffer; execute on ACTION |
| ACTION        | 0x05 | Execute all pending REG_WRITE commands |
| RESET         | 0x06 | Reset servo to factory defaults |
| SYNC_READ     | 0x82 | Read same register from multiple servos |
| SYNC_WRITE    | 0x83 | Write to same register on multiple servos |

### PING
```
TX: FF FF [ID] 02 01 [CHK]
RX: FF FF [ID] 02 [ERR] [CHK]
```

### READ
```
TX: FF FF [ID] 04 02 [ADDR] [LEN] [CHK]
RX: FF FF [ID] [LEN+2] [ERR] [D1] ... [Dn] [CHK]
```

### WRITE
```
TX: FF FF [ID] [LEN+3] 03 [ADDR] [D1] ... [Dn] [CHK]
RX: FF FF [ID] 02 [ERR] [CHK]
```

### SYNC_READ (broadcast)
```
TX: FF FF FE [N+4] 82 [ADDR] [LEN] [ID1] [ID2] ... [IDn] [CHK]
RX: (one status packet per servo, in ID order)
```

### SYNC_WRITE (broadcast, no response)
```
TX: FF FF FE [N*(LEN+1)+4] 83 [ADDR] [LEN] [ID1] [D1..Dn] [ID2] [D1..Dn] ... [CHK]
```

---

## 4. Register Map

All multi-byte values are **little-endian** (low byte first).

### EEPROM Area (persistent, lock before write in normal operation)

| Address | Name | Size | R/W | Default | Range | Description |
|---------|------|------|-----|---------|-------|-------------|
| 0x03 | Model_Number_L | 1 | R | — | — | Model number low byte |
| 0x04 | Model_Number_H | 1 | R | — | — | Model number high byte |
| 0x05 | ID | 1 | R/W | 1 | 0–253 | Servo ID |
| 0x06 | Baud_Rate | 1 | R/W | 0 | 0–7 | See baud rate table below |
| 0x07 | Return_Delay | 1 | R/W | 0 | 0–254 | Return delay (×2 μs) |
| 0x08 | Response_Status_Level | 1 | R/W | 1 | 0–1 | 0=no status return, 1=return on read |
| 0x09 | Min_Angle_Limit_L | 2 | R/W | 0 | 0–4095 | Min position (little-endian) |
| 0x0B | Max_Angle_Limit_L | 2 | R/W | 4095 | 0–4095 | Max position |
| 0x0D | Max_Temperature_Limit | 1 | R/W | 70 | 0–100 | Max temp (°C), triggers error |
| 0x0E | Max_Input_Voltage | 1 | R/W | 90 | 0–254 | Max voltage (×0.1 V) |
| 0x0F | Min_Input_Voltage | 1 | R/W | 60 | 0–254 | Min voltage (×0.1 V) |
| 0x10 | Max_Torque_Limit | 2 | R/W | 1000 | 0–1000 | Max torque (0–1000 = 0–100%) |
| 0x12 | Phase | 1 | R/W | 0 | — | Phase configuration |
| 0x13 | Unloading_Condition | 1 | R/W | 44 | — | Overload unload condition |
| 0x14 | LED_Alarm_Condition | 1 | R/W | 47 | — | LED alarm bitmask |
| 0x15 | P_Coefficient | 1 | R/W | 32 | 0–254 | PID proportional gain |
| 0x16 | D_Coefficient | 1 | R/W | 32 | 0–254 | PID derivative gain |
| 0x17 | I_Coefficient | 1 | R/W | 0 | 0–254 | PID integral gain |
| 0x18 | Minimum_Startup_Force | 2 | R/W | 0 | 0–1000 | Min startup torque |
| 0x1A | CW_Dead_Zone | 1 | R/W | 0 | 0–32 | CW dead zone |
| 0x1B | CCW_Dead_Zone | 1 | R/W | 0 | 0–32 | CCW dead zone |
| 0x1C | Protection_Current | 2 | R/W | 500 | 0–511 | Overload protection current |
| 0x1E | Angular_Resolution | 1 | R/W | 1 | — | Angular resolution |
| 0x1F | Offset_L | 2 | R/W | 0 | — | Position offset (signed) |
| 0x21 | Operating_Mode | 1 | R/W | 0 | 0–3 | See operating mode table |
| 0x22 | Protective_Torque | 1 | R/W | 20 | 0–100 | Torque on overload (%) |
| 0x23 | Protection_Time | 1 | R/W | 200 | 0–254 | Overload time before protection |
| 0x24 | Overload_Torque | 1 | R/W | 25 | 0–100 | Overload torque threshold (%) |
| 0x25 | Speed_closed_loop_P | 1 | R/W | 10 | 0–254 | Speed loop P gain |
| 0x26 | Over_Current_Protection_Time | 1 | R/W | 20 | 0–254 | Overcurrent protection delay |
| 0x27 | Velocity_closed_loop_I | 1 | R/W | 0 | 0–254 | Speed loop I gain |

### RAM Area (volatile, active control)

| Address | Name | Size | R/W | Default | Range | Description |
|---------|------|------|-----|---------|-------|-------------|
| 0x28 | Torque_Enable | 1 | R/W | 0 | 0–1 | 0=off, 1=on |
| 0x29 | Acceleration | 1 | R/W | 0 | 0–254 | Acceleration (0=uncontrolled) |
| 0x2A | Goal_Position_L | 2 | R/W | — | 0–4095 | Target position (little-endian) |
| 0x2C | Goal_Time_L | 2 | R/W | 0 | 0–65535 | Time to reach goal (ms) |
| 0x2E | Goal_Speed_L | 2 | R/W | 0 | 0–32767 | Max speed to goal (steps/s) |
| 0x30 | Torque_Limit_L | 2 | R/W | — | 0–1000 | Runtime torque limit |
| 0x32 | Lock | 1 | R/W | 0 | 0–1 | 1=lock EEPROM writes |
| 0x38 | Present_Position_L | 2 | R | — | 0–4095 | Current position |
| 0x3A | Present_Speed_L | 2 | R | — | — | Current speed (signed) |
| 0x3C | Present_Load_L | 2 | R | — | — | Current load |
| 0x3E | Present_Voltage | 1 | R | — | — | Current voltage (×0.1 V) |
| 0x3F | Present_Temperature | 1 | R | — | — | Current temperature (°C) |
| 0x40 | Status | 1 | R | — | — | Status flags |
| 0x41 | Moving_Status | 1 | R | — | — | 1=moving, 0=stopped |
| 0x42 | Present_Current_L | 2 | R | — | — | Current draw (mA) |

### Baud Rate Table

| Code | Baud Rate |
|------|-----------|
| 0 | 1,000,000 |
| 1 | 500,000 |
| 2 | 250,000 |
| 3 | 128,000 |
| 4 | 115,200 |
| 5 | 76,800 |
| 6 | 57,600 |
| 7 | 38,400 |

### Operating Mode

| Code | Mode |
|------|------|
| 0 | Position control (default) |
| 1 | Speed/velocity control |
| 2 | Step/PWM control |
| 3 | Passive/free mode |

---

## 5. Error Status Bits

The ERROR byte in every status packet is a bitmask:

| Bit | Mask | Description |
|-----|------|-------------|
| 0 | 0x01 | Input voltage error |
| 1 | 0x02 | Sensor error |
| 2 | 0x04 | Overheat error |
| 3 | 0x08 | Overcurrent error |
| 4 | 0x10 | Angle/position limit error |
| 5 | 0x20 | Overload error |

---

## 6. Communication Examples

### Read Present Position (servo ID=1)

```
TX: FF FF 01 04 02 38 02 BE
          ──  ──  ──  ──  ──  ──  ──  ──
          ID  LEN INS ADDR LEN CHK
RX: FF FF 01 04 00 [POS_L] [POS_H] [CHK]
```

Position raw value 0–4095 maps to 0°–360° (resolution: 0.0879°/step).

### Write Goal Position (servo ID=1, target=2048)

```
TX: FF FF 01 05 03 2A 00 08 [CHK]
                    ──  ─────  ──
                    ADDR 2048  ...
RX: FF FF 01 02 00 [CHK]
```

### Enable Torque (servo ID=1)

```
TX: FF FF 01 04 03 28 01 [CHK]
RX: FF FF 01 02 00 [CHK]
```

### Sync Write Goal Position to Multiple Servos (IDs 1–6)

```
TX: FF FF FE [LEN] 83 2A 02
             [01] [POS1_L] [POS1_H]
             [02] [POS2_L] [POS2_H]
             ...
             [06] [POS6_L] [POS6_H]
             [CHK]
```

LEN = 6 × 3 + 4 = 22

---

## 7. Position Encoding

Raw position range: **0 – 4095** (12-bit)

| Raw Value | Angle |
|-----------|-------|
| 0 | 0° |
| 2048 | 180° |
| 4095 | ~359.9° |

Formula:
```
angle_deg = raw_position × (360.0 / 4096)
raw_position = int(angle_deg × (4096 / 360.0))
```

In LeRobot, positions are normalized to **−100 to +100** (RANGE_M100_100) or **0 to 360 degrees** (DEGREES) depending on `use_degrees` config.

---

## 8. Python SDK Usage

The `hiwonder_sdk` library (included in this repository) provides a direct Python interface:

```python
from lerobot.motors.hiwonder.hiwonder_sdk.port_handler import PortHandler
from lerobot.motors.hiwonder.hiwonder_sdk.packet_handler import PacketHandler

port = PortHandler("COM3")  # or "/dev/ttyUSB0"
port.openPort()
port.setBaudRate(1_000_000)

pkt = PacketHandler(port, endianness=0)  # 0 = little-endian

# Read present position of servo ID=1
position, result, error = pkt.read2ByteData(1, 0x38)

# Write goal position 2048 to servo ID=1
data = [pkt.getLowByte(2048), pkt.getHighByte(2048)]
result, error = pkt.writeReadData(1, 0x2A, 2, data)
```

For batch operations, use `GroupSyncRead` / `GroupSyncWrite`:

```python
from lerobot.motors.hiwonder.hiwonder_sdk.group_sync_read import GroupSyncRead
from lerobot.motors.hiwonder.hiwonder_sdk.group_sync_write import GroupSyncWrite

# Sync read Present_Position from servos 1–6
reader = GroupSyncRead(pkt, 0x38, 2)
for id_ in range(1, 7):
    reader.addParam(id_)
result = reader.txRxPacket()
positions = {id_: reader.getData(id_, 0x38, 2) for id_ in range(1, 7)}

# Sync write Goal_Position
writer = GroupSyncWrite(pkt, 0x2A, 2)
for id_, pos in goal_positions.items():
    writer.addParam(id_, [pkt.getLowByte(pos), pkt.getHighByte(pos)])
writer.txPacket()
```

---

## 9. LeRobot Integration

The HX-30HM is integrated into LeRobot via `HiwonderMotorsBus`, which extends `FeetechMotorsBus` (same register layout). Set `motor_model: "hx30hm"` in your robot config:

```yaml
# config_so101_follower.yaml
_target_: lerobot.robots.so_follower.SOFollowerRobotConfig
port: COM3
motor_model: hx30hm
use_degrees: true
```

The bus handles normalization, calibration, sync read/write, and torque management automatically.

---

## 10. Servo Specifications

| Parameter | Value |
|-----------|-------|
| Operating Voltage | 6.0 – 8.4 V |
| Recommended Voltage | 7.4 V |
| Stall Torque | 30 kg·cm @ 7.4 V |
| No-load Speed | 0.17 sec/60° @ 7.4 V |
| Position Resolution | 4096 steps / 360° (0.0879°/step) |
| Communication | Half-duplex UART, TTL |
| Default Baud Rate | 1,000,000 bps |
| Position Sensor | Magnetic encoder |
| Operating Temperature | −20°C to +70°C |
| Weight | ~82 g |

For product details, see:
- Amazon: https://www.amazon.com/dp/B0DJP8HRMK
- Hiwonder: https://www.hiwonder.com/products/hx-30hm

---

*Protocol reverse-engineered from official Hiwonder SDK and HX-30HM register table. Contact: liangfuyuan581@gmail.com*
