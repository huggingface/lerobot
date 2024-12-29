import serial


class HomonculusGlove:
    def __init__(self):
        self.serial_port = "/dev/tty.usbmodem1101"
        self.baud_rate = 115200
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)

    def read(self):
        while True:
            if self.serial.in_waiting > 0:
                vals = self.serial.readline().decode("utf-8").strip()
                vals = vals.split(" ")
                vals = [int(val) for val in vals]

                d = {
                    "thumb_0": vals[0],
                    "thumb_1": vals[1],
                    "thumb_2": vals[2],
                    "thumb_3": vals[3],
                    "index_0": vals[4],
                    "index_1": vals[5],
                    "index_2": vals[6],
                    "middle_0": vals[7],
                    "middle_1": vals[8],
                    "middle_2": vals[9],
                    "ring_0": vals[10],
                    "ring_1": vals[11],
                    "ring_2": vals[12],
                    "pinky_0": vals[13],
                    "pinky_1": vals[14],
                    "pinky_2": vals[15],
                }
                return d

        # if ser.in_waiting > 0:
        #     line = ser.readline().decode('utf-8').strip()
        #     print(line)


if __name__ == "__main__":
    glove = HomonculusGlove()
    d = glove.read()
    lol = 1
