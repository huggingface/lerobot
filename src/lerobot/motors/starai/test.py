from fashionstar_uart_sdk import *




uservo = PocketHandler("/dev/ttyUSB0",1000000)

servos = {
    "调査天":1,
    "112":2,
    "113":3,
}

data = uservo.sync_read["Monitor"](servos)

for key, person in data.items():
    print(f"{key}: {person}")  # 利用 __str__ 方法打印
