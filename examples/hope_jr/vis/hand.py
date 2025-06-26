from lerobot.common.robots.hope_jr import HopeJrHand, HopeJrHandConfig

cfg = HopeJrHandConfig("/dev/cu.usbmodem58760432281", id="left", side="left")
hand = HopeJrHand(cfg)

hand.connect()
hand.calibrate()
hand.disconnect()
