from lerobot.common.robots.hope_jr import HopeJrHand, HopeJrHandConfig

cfg = HopeJrHandConfig("/dev/tty.usbmodem58760433641", id="left", side="left")
hand = HopeJrHand(cfg)

hand.connect()
hand.calibrate()
hand.disconnect()
