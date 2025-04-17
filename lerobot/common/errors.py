class DeviceNotConnectedError(ConnectionError):
    """Exception raised when the device is not connected."""

    def __init__(self, message="This device is not connected. Try calling `connect()` first."):
        self.message = message
        super().__init__(self.message)


class DeviceAlreadyConnectedError(ConnectionError):
    """Exception raised when the device is already connected."""

    def __init__(
        self,
        message="This device is already connected. Try not calling `connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


class InvalidActionError(ValueError):
    """Exception raised when an action is already invalid."""

    def __init__(
        self,
        message="The action is invalid. Check the value follows what it is expected from the action space.",
    ):
        self.message = message
        super().__init__(self.message)
