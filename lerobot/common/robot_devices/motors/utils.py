from typing import Protocol


class MotorsBus(Protocol):

    def connect(self): ...
    def disconnect(self): ...
    @property
    def motor_names(self) -> list[str]: ...
    def set_calibration(self, calibration: dict[str, list[str | int]]): ...
    def apply_calibration(
        self,
        values: NDArray[np.int64] | NDArray[np.float64] | list[int] | list[float],
        motor_names: list[str] | None = None,
    ) -> NDArray[np.int64] | NDArray[np.float64] | list[int] | list[float]: ...
    def revert_calibration(
        self,
        values: NDArray[np.int64] | NDArray[np.float64] | list[int] | list[float],
        motor_names: list[str] | None,
    ) -> NDArray[np.int64] | NDArray[np.float64] | list[int] | list[float]: ...
    def read(self, data_name: str) -> NDArray[np.int64]: ...
    def write(
        self,
        data_name: str,
        values: int | float | NDArray[np.int64] | NDArray[np.float64],
        motor_names: str | list[str] | None = None,
    ): ...
