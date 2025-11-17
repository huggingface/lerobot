from gpiozero import MCP3008
from dataclasses import dataclass
from typing import Optional

# ADC Configuration
ADC_CHANNEL = 0
VREF = 3.3
VOLTAGE_DIVIDER_RATIO = 0.2  # R2/(R1+R2) = 750/(3000+750) = 0.2

# Battery voltage range (adjust these based on your battery pack)
BATTERY_VOLTAGE_MIN = 11.0  # Empty battery voltage
BATTERY_VOLTAGE_MAX = 14.6  # Full battery voltage

# Sampling configuration
AVERAGE_SAMPLES = 10  # Number of samples to average for stability

@dataclass
class BatteryData:
    """Battery data container"""
    voltage: float
    percent: int


# Global ADC instance (initialized on first use)
_adc: Optional[MCP3008] = None


def _get_adc() -> MCP3008:
    """Get or initialize the ADC instance"""
    global _adc
    if _adc is None:
        _adc = MCP3008(channel=ADC_CHANNEL)
    return _adc


def get_battery_voltage() -> float:
    """
    Read battery voltage from ADC with averaging.

    Returns:
        Battery voltage in volts
    """
    adc = _get_adc()
    total = 0.0

    for _ in range(AVERAGE_SAMPLES):
        raw = adc.raw_value
        adc_voltage = raw / 1023.0 * VREF
        battery_voltage = adc_voltage / VOLTAGE_DIVIDER_RATIO
        total += battery_voltage

    return total / AVERAGE_SAMPLES


def get_battery_percent() -> int:
    """
    Get battery percentage based on voltage.

    Returns:
        Battery percentage (0-100)
    """
    voltage = get_battery_voltage()

    # Clamp voltage to valid range
    if voltage <= BATTERY_VOLTAGE_MIN:
        return 0
    if voltage >= BATTERY_VOLTAGE_MAX:
        return 100

    # Linear interpolation between min and max voltage
    percent = int(((voltage - BATTERY_VOLTAGE_MIN) /
                   (BATTERY_VOLTAGE_MAX - BATTERY_VOLTAGE_MIN)) * 100)

    # Clamp to valid range
    return max(0, min(100, percent))


def get_battery_data() -> BatteryData:
    """
    Get both battery voltage and percentage.

    Returns:
        BatteryData containing voltage and percent
    """
    voltage = get_battery_voltage()
    percent = get_battery_percent()
    return BatteryData(voltage=voltage, percent=percent)


if __name__ == "__main__":
    # When run as a script, output JSON (for Rust integration)
    import json
    try:
        battery_data = get_battery_data()
        result = {
            "voltage": battery_data.voltage,
            "percent": battery_data.percent
        }
        print(json.dumps(result))
    except Exception as e:
        # Output error JSON so Rust knows battery reading failed
        print(json.dumps({"voltage": -1.0, "percent": -1}))
