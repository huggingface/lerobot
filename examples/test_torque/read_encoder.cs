#include <DFRobot_VisualRotaryEncoder.h>

DFRobot_VisualRotaryEncoder_I2C sensor(0x54, &Wire);

void setup()
{
  Serial.begin(115200);

  // Attempt to initialize the sensor
  while (NO_ERR != sensor.begin()) {
    // Failed? Just wait a bit and try again
    delay(3000);
  }
}

void loop()
{
  // Read the encoder value
  uint16_t encoderValue = sensor.getEncoderValue();
  
  // Print it followed by a newline
  Serial.println(encoderValue);

  // Delay 10ms between readings
  delay(10);
}
