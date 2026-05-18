/*
  ESP32 USB serial bridge for an L298N motor driver.

  Serial protocol from the Raspberry Pi:
    M <motor1> <motor2>

  Values are floats in [-1.0, 1.0].
  Positive drives forward, negative drives backward, zero stops.
*/

const int M1_IN1 = 25;
const int M1_IN2 = 26;
const int M2_IN1 = 27;
const int M2_IN2 = 4;

const int PWM_FREQ_HZ = 1000;
const int PWM_RESOLUTION_BITS = 8;
const int PWM_MAX = (1 << PWM_RESOLUTION_BITS) - 1;
const float DEAD_ZONE = 0.03;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);

  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);

  ledcAttach(M1_IN1, PWM_FREQ_HZ, PWM_RESOLUTION_BITS);
  ledcAttach(M1_IN2, PWM_FREQ_HZ, PWM_RESOLUTION_BITS);
  ledcAttach(M2_IN1, PWM_FREQ_HZ, PWM_RESOLUTION_BITS);
  ledcAttach(M2_IN2, PWM_FREQ_HZ, PWM_RESOLUTION_BITS);

  stopAll();
}

void driveMotor(int forwardPin, int backwardPin, float speed) {
  speed = constrain(speed, -1.0, 1.0);
  if (abs(speed) < DEAD_ZONE) {
    ledcWrite(forwardPin, 0);
    ledcWrite(backwardPin, 0);
    return;
  }

  int duty = int(abs(speed) * PWM_MAX);
  if (speed > 0) {
    ledcWrite(backwardPin, 0);
    ledcWrite(forwardPin, duty);
  } else {
    ledcWrite(forwardPin, 0);
    ledcWrite(backwardPin, duty);
  }
}

void stopAll() {
  ledcWrite(M1_IN1, 0);
  ledcWrite(M1_IN2, 0);
  ledcWrite(M2_IN1, 0);
  ledcWrite(M2_IN2, 0);
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) {
    return;
  }

  if (line == "STOP") {
    stopAll();
    Serial.println("OK");
    return;
  }

  float m1 = 0.0;
  float m2 = 0.0;
  if (sscanf(line.c_str(), "M %f %f", &m1, &m2) == 2) {
    driveMotor(M1_IN1, M1_IN2, -m1);
    driveMotor(M2_IN1, M2_IN2, -m2);
    Serial.println("OK");
  } else {
    Serial.println("ERR");
  }
}
