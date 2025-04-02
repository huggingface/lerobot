#include <Arduino.h>

// Define multiplexer input pins
#define S0 5
#define S1 6
#define S2 7
#define S3 8
#define SENSOR_INPUT 4

#define SENSOR_COUNT 16

int rawVals[SENSOR_COUNT];

void measureRawValues() {
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    digitalWrite(S0, i & 0b1);
    digitalWrite(S1, i >> 1 & 0b1);
    digitalWrite(S2, i >> 2 & 0b1);
    digitalWrite(S3, i >> 3 & 0b1);
    delay(1);
    
    rawVals[i] = analogRead(SENSOR_INPUT);
  }
}

void printRawValues() {
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    Serial.print(rawVals[i]);
    if (i < SENSOR_COUNT - 1) Serial.print(" ");
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);;  // ~0.0–1.5V range

  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);

  digitalWrite(S0, LOW);
  digitalWrite(S1, LOW);
  digitalWrite(S2, LOW);
  digitalWrite(S3, LOW);
}

void loop() {
  measureRawValues();
  printRawValues();
  delay(1);
}
