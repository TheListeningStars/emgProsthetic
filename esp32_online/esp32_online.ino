// =============================================================
// esp32_online.ino — dumb sensor + actuator node
//
// The ESP32 does NO signal processing. It just:
//   1) samples 3 EMG channels at EMG_FS_HZ and ships raw ADC values
//      to the computer as "R,t_us,raw0,raw1,raw2\n"
//   2) listens for "A,angle_deg\n" lines from the computer and drives
//      the servo to that angle (clamped to [ANGLE_MIN,ANGLE_MAX]).
//
// All filtering, feature extraction, model training, and prediction
// live on the computer (live_train.py). Until the first angle command
// arrives, the servo holds SERVO_REST_DEG.
// =============================================================
#include <Arduino.h>
#include <ESP32Servo.h>

// ---- USER CONFIG -----------------------------------------------------------
static const int   ANALOG_PIN_GRAV = 32;
static const int   ANALOG_PIN_M1   = 34;
static const int   ANALOG_PIN_M2   = 35;
static const int   SERVO_PIN       = 18;
static const long  SERIAL_BAUD     = 921600;

static const float EMG_FS_HZ       = 63.4635f;   // must match laptop filters
static const float SERVO_REST_DEG  = 135.0f;
static const float ANGLE_MIN_DEG   = 90.0f;
static const float ANGLE_MAX_DEG   = 180.0f;

// ---- DERIVED ---------------------------------------------------------------
static const int EMG_PERIOD_US = (int)(1.0e6f / EMG_FS_HZ);

// ---- SERVO -----------------------------------------------------------------
static Servo servo;

// ---- INCOMING SERIAL "A,<angle>\n" ----------------------------------------
static char inbuf[128];
static int  inlen = 0;

static void process_incoming_line(char* line) {
  if (line[0] != 'A' || line[1] != ',') return;
  float a = strtof(line + 2, nullptr);
  if (a < ANGLE_MIN_DEG) a = ANGLE_MIN_DEG;
  if (a > ANGLE_MAX_DEG) a = ANGLE_MAX_DEG;
  servo.write((int)a);
}

static void serial_pump() {
  while (Serial.available()) {
    int c = Serial.read();
    if (c < 0) break;
    if (c == '\n') {
      inbuf[inlen] = 0;
      process_incoming_line(inbuf);
      inlen = 0;
    } else if (inlen < (int)sizeof(inbuf) - 1) {
      inbuf[inlen++] = (char)c;
    } else {
      inlen = 0;  // overrun -> drop line
    }
  }
}

// ---- MAIN LOOP -------------------------------------------------------------
static unsigned long last_emg_us = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  servo.attach(SERVO_PIN);
  servo.write((int)SERVO_REST_DEG);
  Serial.printf("S,boot fs=%.2f period_us=%d rest=%.1f\n",
                EMG_FS_HZ, EMG_PERIOD_US, SERVO_REST_DEG);
  last_emg_us = micros();
}

void loop() {
  serial_pump();

  unsigned long now = micros();
  if ((long)(now - last_emg_us) < EMG_PERIOD_US) return;
  last_emg_us += EMG_PERIOD_US;

  int r0 = analogRead(ANALOG_PIN_GRAV);
  int r1 = analogRead(ANALOG_PIN_M1);
  int r2 = analogRead(ANALOG_PIN_M2);

  Serial.printf("R,%lu,%d,%d,%d\n", now, r0, r1, r2);
}
