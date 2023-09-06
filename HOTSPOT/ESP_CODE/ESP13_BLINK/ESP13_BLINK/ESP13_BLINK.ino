#include "Arduino.h"
#include <WiFiManager.h>

#define LED_BUILTIN 2

WiFiManager wm;

#define MIN_AIR_TIME 500
#define MAX_AIR_TIME 3000

#define MIN_WAIT_TIME 3000
#define MAX_WAIT_TIME 15000

const int VALVE_PINS[] = {26, 16, 17, 18, 19, 23};


void blink(){
  digitalWrite(LED_BUILTIN, HIGH);
  delay(200);
  digitalWrite(LED_BUILTIN, LOW);
}


void setup() {
  Serial.begin(115200);
  Serial.println();

  randomSeed(analogRead(0));

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  for(int count=0;count<6;count++) {
    pinMode(VALVE_PINS[count], OUTPUT);
    digitalWrite(VALVE_PINS[count], LOW);
  }

  WiFi.mode(WIFI_MODE_STA);

  if(wm.autoConnect()){
    Serial.println("connected!");
  } else {
    Serial.println("config portal running...");
  }
  
  blink();
}

void loop() {
  int wait = random(MAX_WAIT_TIME);
  int valve = random(6);
  int onTime = random(MAX_AIR_TIME);

  delay(MIN_WAIT_TIME);
  delay(wait);
  digitalWrite(VALVE_PINS[valve], HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.printf("valve %u on\n", valve);

  delay(MIN_AIR_TIME);
  delay(onTime);
  digitalWrite(VALVE_PINS[valve], LOW);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.printf("valve %u off\n", valve);
}
