#include "Arduino.h"

#if defined(ESP32)
// ------- ESP32 constants ---------------------
#include <AsyncUDP.h>
#define LED_BUILTIN 2
#else
// ------- ESP8266 constants -------------------
#include <ESPAsyncUDP.h>
#define LED_BUILTIN 16
#define LOW 1
#define HIGH 0
#endif

#include <WiFiManager.h>
#define ARDUINOJSON_DECODE_UNICODE 0
#include <ArduinoJson.h>
#include <arduino-timer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

WiFiManager wm;
AsyncUDP udp;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

char mac[18];

#define MAX_AIR_TIME 10
const int VALVE_PINS[] = {26, 16, 17, 18, 19, 23};
// 0: off and ready, 1: off and cooling down, 2: on
bool valveState[] = {0, 0, 0, 0, 0, 0};
int valveTimes[] = {0, 0, 0, 0, 0, 0};


#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

Timer<12> timer;
Timer<> blink_timer;

bool ping(void *){
  blink();
  udp.printf("{\"server\":{\"type\": \"ping\", \"mac\":\"%s\"}}", mac);
  return true;
}

void blink(){
  blink_timer.cancel();
  digitalWrite(LED_BUILTIN, HIGH);
  blink_timer.in(200, [](void*) -> bool {digitalWrite(LED_BUILTIN, LOW);return true;} );
}

bool turnValveOff(void* station){
  digitalWrite(VALVE_PINS[(int)station], LOW);
  valveState[(int)station] = 1;
  // timer.in((int) valveTimes[(int) station], setValveAvaliable, station);
  // setValveAvaliable(station);
  return true;
}

bool setValveAvaliable(void* station){
  valveState[(int)station] = 0;
  return true;
}

void parsePacket(AsyncUDPPacket packet){
  auto data = packet.data();
  blink();

  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, data);

  if(error){
    Serial.println("Error parsing JSON");
    return;
  }

  if(doc.containsKey("rd_samples")){
    JsonObject parameters = doc["rd_samples"];
    delay(100);
    for(int station = 0; station < 6; station++){
      char name[2] = {0};
      itoa(station, name, 10);

      if(!parameters.containsKey(name)){
        continue;
      }

      float val = parameters[name];

      if(val > 0.5 && valveState[station] == 0){
        Serial.printf("valve %u activated\n", station);
        digitalWrite(VALVE_PINS[station], HIGH);
        valveState[station] = 2;
        timer.in(500, turnValveOff, (void *)station);
      } else if(val < 0.5 && valveState[station] == 1){
        setValveAvaliable((void*) station);
      }

    }

  }

}


void setup() {
  Serial.begin(115200);
  Serial.println();

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  for(int count=0;count<6;count++) {
    pinMode(VALVE_PINS[count], OUTPUT);
    digitalWrite(VALVE_PINS[count], LOW);
  }

#if defined(ESP32)
  WiFi.mode(WIFI_MODE_STA);
#else
  WiFi.mode(WIFI_STA);
#endif

  if(wm.autoConnect()){
    Serial.println("connected!");
  } else {
    Serial.println("config portal running...");
  }

  if(udp.listenMulticast(MCAST_GRP, UDP_PORT)) {
    Serial.print("UDP Listening on IP: ");
    Serial.println(WiFi.localIP());
    udp.onPacket([](AsyncUDPPacket packet) {
        parsePacket(packet);
    });
  }

  // init screen...
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }

  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(20, 20);
  display.println(" ..START UP..");
  display.display(); 

  getMac();
  Serial.print("** MAC: ");
  Serial.println(mac);
  
  timer.every(10000, ping);
  ping(0);
}

void loop() {
  timer.tick();
  blink_timer.tick();
}

void getMac() {
  byte baseMac[6];
  WiFi.macAddress(baseMac);
  snprintf(
    mac, 
    sizeof(mac),
    "%02X:%02X:%02X:%02X:%02X:%02X", 
    baseMac[0], baseMac[1], baseMac[2], baseMac[3], baseMac[4], baseMac[5]
  );
}