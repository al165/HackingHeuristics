#include "Arduino.h"

#if defined(ESP32)
// ------- ESP32 constants ---------------------
#include <AsyncUDP.h>
#define LED_BUILTIN 2
#else
// ------- ESP8266 constants -------------------
#include <ESPAsyncUDP.h>
#define LED_BUILTIN 16
#endif

#include <WiFiManager.h>
#include <ArduinoJson.h>
#include <arduino-timer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const char* WIFI_NAME = "H369A09B46E";
const char* WIFI_PWD = "2FF3F4323667";

WiFiManager wm;
AsyncUDP udp;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;
String mac;

const int MAX_AIR_TIME = 10;
const int VALVE_PINS[] = {26, 16, 17, 18, 19, 23};
// 0: off and ready, 1: off and cooling down, 2: on
bool valveState[] = {0, 0, 0, 0, 0, 0};
int valveTimes[] = {0, 0, 0, 0, 0, 0};


#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

Timer<4> timer;
Timer<> blink_timer;

bool ping(void *){
  udp.printf("{\"server\":{\"type\": \"ping\", \"mac\":\"%s\"}}", mac.c_str());
  blink();
  return true;
}

void blink(){
  blink_timer.cancel();
  digitalWrite(LED_BUILTIN, HIGH);
  blink_timer.in(100, [](void*) -> bool {digitalWrite(LED_BUILTIN, LOW);return true;} );
}

bool turnValveOff(void* station){
  digitalWrite(VALVE_PINS[(int)station], LOW);
  valveState[(int)station] = 0;
  return true;
}


void parsePacket(AsyncUDPPacket packet){
  blink();
  String data(reinterpret_cast<char *>(packet.data()));

  Serial.println("--------");
  Serial.println(data);

  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, data);

  if(error){
    Serial.println("Error parsing JSON");
    return;
  }

  JsonObject obj = doc.as<JsonObject>();

  // JsonObject parameters = obj[mac];
  // String output;
  // serializeJson(parameters, output);
  // Serial.println("recieved parameters:");
  // Serial.println(output);

  int station;
  for(station = 0; station < 6; station++){
    String name = (String) station;

    if(!obj.containsKey(name)){
      continue;
    }

    JsonObject parameters = obj[name];

    if(parameters.containsKey("airtime")){
      float airtime = parameters["airtime"];
      valveTimes[station] = (int) (airtime * 1000);
    }

    if(parameters.containsKey("airon")){
      float val = parameters["airon"];
      if(val > MAX_AIR_TIME){
        val = (float) MAX_AIR_TIME;
      }

      display.clearDisplay();
      display.setTextSize(4);
      display.setTextColor(WHITE);
      display.setCursor(40, 20);

      // if(val < 0.5 && valveState[station] == 2){
      //   display.print(station);
      //   display.println(": off");
      // } else 
      
      if(val >= 0.5 && valveState[station] == 0){
        display.print(station);
        display.println(": on");

        valveState[station] = 2;
        timer.in((int) (val * 1000), turnValveOff, (void *)station);
      }

      display.display();
    }
  }

  return;
}


void setup() {
  Serial.begin(115200);
  Serial.println();

  pinMode(LED_BUILTIN, OUTPUT);

  for(int count=0;count<6;count++) {
    pinMode(VALVE_PINS[count], OUTPUT);
    digitalWrite(VALVE_PINS[count], LOW);
  }

#if defined(ESP32)
  WiFi.mode(WIFI_MODE_STA);
#else
  WiFi.mode(WIFI_STA);
#endif

  if(wm.autoConnect(WIFI_NAME, WIFI_PWD)){
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

  mac = getMac();
  
  timer.every(10000, ping);
  ping(0);
}

void loop() {
  timer.tick();
  blink_timer.tick();
}

String getMac(){
  byte baseMac[6];
  WiFi.macAddress(baseMac);
  char baseMacChr[18] = {0};
  sprintf(
    baseMacChr, 
    "%02X:%02X:%02X:%02X:%02X:%02X", 
    baseMac[0], baseMac[1], baseMac[2], baseMac[3], baseMac[4], baseMac[5]
  );
  return String(baseMacChr);
}