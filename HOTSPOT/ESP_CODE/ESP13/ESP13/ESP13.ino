#include "Arduino.h"

#if defined(ESP32)
// ------- ESP32 constants ---------------------
#include <AsyncUDP.h>
#define LED_BUILTIN 2
#else
// ------- ESP8266 constants -------------------
#include <ESPAsyncUDP.h>
#undef LED_BUILTIN
#define LED_BUILTIN 16
#undef HIGH
#define HIGH 0
#undef LOW
#define LOW 1
#endif

#include <WiFiManager.h>
#define ARDUINOJSON_DECODE_UNICODE 0
#include <ArduinoJson.h>
#include <arduino-timer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);


WiFiManager wm;
AsyncUDP udp;

WiFiClient tcpClient;
IPAddress host(192, 168, 2, 9);
int tcp_port = 0;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

char mac[18];

#define MAX_AIR_TIME_S 10
#define PULSE_ON_TIME_MS 500
const int VALVE_PINS[] = {26, 16, 17, 18, 19, 23};
// 0: off and avaliable, 1: off and deflating 2: on
bool valveState[] = {0, 0, 0, 0, 0, 0};
// int valveTimes[] = {0, 0, 0, 0, 0, 0};
float valveProbabilities[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
float probOffset = 0.05;
float observerOffset = 0.0;


Timer<> timer;
Timer<> blink_timer;
Timer<> breath_timer;
Timer<> valve_timers[6];

// struct Times {
//   unsigned long last_blink_time;
//   unsigned long last_breath_time;
//   unsigned long last_valve_time[6];
// };
// Times times;
// bool blinking = false;

#define BLINK_TIME_MS 200
#define VALVE_TIME_MS 500
int breath_time_ms = 5000;


bool checkConnection(void *){
  if (!tcpClient.connected() && tcp_port != 0){
    tcpClient.connect(host, tcp_port);
  }

  return true;
}

bool ping(void *){
  if(tcpClient.connected()){
    tcpClient.printf("{\"server\":{\"type\":\"ping\",\"mac\":\"%s\"}}\n", mac);
    blink();
  }
  return true;
}

void blink(){
  digitalWrite(LED_BUILTIN, HIGH);
  blink_timer.in(200, [](void*) -> bool {digitalWrite(LED_BUILTIN, LOW);return true;} );
}

bool breathe(void* station){
  // times.last_breath_time = millis();
  display.clearDisplay();
  Serial.print("breathe: ");
  float p = random(10000)/10000.0;
  // Serial.printf("%f p %f probOffset", p, probOffset);

  if(p < probOffset + observerOffset){
    int i = random(6);
    turnValveOn(i);
    Serial.println("on");
  } else {
    Serial.println("-");
  }

  breath_timer.in(1000 + random(5000), breathe);
  return false;
}

void turnValveOn(int station){
  Serial.printf("valve %u on\n", station);

  display.clearDisplay();
  display.printf("%u", station);
  display.display();

  digitalWrite(VALVE_PINS[station], HIGH);
  valveState[(int) station] = 2;
  valve_timers[station].cancel();
  valve_timers[station].in(500, turnValveOff, (void*)station);
}

bool turnValveOff(void* station){
  digitalWrite(VALVE_PINS[(int) station], LOW);
  valveState[(int) station] = 1;
  valve_timers[(int) station].in(PULSE_ON_TIME_MS, setValveAvaliable, station);
  return false;
}

bool setValveAvaliable(void* station){
  valveState[(int) station] = 0;
  return false;
}

void parsePacket(AsyncUDPPacket packet){
  auto data = packet.data();
  blink();

  DynamicJsonDocument doc(4096);
  DeserializationError error = deserializeJson(doc, data);

  if(error){
    Serial.println("Error parsing JSON");
    return;
  }

  if(doc.containsKey("all")){
    if(doc["all"]["type"] == "lighthouse"){
      if(!tcpClient.connected()){
        Serial.println("lighthouse");
        host = packet.remoteIP();
        tcp_port = doc["all"]["tcp_port"];
        Serial.printf("%u.%u.%u.%u host, %u tcp_port\n", host[0], host[1], host[2], host[3], tcp_port);
      }
    }
  }

  if(doc.containsKey("camera_result")){
    // Serial.println((float)doc["camera_result"]);
    float movement = doc["camera_result"]["movement"];
    probOffset = map(movement, 0.0, 1.0, 0.10, 0.01, true);
    Serial.printf("movement: %f\n", movement);
    Serial.print("probOffset set to ");
    Serial.println(probOffset);
  }

  if(doc.containsKey("triggered")){
    const char* name = doc["triggered"].as<const char *>();
    int station = atoi(name);

    turnValveOn(station);

    // float prob = valveProbabilities[station] + probOffset + observerOffset;
    // float p = random(10000) / 10000.0;

    // Serial.printf("%u station %u prob %u p", station, prob, p);

    // if(p < prob){
    //   Serial.printf("valve %s activated\n", name);
    //   turnValveOn((void*)station);
    // }
  }

  if(doc.containsKey("ESP13")){
    float observer_count = doc["ESP13"]["observer_count"].as<float>();
    observerOffset = map(observer_count, 0.0, 12.0, 0.0, 0.5, true);
    Serial.printf("observer_count %.1f, observerOffset %.2f\n", observer_count, observerOffset);
  }

  for(int i=0; i<6; i++){
    char name[2];
    snprintf(name, 2, "%u", i);

    if(!doc.containsKey(name)){
      continue;
    }
    if(!doc[name].containsKey("touch_count")){
      continue;
    }

    int observers = doc[name]["touch_count"];
    valveProbabilities[i] = 0.5*observers;

    Serial.printf("probability for %u set to %.3f, touchers: %u\n", i, valveProbabilities[i], observers);
  }

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
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) { // Address 0x3D for 128x64
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
  
  delay(3000);
  blink();

  timer.every(4000, ping);
  timer.every(2000, checkConnection);
  breath_timer.in(5000, breathe);
}

void loop() {
  timer.tick();
  blink_timer.tick();
  breath_timer.tick();
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

float map(float x, float x1, float x2, float y1, float y2, bool clip){
  float p = (x - x1) / (x2 - x1);
  if(clip){
    p = constrain(p, 0.0, 1.0);
  }
  return y1 + p * (y2 - y1);
}

float map(float x, float x1, float x2, float y1, float y2){
  return map(x, x1, x2, y1, y2, false);
}