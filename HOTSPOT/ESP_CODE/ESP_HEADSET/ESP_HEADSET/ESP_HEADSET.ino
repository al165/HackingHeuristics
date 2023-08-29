/*
  HEADSET ESP CODE

  - collects sensor data
  - receives values for the temp accuators and the LED
  - updates attached screen (TODO)
*/

#include "Arduino.h"

#if defined(ESP32)
// ------- ESP32 constants ---------------------
#include <HTTPClient.h>
#include <AsyncUDP.h>
#define LED_BUILTIN 2
const int SENSOR0 = 34;  // EEG
const int SENSOR1 = 32;  // EOG
const int SENSOR2 = 39;  // heart rate sensor
const int SENSOR3 = 36;  // GSR
const int MAX_ANALOG_INPUT = 4095;
#define AIN1 13
#define AIN2 12
#define BIN1 26
#define BIN2 27
#else
// ------- ESP8266 constants -------------------
#include <ESP8266HTTPClient.h>
#include <ESPAsyncUDP.h>
// #include <ESP8266WiFi.h>
#define LED_BUILTIN 16
const int SENSOR0 = A0;  // EEG
const int SENSOR1 = A0;  // EOG
const int SENSOR2 = A0;  // heart rate sensor
const int SENSOR3 = A0;  // GSR
const int MAX_ANALOG_INPUT = 1023;
// Temperature Pins :
#define AIN1 4
#define AIN2 5
#define BIN1 12
#define BIN2 13
// ----------------------------------------------
#endif

// proximity indicator pin(s)
#define IND1 14

// HEAT_DIRECTION indicates which way the pins have to be powered to make the
// temp things warm - set to either 0 or 1.
#define HEAT_DIRECTION 0

#include <WiFiClient.h>
#include <WiFiManager.h>
#define ARDUINOJSON_DECODE_UNICODE 0
#include <ArduinoJson.h>
#include <arduino-timer.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define SCREEN_WIDTH 128 
#define SCREEN_HEIGHT 64 
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
char screenMessage[32];

WiFiManager wm;
AsyncUDP udp;

WiFiClient client;
IPAddress host(192, 168, 2, 9);
int port = 8080;

char SERVER_URL[80];
bool connected = false;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

char mac[18];
char station[2] = {0, 0};

#define DATA_LENGTH 128
#define SAMPLE_RATE 512

const int json_capacity = JSON_OBJECT_SIZE(6) + 4 * JSON_OBJECT_SIZE(DATA_LENGTH)  + JSON_OBJECT_SIZE(1);

int reads[4] = {0, 0, 0, 0};

struct SensorData {
  bool active;
  int data0[DATA_LENGTH];
  int data1[DATA_LENGTH];
  int data2[DATA_LENGTH];
  int data3[DATA_LENGTH];
};

SensorData sensorData;

// Temperature "on" time
const long TEMP_ON = 5000;

// timer to simplify scheduling events
Timer<6> timer;
Timer<> blink_timer;


bool ping(void *){
  udp.printf("{\"server\":{\"type\": \"ping\", \"mac\":\"%s\"}}", mac);
  blink();
  // Serial.printf("{\"server\":{\"type\": \"ping\", \"mac\":\"%s\"}}\n", mac);
  return true;
}

void blink(){
  blink_timer.cancel();
  digitalWrite(LED_BUILTIN, HIGH);
  blink_timer.in(100, [](void*) -> bool {digitalWrite(LED_BUILTIN, LOW);return true;} );
}

bool pulse(void *){
  static int fade_amt = -1;
  static int brightness = 128;

  ledcWrite(LED_BUILTIN, brightness);

  if(brightness <= 0 || brightness >= 255){
    fade_amt = -fade_amt;
  }

  brightness += fade_amt;
  return true;
}


bool readPins(void *){
  static int data_ptr = 0;
  static int avg_active = 0;
  static bool active = false;
  static bool sleep_post = false;

  reads[0] = analogRead(SENSOR0);
  reads[1] = analogRead(SENSOR1);
  reads[2] = analogRead(SENSOR2);
  reads[3] = analogRead(SENSOR3);

  sensorData.data0[data_ptr] = reads[0];
  sensorData.data1[data_ptr] = reads[1];
  sensorData.data2[data_ptr] = reads[2];
  sensorData.data3[data_ptr] = reads[3];

  data_ptr++;
  avg_active += reads[3];

  if(data_ptr >= DATA_LENGTH) {
    active = (float) avg_active / DATA_LENGTH > 8.0;

    data_ptr = 0;
    avg_active = 0;

    if(!connected || sleep_post){
      sleep_post = !active;
      return true;
    }

    DynamicJsonDocument doc(json_capacity);
    JsonObject server = doc.createNestedObject("server");
    JsonArray data0 = server.createNestedArray("data0");
    JsonArray data1 = server.createNestedArray("data1");
    JsonArray data2 = server.createNestedArray("data2");
    JsonArray data3 = server.createNestedArray("data3");
    for(int i=0; i< DATA_LENGTH; i++){
      data0.add(sensorData.data0[i]);
      data1.add(sensorData.data1[i]);
      data2.add(sensorData.data2[i]);
      data3.add(sensorData.data3[i]);
    }
    server["type"] = "sensor";
    server["active"] = active;
    size_t len = measureJson(doc);
    if(doc.overflowed()){
      Serial.println("doc overflowed!");
    }
    char output[3072];
    serializeJson(doc, output);

    HTTPClient http;
    http.begin(client, SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    
    int httpResponseCode = http.POST(output);
    if(httpResponseCode < 200 || httpResponseCode >= 300){
      connected = false;
    }
    http.end();

    blink();

    sleep_post = !active;
  }

  return true;
}

bool turnTempsOff(void *){
  digitalWrite(AIN2, LOW);
  digitalWrite(AIN1, LOW);
  digitalWrite(BIN2, LOW);
  digitalWrite(BIN1, LOW);

  return true;
}

void parsePacket(AsyncUDPPacket packet){
  auto data = packet.data();
  blink();

  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, data);

  if(error){
    Serial.println("Error parsing JSON");
    return;
  }

  if(doc.containsKey("all")){
    if(doc["all"]["type"] == "lighthouse"){
      host = packet.remoteIP();
      port = doc["all"]["port"];
      connect();
    }
  }

  if(doc.containsKey(mac)){
    JsonObject details = doc[mac];
    if(details.containsKey("station")){
      itoa(details["station"], station, 10);
    }
  }

  if(!doc.containsKey(station)){
    return;
  }

  JsonObject parameters = doc[station];

  // LED Indicator
  if(parameters.containsKey("highlight")){
    if(parameters["highlight"]){
      digitalWrite(IND1, HIGH);
    } else {
      digitalWrite(IND1, LOW);
    }
  }

  if(parameters.containsKey("touch_count")){
    int val = parameters["touch_count"].as<int>();
    if(val > 0){
      Serial.println("turning on temps");

      if(HEAT_DIRECTION){
        digitalWrite(AIN2, LOW);
        digitalWrite(AIN1, HIGH);
        digitalWrite(BIN2, LOW);
        digitalWrite(BIN1, HIGH);
      } else {
        digitalWrite(AIN2, HIGH);
        digitalWrite(AIN1, LOW);
        digitalWrite(BIN2, HIGH);
        digitalWrite(BIN1, LOW);
      }
      timer.in(TEMP_ON, turnTempsOff);
    } else {
      turnTempsOff(0);
    }
  }

  return;
}

void setup() {
  Serial.begin(115200);
  Serial.println();

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);

  pinMode(IND1, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  display.clearDisplay();

  ledcSetup(LED_BUILTIN, 5000, 8);
  ledcAttachPin(LED_BUILTIN, 0);

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

  Serial.println(WiFi.localIP());

  if(udp.listenMulticast(MCAST_GRP, UDP_PORT)) {
    Serial.print("UDP Listening on IP: ");
    Serial.println(WiFi.localIP());
    udp.onPacket([](AsyncUDPPacket packet) {
        parsePacket(packet);
    });
  }

  getMac();

  Serial.println("**********");
  Serial.print(" MAC: ");
  Serial.println(mac);
  Serial.println("**********");

  // snprintf(screenMessage, sizeof(screenMessage), "* MAC %s *", mac);
  // screenMessage = "* MAC " + mac + " *";
  // timer.in(4000, [](void*) -> bool {screenMessage[0] = 0; return true;});

  timer.every(10000, ping);
  timer.every(1000/SAMPLE_RATE, readPins);
  timer.every(100, updateScreen);

  ping(0);
  connect();

  Serial.print("json_capacity ");
  Serial.println(json_capacity);
}

void loop() {
  // Serial.print("loop");
  timer.tick();
  blink_timer.tick();
}

bool updateScreen(void*) {
  display.clearDisplay();
  
  // int maxRadius = min(display.width(), display.height());
 
  // int r0 = map(reads[0], 0, MAX_ANALOG_INPUT, 0, maxRadius);
  // int r1 = map(reads[1], 0, MAX_ANALOG_INPUT, 0, maxRadius);
  // int r2 = map(reads[2], 0, MAX_ANALOG_INPUT, 0, maxRadius);
  // int r3 = map(reads[3], 0, MAX_ANALOG_INPUT, 0, maxRadius);
 
  // int x0 = 10;
  // int y0 = display.height() / 2;

  // int x1 = 40;
  // int x2 = 70;
  // int x3 = 100;

  // display.fillCircle(x0, y0,  r0, SSD1306_WHITE);
  // display.fillCircle(x1, y0,  r1, SSD1306_WHITE);
  // display.fillCircle(x2, y0,  r2, SSD1306_WHITE);
  // display.fillCircle(x3, y0,  r3, SSD1306_WHITE);

  // show message
  display.setTextSize(1);
  display.setTextColor(WHITE, BLACK);
  for(int i = 0; i < 4; i++){
    display.setCursor(64, 10+i*10);
    display.println(reads[i]);
  }
  display.println(screenMessage);

  display.display(); 

  return true;
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

void connect(){
  snprintf(
    SERVER_URL,
    sizeof(SERVER_URL),
    "http://%u.%u.%u.%u:%u/", 
    host[0], 
    host[1], 
    host[2], 
    host[3], 
    port
  );
  connected = true;
  udp.print("{\"server\":{\"type\": \"whoami\"}}");
}
