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

#include <WiFiClient.h>
#include <WiFiManager.h>
#include <ArduinoJson.h>
#include <arduino-timer.h>

const char* WIFI_NAME = "H369A09B46E";
const char* WIFI_PWD = "2FF3F4323667";

WiFiManager wm;
AsyncUDP udp;

WiFiClient client;
IPAddress host(192, 168, 2, 9);
unsigned int port = 8080;

char SERVER_URL[80];
bool connected = false;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

String mac;
String station;

const int DATA_LENGTH = 128;
const int SAMPLE_RATE = 512;
const int json_values_capacity = JSON_OBJECT_SIZE(6) + 4 * JSON_OBJECT_SIZE(DATA_LENGTH);
const int json_capacity = json_values_capacity  + JSON_OBJECT_SIZE(1);
StaticJsonDocument<json_capacity> doc;

int data_ptr = 0;
JsonArray data0;
JsonArray data1;
JsonArray data2;
JsonArray data3;

// sensor reads
int read0 = 0;
int read1 = 0;
int read2 = 0;
int read3 = 0;

// Active status and timers
bool active = true;
int avg_active = 0;

// Temperature "on" time
const long TEMP_ON = 5000;
bool ignoreTemp = false;

// timer to simplify scheduling events
Timer<2> timer;

auto blink_timer = timer_create_default();


bool ping(void *){
  udp.printf("{\"server\":{\"type\": \"ping\", \"mac\":\"%s\"}}", mac.c_str());
  Serial.printf("{\"server\":{\"type\": \"ping\", \"mac\":\"%s\"}}\n", mac.c_str());
  return true;
}

bool readPins(void *){
  read0 = analogRead(SENSOR0);
  read1 = analogRead(SENSOR1);
  read2 = analogRead(SENSOR2);
  read3 = analogRead(SENSOR3);

  data0[data_ptr] = read0;
  data1[data_ptr] = read1;
  data2[data_ptr] = read2;
  data3[data_ptr] = read3;

  data_ptr++;
  avg_active += read3;

  if(data_ptr >= DATA_LENGTH) {
    active = (float) avg_active / DATA_LENGTH > 8.0;

    char output[2048];

    serializeJson(doc, output);
    
    data_ptr = 0;
    avg_active = 0;

    if(!connected){
      return true;
    }

    HTTPClient http;
    http.begin(client, SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    
    int httpResponseCode = http.POST(output);
    if(httpResponseCode < 200 || httpResponseCode >= 300){
      connected = false;
      Serial.println("disconnect");
    }

    blink_timer.cancel();
    digitalWrite(LED_BUILTIN, HIGH);
    blink_timer.in(100, [](void*) -> bool {digitalWrite(LED_BUILTIN, LOW);return true;} );
    
    http.end();
  }

  return true;
}

bool turnTempsOff(void *){
  digitalWrite(AIN2, LOW);
  digitalWrite(AIN1, LOW);
  digitalWrite(BIN2, LOW);
  digitalWrite(BIN1, LOW);

  ignoreTemp = false;
  return true;
}

void parsePacket(AsyncUDPPacket packet){
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

  if(obj.containsKey("all")){
    if(obj["all"]["type"] == "lighthouse"){
      host = packet.remoteIP();
      port = obj["all"]["port"];
      connect();
    }
  }

  if(obj.containsKey(mac)){
    JsonObject details = obj[mac];
    if(details.containsKey("station")){
      station = details["station"].as<String>();
      Serial.print("updated station to ");
      Serial.println(station);
    }
  }

  if(!obj.containsKey(station)){
    return;
  }

  JsonObject parameters = obj[station];
  // String output;
  // serializeJson(parameters, output);
  // Serial.println("recieved parameters:");
  // Serial.println(output);

  // LED Indicator
  if(parameters.containsKey("highlight")){
    if(parameters["highlight"]){
      digitalWrite(IND1, HIGH);
    } else {
      digitalWrite(IND1, LOW);
    }
  }

  // Temperatures
  bool tempChange = false;
  if(parameters.containsKey("temp1") && !ignoreTemp){
    float temp = parameters["temp1"];

    if(temp > 0.2){
      digitalWrite(AIN2, LOW);
      digitalWrite(AIN1, HIGH);
    } else if(temp < 0.2){
      digitalWrite(AIN1, LOW);
      digitalWrite(AIN2, HIGH);
    } else {
      digitalWrite(AIN1, LOW);
      digitalWrite(AIN2, LOW);
    }

    tempChange = true;
  }

  if(parameters.containsKey("temp2") && !ignoreTemp){
    float temp = parameters["temp2"];

    if(temp > 0.2){
      digitalWrite(BIN2, LOW);
      digitalWrite(BIN1, HIGH);
    } else if(temp < 0.2){
      digitalWrite(BIN1, LOW);
      digitalWrite(BIN2, HIGH);
    } else {
      digitalWrite(BIN1, LOW);
      digitalWrite(BIN2, LOW);
    }

    tempChange = true;
  }

  if(tempChange){
    ignoreTemp = true;
    timer.in(TEMP_ON, turnTempsOff);
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

  Serial.println(WiFi.localIP());

  if(udp.listenMulticast(MCAST_GRP, UDP_PORT)) {
    Serial.print("UDP Listening on IP: ");
    Serial.println(WiFi.localIP());
    udp.onPacket([](AsyncUDPPacket packet) {
        parsePacket(packet);
    });
  }

  mac = getMac();

  Serial.print("MAC: ");
  Serial.println(mac);

  timer.every(10000, ping);
  timer.every(1000/SAMPLE_RATE, readPins);


  JsonObject server = doc.createNestedObject("server");
  server["type"] = "sensor";
  server["active"] = true;

  data0 = server.createNestedArray("data0");
  data1 = server.createNestedArray("data1");
  data2 = server.createNestedArray("data2");
  data3 = server.createNestedArray("data3");

  ping(0);
  connect();
}

void loop() {
  timer.tick();
  blink_timer.tick();
}

String getMac() {
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

void connect(){
  Serial.println("connect");
  sprintf(SERVER_URL, "http://%s:%u/", host.toString().c_str(), port);
  connected = true;
}