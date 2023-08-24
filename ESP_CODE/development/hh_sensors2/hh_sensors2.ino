#if defined(ESP32)
// ------- ESP32 constants ---------------------
#include <WiFi.h>
#include <WebServer.h>
WebServer server(80);
const int SENSOR0 = 34;  // EEG
const int SENSOR1 = 32;  // EOG
const int SENSOR2 = 39;  // heart rate sensor
const int SENSOR3 = 36;  // GSR
const int MAX_ANALOG_INPUT = 4095;
// Temperature Pins:
#define AIN1 13
#define AIN2 12
#define BIN1 26
#define BIN2 27
#else
// ------- ESP8266 constants -------------------
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
ESP8266WebServer server(80);
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

#include <ArduinoJson.h>
#include <WiFiClient.h>
#include <WiFiManager.h>
#include <WiFiUdp.h>

#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define SCREEN_WIDTH 128 
#define SCREEN_HEIGHT 64 
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// proximity indicator pin(s)
#define IND1 14

WiFiManager wm;
// IP Address of computer running HH_server.py:
IPAddress host(192, 168, 2, 80);
const unsigned int PORT = 8080;
const unsigned int PORT_RANGE = 10;
unsigned int currentPort = PORT;

WiFiClient client;

// Stores the sensor readings
String data0 = "";
String data1 = "";
String data2 = "";
String data3 = "";

const int DATA_LENGTH = 64;
const int SAMPLE_RATE = 512;

WiFiUDP UDP;
char packet[255];
const unsigned int UDP_PORT = 4210;

// Unique MAC address as ID
String mac;

// Active status and timers
bool active = true;
long lastActivePing = 0;
const long ACTIVE_PING = 1000000;

// Temperature update time
const long TEMP_UPDATE = 10000000;
// Temperature "on" time
const long TEMP_ON = 5000000;

long lastTempTime = 0;
long tempOffTime = 0;
bool tempOn = false;

// Screen refresh rate
long lastDisplayTime = 0;
const long DISPLAY_REFRESH = 200000;

// sensor reads
int read0 = 0;
int read1 = 0;
int read2 = 0;
int read3 = 0;


JsonObject getPostBody() {
  /*
   * Reads and parses latest POST Json body
   */
   
  String postBody = server.arg("plain");
  Serial.println("--------\nPostBody:");
  Serial.println(postBody);

  DynamicJsonDocument doc(512);
  DeserializationError error = deserializeJson(doc, postBody);

  if (error) {
    Serial.print(F("Error parsing JSON "));
    Serial.println(error.c_str());

    doc["ok"] = false;
  } else {
    doc["ok"] = true;
  }

  JsonObject postObj = doc.as<JsonObject>();
  Serial.println(postObj);
  return postObj;
}

void updateOutput() {
  /*
   * Updates the host's outputs (Highlight, and 2 temp stimulators)
   */

  String postBody = server.arg("plain");
//  Serial.println("--------");
//  Serial.println(postBody);

  DynamicJsonDocument doc(512);
  DeserializationError error = deserializeJson(doc, postBody);

  if (error) {
    Serial.print(F("Error parsing JSON "));
    Serial.println(error.c_str());

    doc["ok"] = false;
  } else {
    doc["ok"] = true;
  }

  JsonObject postObj = doc.as<JsonObject>();

  if(!postObj["ok"]){
    server.send(400, F("text/html"), "Error in parsing JSON body\n");
    return;
  }

  bool success = false;

  // Update highlight status:
  if(postObj.containsKey("highlight")){
    success = true;
    if(postObj["highlight"]){
//      Serial.println("highlight TRUE");
      digitalWrite(IND1, HIGH);
    } else {
//      Serial.println("highlight FALSE");
      digitalWrite(IND1, LOW);
    }
  }


  // check if should ignore temps...
  bool ignoreTemp = false;
  if (micros() - lastTempTime < TEMP_UPDATE) {
    ignoreTemp = true;
  }

  if (postObj.containsKey("temp1") && !ignoreTemp) {
    success = true;
    float temp = postObj["temp1"];

    if (temp > 0.2) {
      // make warm...
      digitalWrite(AIN2, LOW);
      digitalWrite(AIN1, HIGH);

    } else if (temp < -0.2) {
      // make cold...
      digitalWrite(AIN1, LOW);
      digitalWrite(AIN2, HIGH);

    } else {
      // turn off...
      digitalWrite(AIN1, LOW);
      digitalWrite(AIN2, LOW);
    }
    lastTempTime = micros();
    tempOffTime = lastTempTime + TEMP_ON;
    tempOn = true;
  }

  if (postObj.containsKey("temp2") && !ignoreTemp) {
    success = true;
    float temp = postObj["temp2"];

    if (temp > 0.2) {
      // make warm...
      digitalWrite(BIN2, LOW);
      digitalWrite(BIN1, HIGH);

    } else if (temp < -0.2) {
      // make cold...
      digitalWrite(BIN2, LOW);
      digitalWrite(BIN1, HIGH);

    } else {
      // turn off...
      digitalWrite(BIN2, LOW);
      digitalWrite(BIN1, LOW);
    }
    lastTempTime = micros();
    tempOffTime = lastTempTime + TEMP_ON;
    tempOn = true;
  }

  if (success) {
    server.send(201);
  } else {
    server.send(400, F("text/html"), "neither temp1, temp2 or hightlight specified\n");
  }
}

void turnTempsOff() {
  /*
   * Turn all temperature stimulators off.
   */
  digitalWrite(AIN2, LOW);
  digitalWrite(AIN1, LOW);
  digitalWrite(BIN2, LOW);
  digitalWrite(BIN1, LOW);

  tempOn = false;
}

void sendPing(){
  String json = "{\"mac\":\"" + mac + "\",";
  
  if(active){
    json += "\"active\":true}#";
  } else {
    json += "\"active\":false}#";
  }

//  Serial.println(json);
  client.print(json);
}

void readPins(){
  read0 = analogRead(SENSOR0);
  read1 = analogRead(SENSOR1);
  read2 = analogRead(SENSOR2);
  read3 = analogRead(SENSOR3);
}

void updateScreen() {
  display.clearDisplay();
  
  int maxRadius = min(display.width(), display.height());
 
  int r0 = map(read0, 0, MAX_ANALOG_INPUT, 0, maxRadius);
  int r1 = map(read1, 0, MAX_ANALOG_INPUT, 0, maxRadius);
  int r2 = map(read2, 0, MAX_ANALOG_INPUT, 0, maxRadius);
  int r3 = map(read3, 0, MAX_ANALOG_INPUT, 0, maxRadius);
 
  int x0 = 10;
  int y0 = display.height() / 2;

  int x1 = 40;
  int x2 = 70;
  int x3 = 100;

  display.fillCircle(x0, y0,  r0, SSD1306_WHITE);
  display.fillCircle(x1, y0,  r1, SSD1306_WHITE);
  display.fillCircle(x2, y0,  r2, SSD1306_WHITE);
  display.fillCircle(x3, y0,  r3, SSD1306_WHITE);

  display.display(); 
}

void setup() {
  Serial.begin(115200);
  Serial.println();

  WiFi.mode(WIFI_STA);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  display.clearDisplay();

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);

  pinMode(IND1, OUTPUT);

  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);

  Serial.println("connecting to WIFI...");
  if (wm.autoConnect("hackingHeuristics", "12341234")) {
    Serial.println("connected!");
  }
  else {
    Serial.println("Configportal running...");
  }
  Serial.println(" done");

  // listen for temperature changes
  server.on(F("/update"), HTTP_POST, updateOutput);
  server.begin();

  // get unique ID
  byte baseMac[6];
  WiFi.macAddress(baseMac);
  char baseMacChr[18] = {0};
  sprintf(
    baseMacChr, 
    "%02X:%02X:%02X:%02X:%02X:%02X", 
    baseMac[0], baseMac[1], baseMac[2], baseMac[3], baseMac[4], baseMac[5]
  );
  mac = String(baseMacChr);

  Serial.print("MacAddress: ");
  Serial.println(mac);

  // listen for UDP packets
  UDP.begin(UDP_PORT);

  // broadcast
  for (int i = 0; i < PORT_RANGE; i++) {
    UDP.beginPacket("255.255.255.255", PORT + i);
    UDP.println(mac);
    UDP.endPacket();
  }
}

void loop() {
  server.handleClient();

//  bool conn = checkConnection();
  int packetSize = UDP.parsePacket();
  if (packetSize) {
    int len = UDP.read(packet, 255);
    host = UDP.remoteIP();
    currentPort = UDP.remotePort();
    Serial.println(host);
  }

  // check connection:
  if (!client.connected()) {
    if (client.connect(host, currentPort)) {
      Serial.print("Connected to Gateway IP ");
      Serial.print(host);
      Serial.print(":");
      Serial.println(currentPort);
    } else {
      for (int i = 0; i < PORT_RANGE; i++) {
        UDP.beginPacket("255.255.255.255", PORT + i);
        UDP.println(mac);
        UDP.endPacket();
      }
      delay(500);
      return;
    }
  }
  

  // calculate elapsed time
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;

  static long timer = 0;
  timer -= interval;

  if(present > tempOffTime){
    turnTempsOff();
  }

  if(present > lastDisplayTime + DISPLAY_REFRESH){
    updateScreen();
    lastDisplayTime = present;
  }
  
  if (timer >= 0) {
    return;
  }

  timer += 1000000 / SAMPLE_RATE;

  readPins();

  active = read3 > 8;

  static int count = 0;

  // Data0:
  data0 += read0;
  if (count < DATA_LENGTH - 1) {
    data0 += ",";
  }

  // Data1
  data1 += read1;
  if (count < DATA_LENGTH - 1) {
    data1 += ",";
  }

  // Data2
  data2 += read2;
  if (count < DATA_LENGTH - 1) {
    data2 += ",";
  }

  // Data3
  data3 += read3;
  if (count < DATA_LENGTH - 1) {
    data3 += ",";
  }

  count++;

  // Construct data packet:
  if (count >= DATA_LENGTH) {
    String json = "{";
    json += "\"mac\":\"" + mac + "\",";
    json += "\"data0\":[" + data0 + "],";
    json += "\"data1\":[" + data1 + "],";
    json += "\"data2\":[" + data2 + "],";
    json += "\"data3\":[" + data3 + "],";

    if(active){
      json += "\"active\":true";
    } else {
      json += "\"active\":false";
    }

    json += "}#";

    // send data packet:
    client.println(json);

    // reset data streams:
    data0 = "";
    data1 = "";
    data2 = "";
    data3 = "";

    count = 0;
  }

}

bool checkConnection(){
  // check if broadcast has been accepted
  int packetSize = UDP.parsePacket();
  if (packetSize) {
    int len = UDP.read(packet, 255);
    host = UDP.remoteIP();
    currentPort = UDP.remotePort();
  }

  // check connection:
  if (!client.connected()) {
    if (client.connect(host, currentPort)) {
      Serial.print("Connected to Gateway IP ");
      Serial.print(host);
      Serial.print(":");
      Serial.println(currentPort);
    } else {
      for (int i = 0; i < PORT_RANGE; i++) {
        UDP.beginPacket("255.255.255.255", PORT + i);
        UDP.println(mac);
        UDP.endPacket();
      }
      delay(500);
      return false;
    }
  }

  return true;
}
