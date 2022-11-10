#if defined(ESP32)
#include <WiFi.h>
#include <WebServer.h>
WebServer server(80);
const int SENSOR0 = 34;  // EEG
const int SENSOR1 = 32;  // EOG
const int SENSOR2 = 39;  // heart rate sensor
const int SENSOR3 = 36;  // GSR
#else
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
ESP8266WebServer server(80);
const int SENSOR0 = A0;  // EEG
const int SENSOR1 = A0;  // EOG
const int SENSOR2 = A0;  // heart rate sensor
const int SENSOR3 = A0;  // GSR
#endif


#include <ArduinoJson.h>
#include <WiFiClient.h>

//#include <strings_en.h>
#include <WiFiManager.h>

#include <WiFiUdp.h>

WiFiManager wm;
IPAddress host(192, 168, 2, 8);
const unsigned int PORT = 8080;
const unsigned int PORT_RANGE = 10;
unsigned int currentPort = PORT;

WiFiClient client;

String data0 = "";
String data1 = "";
String data2 = "";
String data3 = "";

const int DATA_LENGTH = 64;
const int SAMPLE_RATE = 1024;

WiFiUDP UDP;
char packet[255];
const unsigned int UDP_PORT = 4210;

String mac;

// Temperature Pins:
#define AIN1 13
#define AIN2 12

#define BIN1 26
#define BIN2 27

// Temperature update time
const long TEMP_UPDATE = 5000000;
// Temperature "on" time
const long TEMP_ON = 2000000;

long lastTempTime = 0;
long tempOffTime = 0;
bool tempOn = false;

void setTemp() {
  String postBody = server.arg("plain");
  Serial.println("--------");
  Serial.println(postBody);

  DynamicJsonDocument doc(512);
  DeserializationError error = deserializeJson(doc, postBody);

  if (error) {
    Serial.print(F("Error parsing JSON "));
    Serial.println(error.c_str());

    server.send(400, F("text/html"), "Error in parsing JSON body");
    return;
  }

  // check if should ignore...
  if (micros() - lastTempTime < TEMP_UPDATE) {
    server.send(201);
    return;
  }

  JsonObject postObj = doc.as<JsonObject>();

  bool success = false;

  if (postObj.containsKey("temp1")) {
    success = true;
    float temp = postObj["temp1"];

    if (temp > 0.2) {
      // make warm...
      Serial.println("warming up 1");

      digitalWrite(AIN2, LOW);
      digitalWrite(AIN1, HIGH);

    } else if (temp < -0.2) {
      // make cold...
      Serial.println("cooling down 1");

      digitalWrite(AIN1, LOW);
      digitalWrite(AIN2, HIGH);

    } else {
      // turn off...
      Serial.println("turning off 1");

      digitalWrite(AIN1, LOW);
      digitalWrite(AIN2, LOW);
    }
  }

  if (postObj.containsKey("temp2")) {
    success = true;
    float temp = postObj["temp2"];

    if (temp > 0.2) {
      // make warm...
      Serial.println("warming up 2");

      digitalWrite(BIN2, LOW);
      digitalWrite(BIN1, HIGH);

    } else if (temp < -0.2) {
      // make cold...
      Serial.println("cooling down 2");

      digitalWrite(BIN2, LOW);
      digitalWrite(BIN1, HIGH);

    } else {
      // turn off...
      Serial.println("turning off 2");

      digitalWrite(BIN2, LOW);
      digitalWrite(BIN1, LOW);
    }
  }

  if (success) {
    lastTempTime = micros();
    tempOffTime = lastTempTime + TEMP_ON;
    tempOn = true;
    server.send(201);
  } else {
    server.send(400, F("text/html"), "neither temp1 or temp2 specified");
  }
}

void turnTempsOff() {
  digitalWrite(AIN2, LOW);
  digitalWrite(AIN1, LOW);
  digitalWrite(BIN2, LOW);
  digitalWrite(BIN1, LOW);

  tempOn = false;
}

void setup() {
  Serial.begin(115200);
  Serial.println();

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);

  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);

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
  server.on(F("/temp"), HTTP_POST, setTemp);
  server.begin();

  // get unique ID
  byte baseMac[6];
  WiFi.macAddress(baseMac);
  char baseMacChr[18] = {0};
  sprintf(baseMacChr, "%02X:%02X:%02X:%02X:%02X:%02X", baseMac[0], baseMac[1], baseMac[2], baseMac[3], baseMac[4], baseMac[5]);
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

  // check if broadcast has been accepted
  int packetSize = UDP.parsePacket();
  if (packetSize) {
    Serial.println("recieved packet from broadcast");
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
      Serial.println();
    } else {
      Serial.print("Could not connect to Gateway IP ");
      Serial.print(host);
      Serial.print(":");
      Serial.println(currentPort);
      Serial.println();

      Serial.print("checking ports ");
      for (int i = 0; i < PORT_RANGE; i++) {
        UDP.beginPacket("255.255.255.255", PORT + i);
        UDP.println(mac);
        UDP.endPacket();
        Serial.print(PORT + i);
        Serial.print(", ");
      }
      Serial.println();
      delay(500);
      return;
    }

  }

  server.handleClient();

  // calculate elapsed time
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;

  static long timer = 0;
  timer -= interval;

  if(tempOn && (present > tempOffTime)){
    turnTempsOff();
  }

  if (timer >= 0) {
    return;
  }

  timer += 1000000 / SAMPLE_RATE;

  // Read data pins:
  static int count = 0;

  // Data0:
  data0 += analogRead(SENSOR0);
  if (count < DATA_LENGTH - 1) {
    data0 += ",";
  }

  // Data1
  data1 += analogRead(SENSOR1);
  if (count < DATA_LENGTH - 1) {
    data1 += ",";
  }

  // Data2
  data2 += analogRead(SENSOR2);
  if (count < DATA_LENGTH - 1) {
    data2 += ",";
  }

  // Data3
  data3 += analogRead(SENSOR3);
  if (count < DATA_LENGTH - 1) {
    data3 += ",";
  }

  count++;

  // Construct data packet:
  if (count == DATA_LENGTH) {
    String json = "{";
    json += "\"mac\":\"" + mac + "\",";
    json += "\"data0\":[" + data0 + "],";
    json += "\"data1\":[" + data1 + "],";
    json += "\"data2\":[" + data2 + "],";
    json += "\"data3\":[" + data3 + "]";
    json += "}#";

    // send data packet:
    client.print(json);
    //    Serial.println(json);

    // reset data streams:
    data0 = "";
    data1 = "";
    data2 = "";
    data3 = "";

    count = 0;
  }

}
