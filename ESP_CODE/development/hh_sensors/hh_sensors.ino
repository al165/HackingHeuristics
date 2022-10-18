//#include <ESP8266WiFi.h>
#include <WiFi.h>

#include <strings_en.h>
#include <WiFiManager.h>

#include <WiFiUdp.h>


WiFiManager wm;
IPAddress server(192, 168, 23, 90);
const unsigned int PORT = 8080;
const unsigned int PORT_RANGE = 10;
unsigned int currentPort = PORT;

WiFiClient client;

//const int SENSOR0 = 34;

const int SENSOR0 = 34;  // EEG
const int SENSOR1 = 32;  // EOG
const int SENSOR2 = 39;  // heart rate sensor
const int SENSOR3 = 36;  // GSR

String data0 = "";
String data1 = "";
String data2 = "";
String data3 = "";

const int DATA_LENGTH = 64;
const int SAMPLE_RATE = 1024;

WiFiUDP UDP;
char packet[255];
const unsigned int UDP_PORT = 4210;

void setup() {
  Serial.begin(115200);
  Serial.println();

  Serial.println("connecting to WIFI...");
  if (wm.autoConnect("hackingHeuristics", "12341234")) {
    Serial.println("connected!");
  }
  else {
    Serial.println("Configportal running...");
  }
  Serial.println(" done");

  // listen for UDP packets
  UDP.begin(UDP_PORT);

  // broadcast
  for (int i = 0; i < PORT_RANGE; i++) {
    UDP.beginPacket("255.255.255.255", PORT + i);
    UDP.println("HH");
    UDP.endPacket();
  }
}

void loop() {
  // check if broadcast has been accepted
  int packetSize = UDP.parsePacket();
  if (packetSize) {
    Serial.println("recieved packet from broadcast");
    int len = UDP.read(packet, 255);
    server = UDP.remoteIP();
    currentPort = UDP.remotePort();
  }

  // check connection:
  if (!client.connected()) {
    if (client.connect(server, currentPort)) {
      Serial.print("Connected to Gateway IP ");
      Serial.print(server);
      Serial.print(":");
      Serial.println(currentPort);
      Serial.println();
    } else {
      Serial.print("Could not connect to Gateway IP ");
      Serial.print(server);
      Serial.print(":");
      Serial.println(currentPort);
      Serial.println();

      Serial.print("checking ports ");
      for (int i = 0; i < PORT_RANGE; i++) {
        UDP.beginPacket("255.255.255.255", PORT + i);
        UDP.println("HH");
        UDP.endPacket();
        Serial.print(PORT + i);
        Serial.print(", ");
      }
      Serial.println();
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
