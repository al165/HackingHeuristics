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
#define LED_BLINK 2
const int EEG_PIN = 34;  // EEG
const int EOG_PIN = 32;  // EOG
const int HRT_PIN = 39;  // heart rate sensor
const int GSR_PIN = 36;  // GSR
const int MAX_ANALOG_INPUT = 4095;
#define AIN1 13
#define AIN2 12
#define BIN1 26
#define BIN2 27
#else
// ------- ESP8266 constants -------------------
#include <ESP8266HTTPClient.h>
#include <ESPAsyncUDP.h>
#undef LED_BLINK
#define LED_BLINK 16
const int EEG_PIN = A0;  // EEG
const int EOG_PIN = A0;  // EOG
const int HRT_PIN = A0;  // heart rate sensor
const int GSR_PIN = A0;  // GSR
const int MAX_ANALOG_INPUT = 1023;
// Temperature Pins :
#undef AIN1
#define AIN1 4
#undef AIN2
#define AIN2 5
#undef BIN1
#define BIN1 12
#undef BIN2
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
#include <Filters.h>
#include <Filters/IIRFilter.hpp>
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

char SERVER_URL[80];
bool connected = false;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

char mac[18];
char station[2] = {0, 0};

#define SAMPLE_RATE 512
#define DATA_LENGTH_SAMPLES 1024
#define FRAME_LENGTH 512
#define FR 4

const int sampleTime = 1000000 / SAMPLE_RATE;

#include <AH/Containers/Array.hpp>

// obained from:
// >>> import scipy.signal
// >>> b, a = scipy.signal.iirfilter(4, (25, 45), fs=512, ftype="butter")
Array<float, 9> b = {0.00016777, 0.0 , -0.00067107, 0.0, 0.00100661, 0.0, -0.00067107, 0.0, 0.00016777};
Array<float, 9> a = {1.0, -6.74163573, 20.43945109, -36.33587363, 41.39249857, -30.93258453, 14.81324491, -4.16010371, 0.52557897};
IIRFilter<9> iirFilter(b, a);


struct SensorData {
  float eeg[DATA_LENGTH_SAMPLES];
  float eog[DATA_LENGTH_SAMPLES];
  float hrt[DATA_LENGTH_SAMPLES];
  float gsr[DATA_LENGTH_SAMPLES];
};

SensorData sensorData;

struct SensorFeatures {
  float eeg_mean;
  float eeg_std;
  float eeg_delta;
  float eog_activity;
  float gsr_mean;
  float gsr_std;
  float gsr_delta;
  float hrt_mean;
  float hrt_std;
  bool active;
};
SensorFeatures sensorFeatures;

// Temperature "on" time
const long TEMP_ON = 5000;

// timer to simplify scheduling events
Timer<4> timer;
Timer<> blink_timer;
Timer<1, micros> sensor_timer;
Timer<> temp_timer;

//                              10 Features         +      2 Meta
const int json_capacity = 10 * JSON_OBJECT_SIZE(1)  + 2 * JSON_OBJECT_SIZE(1);


bool checkConnection(void *){
  if (!tcpClient.connected() && tcp_port != 0){
    tcpClient.connect(host, tcp_port);
  }

  return true;
}

bool ping(void *){
  if(tcpClient.connected()){
    // Serial.println("ping");
    tcpClient.printf("{\"server\":{\"type\":\"ping\",\"mac\":\"%s\"}}\n", mac);
    if(station[0] == 0){
      tcpClient.println("{\"server\":{\"type\":\"whoami\"}}");
    }
    blink();
  }
  return true;
}

void blink(){
  // blink_timer.cancel();
  digitalWrite(LED_BLINK, HIGH);
  blink_timer.in(100, [](void*) -> bool {digitalWrite(LED_BLINK, LOW);return false;} );
}

void getStats(float *data, int dataLength, float *mean, float *std){
  float sum = 0.0;
  float sum2 = 0.0;

  for(int i=0; i<dataLength; i++){
    sum += data[i];
    sum2 += (data[i] * data[i]);
  }

  *mean = sum/dataLength;
  float var = max((float)0.0, (sum2/dataLength) - (*mean * *mean));
  *std = sqrt(var);
}

void getDelta(float *data, int dataLength, float *delta){
  float sum = 0.0;
  for(int i=1; i<dataLength; i++){
    sum += (data[i] - data[i-1]);
  }

  *delta = sum/dataLength;
}

void getActivity(float *data, int dataLength, float mean, float sensitvity, float *activity){
  int numP = 0;
  int numN = 0;

  for(int i=0; i<dataLength; i++){
    if(data[i] > mean + sensitvity){
      numP++;
    } else if(data[i] < mean - sensitvity){
      numN++;
    }
  }
  *activity = (numN + numP) / (float)dataLength;
}

bool getFeatures(void *) {  
  getStats(sensorData.eeg, DATA_LENGTH_SAMPLES, &sensorFeatures.eeg_mean, &sensorFeatures.eeg_std);
  getStats(sensorData.gsr, DATA_LENGTH_SAMPLES, &sensorFeatures.gsr_mean, &sensorFeatures.gsr_std);
  getStats(sensorData.hrt, DATA_LENGTH_SAMPLES, &sensorFeatures.hrt_mean, &sensorFeatures.hrt_std);
  getDelta(sensorData.eeg, DATA_LENGTH_SAMPLES, &sensorFeatures.eeg_delta);
  getDelta(sensorData.gsr, DATA_LENGTH_SAMPLES, &sensorFeatures.gsr_delta);
  float eog_mean, eog_std;
  getStats(sensorData.eog, DATA_LENGTH_SAMPLES, &eog_mean, &eog_std);
  getActivity(sensorData.eog, DATA_LENGTH_SAMPLES, eog_mean, 0.4, &sensorFeatures.eog_activity);
  if(sensorFeatures.gsr_mean > 0.0){
    sensorFeatures.active = true;
  } else {
    sensorFeatures.active = false;
  }

  sendFeatures();
  return true;
}

int mod(int a, int b) {
  int c = a % b;
  return (c < 0) ? c + b : c;
}

bool readInput(void *){
  static int sample_ptr = 0;

  int eeg = analogRead(EEG_PIN);
  int eog = analogRead(EOG_PIN);
  int hrt = analogRead(HRT_PIN);
  int gsr = analogRead(GSR_PIN);

  //// DATA PROCESS ---------------------------------------
  // EEG Process
  float eeg_p = eeg / (float) MAX_ANALOG_INPUT;
  eeg_p = iirFilter(eeg_p);
  eeg_p = eeg_p * eeg_p;
  eeg_p = 2.0 * eeg_p - 1.0;

  // EOG Process
  float eog_p = 2.0 * eog / (float) MAX_ANALOG_INPUT - 1.0;

  // HRT Process
  float hrt_p = 2.0 * hrt / (float) MAX_ANALOG_INPUT - 1.0;

  // GSR Process
  float gsr_p = 2.0 * gsr / (float) MAX_ANALOG_INPUT - 1.0;


  sensorData.eeg[sample_ptr] = eeg_p;
  sensorData.eog[sample_ptr] = eog_p;
  sensorData.hrt[sample_ptr] = hrt_p;
  sensorData.gsr[sample_ptr] = gsr_p;

  sample_ptr = (sample_ptr + 1) % DATA_LENGTH_SAMPLES;

  return true;
}

void printFeatures(){
  Serial.printf("Features:\n");
  Serial.printf("  eeg_mean: %f\n", sensorFeatures.eeg_mean);
  Serial.printf("  eeg_std: %f\n", sensorFeatures.eeg_std);
  Serial.printf("  eeg_delta: %f\n", sensorFeatures.eeg_delta);
  Serial.printf("  eog_activity: %f\n", sensorFeatures.eog_activity);
  Serial.printf("  gsr_mean: %f\n", sensorFeatures.gsr_mean);
  Serial.printf("  gsr_std: %f\n", sensorFeatures.gsr_std);
  Serial.printf("  gsr_delta: %f\n", sensorFeatures.gsr_delta);
  Serial.printf("  hrt_mean: %f\n", sensorFeatures.hrt_mean);
  Serial.printf("  hrt_std: %f\n", sensorFeatures.hrt_std);
  Serial.printf("  active: %s\n", sensorFeatures.active ? "true":"false");
  Serial.println();
}

void sendFeatures(){
  if(!tcpClient.connected()){
    return;
  }

  DynamicJsonDocument doc(json_capacity);
  JsonObject server = doc.createNestedObject("server");
  server["active"] = sensorFeatures.active;
  server["EEG:mean"] = sensorFeatures.eeg_mean;
  server["EEG:std"] = sensorFeatures.eeg_std;
  server["EEG:delta"] = sensorFeatures.eeg_delta;
  server["EOG:all_rate"] = sensorFeatures.eog_activity;
  server["GSR:mean"] = sensorFeatures.gsr_mean;
  server["GSR:std"] = sensorFeatures.gsr_std;
  server["GSR:delta"] = sensorFeatures.gsr_delta;
  server["heart:mean"] = sensorFeatures.hrt_mean;
  server["heart:std"] = sensorFeatures.hrt_std;
  server["type"] = "sensor_features";

  if(doc.overflowed()){
    Serial.print("doc overflowed, expand capacity?");
  }

  char output[512];
  serializeJson(doc, output);
  // Serial.println(output);

  tcpClient.println(output);
  blink();
}

bool turnTempsOff(void *){
  Serial.println("Turning OFF temps");
  digitalWrite(AIN2, LOW);
  digitalWrite(AIN1, LOW);
  digitalWrite(BIN2, LOW);
  digitalWrite(BIN1, LOW);

  return true;
}

void parsePacket(AsyncUDPPacket packet){
  // Serial.print("parsePacket... ");
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

  if(doc.containsKey(mac)){
    JsonObject details = doc[mac];
    if(details.containsKey("station")){
      itoa(details["station"], station, 10);
      Serial.printf("Station set to %s\n", station);
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
      Serial.println("HIGHLIGHT ON");
    } else {
      digitalWrite(IND1, LOW);
      Serial.println("HIGHLIGHT OFF");
    }
  }

  if(parameters.containsKey("touch_count")){
    int val = parameters["touch_count"].as<int>();
    Serial.printf("touch_count: %u\n", val);

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
      temp_timer.in(TEMP_ON, turnTempsOff);
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
  pinMode(LED_BLINK, OUTPUT);
  digitalWrite(LED_BLINK, LOW);

  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {   // <<<<<< SHOULD BE 0x3D for 128x64 displays???
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  display.clearDisplay();

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

  sensor_timer.every(sampleTime, readInput);
  timer.every(4000, ping);
  timer.every(2000, checkConnection);
  timer.every(2000, getFeatures);
  timer.every(100, updateScreen);

  delay(100);
}

void loop() {
  timer.tick();
  blink_timer.tick();
  sensor_timer.tick();
  temp_timer.tick();
}

bool updateScreen(void *) {
  // Serial.println("updateScreen");
  // u8x8.setFont(u8x8_font_chroma48medium8_r);
  // u8x8.drawString(0,1,"Hello World!");
  // u8x8.setInverseFont(1);
  // u8x8.drawString(0,0,"012345678901234567890123456789");
  // u8x8.setInverseFont(0);
  // display.clearDisplay();

  // int maxRadius = min(display.width(), display.height());

  // int r0 = random(maxRadius);
  // int r1 = random(maxRadius);
  // int r2 = random(maxRadius);
  // int r3 = random(maxRadius);
 
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
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE, BLACK);
  display.printf(
    "%f.3f\n%f.3f\n%f.3f\n%f.3f", 
    sensorData.eeg[0],
    sensorData.eog[0],
    sensorData.gsr[0],
    sensorData.hrt[0]
  );
  // for(int i = 0; i < 4; i++){
  //   display.setCursor(64, 10+i*10);
  //   display.println(reads[i]);
  // }
  // display.println(screenMessage);

  display.display(); 
  // Serial.println("display done");
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

