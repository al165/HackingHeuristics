/*
  BLOB ESP CODE

  - Receives stimulator commands
  - Controls air valves

*/

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
#include "NeuroStimDuino.h"

#define VALVE_PIN 15

const char* WIFI_NAME = "H369A09B46E";
const char* WIFI_PWD = "2FF3F4323667";

WiFiManager wm;
AsyncUDP udp;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

String mac;
String station;

// timer to simplify scheduling events
Timer<4> timer;

const int MAX_AIR_TIME = 10;
const int AIR_ON_WAIT = 10;
float airTime = 0;
Timer<> air_timer;
// 0: off and ready, 1: off and cooling down, 2: on
bool valveState = 0;  

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

  if(obj.containsKey(mac)){
    JsonObject details = obj[mac];
    if(details.containsKey("station")){
      // sprintf(station, "%d", details["station"]);
      station = details["station"].as<String>();
      Serial.print("updated station to ");
      Serial.println(station);
    }
  }

  if(!obj.containsKey(station)){
    return;
  }

  JsonObject parameters = obj[mac];
  String output;
  serializeJson(parameters, output);
  Serial.println("recieved parameters:");
  Serial.println(output);

  int last_error = -1;

  if (parameters.containsKey("freq")) {
    int val = parameters["freq"];
    last_error = setFrequency(1, val);
    last_error = setFrequency(2, val);
  }

  if (parameters.containsKey("durn")) {
    int val = parameters["durn"];
    last_error = setDuration(1, val);
    last_error = setDuration(2, val);
  }

  if (parameters.containsKey("idly")) {
    int val = parameters["idly"];
    last_error = setInterPhaseDelay(1, val);
    last_error = setInterPhaseDelay(2, val);
  }

  if (parameters.containsKey("pulse")) {
    last_error = startPulse(1);
    last_error = startPulse(2);
  }

  last_error = startStimulation(1, 2);
  last_error = startStimulation(2, 2);

  if (parameters.containsKey("airtime")){
    // set length of air valve time
    airTime = parameters["airtime"];
  }

  if (parameters.containsKey("airon")){
    // open or close air valve
    float val = parameters["airon"];
    if(val > MAX_AIR_TIME){
      val = (float) MAX_AIR_TIME;
    }

    if(val < 0.5 && valveState == 2){
      turnValveOff(0);
    } else if(val >= 0.5 && valveState == 0) {
      valveState = 2;
      digitalWrite(VALVE_PIN, HIGH);
      
      air_timer.in((int)(val * 1000), turnValveOff);
    } 
  }

  return;
}


bool turnValveOff(void *){
  air_timer.cancel();
  digitalWrite(VALVE_PIN, LOW);
  valveState = 1;

  air_timer.in((int)(AIR_ON_WAIT * 1000), setValveState, (void*) 0);
  return true;
}

bool setValveState(void* state){
  Serial.print("setValveState ");
  Serial.println((int) state);
  valveState = (int) state; 
  return true;
}


void setup() {
  Serial.begin(115200);
  Serial.println();

  pinMode(VALVE_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

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
  ping(0);

}

void loop() {
  timer.tick();
  air_timer.tick();
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


// NeuroStimDuino Stuff here ----------------------------
uint8_t setAmplitude(uint8_t channel, int val) {
  Serial.print("setAmplitude, status ");

  if (val < AMPL_LOW_LIMIT || val > 10) {
    Serial.print("value outside of range: ");
    Serial.print(val);
    Serial.print(" not in (");
    Serial.print(AMPL_LOW_LIMIT);
    Serial.print(", ");
    Serial.print(AMPL_UPP_LIMIT);
    Serial.println(")");

    return 1;
  }

  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, AMPL, channel, val, -1);

  Serial.println(i2c_error);
  return i2c_error;
}

uint8_t setFrequency(uint8_t channel, int val) {
  Serial.print("setFrequency, status ");

  if (val < FREQ_LOW_LIMIT || val > FREQ_UPP_LIMIT) {
    Serial.print("value outside of range: ");
    Serial.print(val);
    Serial.print(" not in (");
    Serial.print(FREQ_LOW_LIMIT);
    Serial.print(", ");
    Serial.print(FREQ_UPP_LIMIT);
    Serial.println(")");

    return 1;
  }

  //  uint8_t val_lsb = (val & 255);
  //  uint8_t val_msb = ((val >> 8) & 255);
  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, FREQ, channel, val, -1);
  Serial.println(i2c_error);
  return i2c_error;
}

uint8_t setDuration(uint8_t channel, int val) {
  Serial.print("setDuration, status ");

  if (val < DURN_LOW_LIMIT || val > DURN_UPP_LIMIT) {
    Serial.print("value outside of range: ");
    Serial.print(val);
    Serial.print(" not in (");
    Serial.print(DURN_LOW_LIMIT);
    Serial.print(", ");
    Serial.print(DURN_UPP_LIMIT);
    Serial.println(")");

    return 1;
  }

  uint8_t val_lsb = (val & 255);
  uint8_t val_msb = ((val >> 8) & 255);
  uint8_t i2c_error = I2Cwrite(FourBytesCommds, DURN, channel, val_lsb, val_msb);

  Serial.println(i2c_error);
  return i2c_error;
}

uint8_t setInterPhaseDelay(uint8_t channel, int val) {
  Serial.print("setInterPhaseDelay, status ");

  if (val < IDLY_LOW_LIMIT || val > IDLY_UPP_LIMIT) {
    Serial.print("value outside of range: ");
    Serial.print(val);
    Serial.print(" not in (");
    Serial.print(IDLY_LOW_LIMIT);
    Serial.print(", ");
    Serial.print(IDLY_UPP_LIMIT);
    Serial.println(")");

    return 1;
  }

  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, IDLY, channel, val, -1);
  Serial.println(i2c_error);
  return i2c_error;
}

uint8_t startPulse(uint8_t channel) {
  uint8_t i2c_error = I2Cwrite(TwoBytesCommds, SAMP, channel, -1, -1);
  return i2c_error;
}

uint8_t enable(uint8_t channel, int val) {
  Serial.print("enable, status ");

  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, ENAB, channel, val, -1);
  Serial.println(i2c_error);
  return i2c_error;
}


uint8_t startStimulation(uint8_t channel, uint8_t val) {
  uint8_t flag;
  flag = 0;

  Serial.print("startStimulation, status ");

  if (val < 0 || val > 255) {
    Serial.print("value outside of range: ");
    Serial.print(val);
    Serial.print(" not in (");
    Serial.print(0);
    Serial.print(", ");
    Serial.print(255);
    Serial.println(")");

    return 1;
  }
  uint8_t i2c_error = I2Cwrite(FourBytesCommds, STIM, channel, val, flag);

  Serial.println(i2c_error);
  return i2c_error;
}

uint8_t setAddress(uint8_t address, uint8_t flag) {
  if (flag == 1) {
    uint8_t val_lsb = (address & 255);
    uint8_t val_msb = ((address >> 8) & 255);
    uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, ADDR, -1, val_lsb, val_msb);
    if (i2c_error != 0) {
      Serial.print("I2C error = ");
      Serial.println(i2c_error);
    } else {
      NSDuino_address = address; // Update I2C address of current slave that will be communicated with
      Serial.print("Updated I2C slave address = ");
      Serial.println(NSDuino_address, DEC);
    }

    return i2c_error;
  } else {
    //Only change I2C address of current slave device. Do not program the I2C address
    NSDuino_address = address;
    Serial.print("Now connected to I2C slave address = ");
    Serial.println(NSDuino_address, DEC);

    return 0;
  }
}