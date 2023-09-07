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
#undef LED_BUILTIN
#define LED_BUILTIN 16
#endif

// #include <WifiClient.h>
#include <WiFiManager.h>
#define ARDUINOJSON_DECODE_UNICODE 0
#include <ArduinoJson.h>
#include <arduino-timer.h>

#include <Wire.h>
#include "NeuroStimDuino.h"
#include "NoiselessTouchESP32.h"

#define VALVE_PIN 15

#define TOUCH_HISTORY_LEN 6 // history length 
#define TOUCH_HYSTERESIS 2 // this is the sensitivity
const int TOUCH_PIN[] = {T9, T8};

NoiselessTouchESP32 touchsensor1(TOUCH_PIN[0], TOUCH_HISTORY_LEN, TOUCH_HYSTERESIS);
NoiselessTouchESP32 touchsensor2(TOUCH_PIN[1], TOUCH_HISTORY_LEN, TOUCH_HYSTERESIS);
NoiselessTouchESP32 touchsensor[2] = {touchsensor1,touchsensor2}; 

WiFiClient client;
WiFiManager wm;
AsyncUDP udp;

IPAddress host;
int port = 0;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

char mac[18];
char station[2] = {0, 0};

#define RAMP_TIME 300
int rampState = 0;
int rampVal = 0;
int targetVal = 0;

Timer<8> timer;

#define MAX_AIR_TIME 8
#define AIR_ON_WAIT 5

Timer<> air_timer;
// 0: off and ready, 1: off and cooling down, 2: on
bool valveState = 0;  

Timer<> blink_timer;


bool checkConnection(void *){
  if (!client.connected() && port != 0){
    client.connect(host, port);
  }

  return true;
}

bool ping(void *){
  if(client.connected()){
    client.printf("{\"server\":{\"type\": \"ping\", \"mac\":\"%s\"}}\n", mac);
    if(station[0] == 0){
      client.println("{\"server\":{\"type\": \"whoami\"}}");
    }
    blink();
  }

  return true;
}

void blink(){
  blink_timer.cancel();
  digitalWrite(LED_BUILTIN, HIGH);
  blink_timer.in(200, [](void*) -> bool {digitalWrite(LED_BUILTIN, LOW); return false;} );
}

void parsePacket(AsyncUDPPacket packet){
  static int airTime = 0;

  auto data = packet.data();
  blink();

  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, data);

  if(error){
    Serial.println("Error parsing JSON");
    return;
  }

  if(doc.containsKey(mac)){
    JsonObject details = doc[mac];
    if(details.containsKey("station")){
      itoa(details["station"], station, 10);
      Serial.print("station set to ");
      Serial.println(station);
    }

    if(details.containsKey("i2c")){
      uint8_t addr = details["i2c"];
      setAddress(addr, 0);
      Serial.printf("set i2c address to %u", addr);
    }
  }

  if(doc.containsKey("all")){
    if(doc["all"]["type"] == "lighthouse"){
      if(!client.connected()){
        Serial.println("lighthouse");
        host = packet.remoteIP();
        port = doc["all"]["tcp_port"];
      }
    }
  }

  if(!doc.containsKey(station) || station[0] == 0){
    return;
  }

  JsonObject parameters = doc[station];

  int last_error = -1;

  if (parameters.containsKey("ampl")) {
    int val = parameters["ampl"];
    targetVal = val;
    if (rampVal > targetVal) {
      rampState = -1;
    } else {
      rampState = 1;
    }
  }

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
    if(airTime > MAX_AIR_TIME){
      airTime = MAX_AIR_TIME;
    }
  }

  if (parameters.containsKey("airon")){
    // open or close air valve
    float val = parameters["airon"];

    // if(val < 0.5 && valveState == 2){
    //   turnValveOff(0);
    // }
    if(val >= 0.5 && valveState == 0) {
      valveState = 2;
      digitalWrite(VALVE_PIN, HIGH);
      Serial.println("turnValveOn");

      sendValveState();
      
      air_timer.in((int)(airTime * 1000), turnValveOff);
    } 
  }

  return;
}

bool turnValveOff(void *){
  Serial.println("turnValveOff");
  digitalWrite(VALVE_PIN, LOW);
  valveState = 1;
  sendValveState();

  air_timer.in((int)(AIR_ON_WAIT * 1000), setValveAvaliable);
  return true;
}

bool setValveAvaliable(void*){
  Serial.println("valveAvaliable again");
  valveState = 0; 
  sendValveState();
  return true;
}

bool updateTouch(void*){
  static int touchCount = 0;
  int count = 0;

  for (int i=0; i<2; i++){
    if(touchsensor[i].touching()){
      count++;
    }
  }

  if(count != touchCount){
    Serial.print("count ");
    Serial.println(count);

    touchCount = count;

    if(client.connected()){
      client.printf("{\"server\":{\"type\": \"touch_count\", \"touch_count\": %i, \"station\": \"%s\"}}\n", touchCount, station);
    }

    if(touchCount > 0 && valveState != 2){
      valveState = 2;
      digitalWrite(VALVE_PIN, HIGH);
      Serial.println("turnValveOn");

      sendValveState();
      
      air_timer.in((int)(5000), turnValveOff);
    }
  }
  return true;
}

void sendValveState(){
  if(client.connected()){
    client.printf("{\"server\":{\"type\": \"valve_state\", \"valve_state\": %i, \"station\": \"%s\"}}\n", valveState, station);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println();

  pinMode(VALVE_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  NSWire.begin();

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

  Serial.print("** MAC: ");
  Serial.println(mac);

  timer.every(5000, ping);
  timer.every(1000, checkConnection);
  timer.every(RAMP_TIME, updateRamp);
  timer.every(1000, updateTouch);
  sendValveState();
}

void loop() {
  timer.tick();
  blink_timer.tick();
  air_timer.tick();
}

bool updateRamp(void *) {
  if (rampState == -1) {
    if (rampVal <= targetVal) {
      rampState = 0;
    } else {
      rampVal -= 1;
      setAmplitude(1, rampVal);
      setAmplitude(2, rampVal);

      startStimulation(1, 2);
      startStimulation(2, 2);
    }
  } else if (rampState == 1) {
    if (rampVal >= targetVal) {
      rampState = 0;
    } else {
      rampVal += 1;
      setAmplitude(1, rampVal);
      setAmplitude(2, rampVal);

      startStimulation(1, 2);
      startStimulation(2, 2);
    }
  }

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


// NeuroStimDuino Stuff here ----------------------------
uint8_t setAmplitude(uint8_t channel, int val) {
  // Serial.print("setAmplitude, status ");

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

  // Serial.println(i2c_error);
  return i2c_error;
}

uint8_t setFrequency(uint8_t channel, int val) {
  // Serial.print("setFrequency, status ");

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
  // Serial.println(i2c_error);
  return i2c_error;
}

uint8_t setDuration(uint8_t channel, int val) {
  // Serial.print("setDuration, status ");

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

  // Serial.println(i2c_error);
  return i2c_error;
}

uint8_t setInterPhaseDelay(uint8_t channel, int val) {
  // Serial.print("setInterPhaseDelay, status ");

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
  // Serial.println(i2c_error);
  return i2c_error;
}

uint8_t startPulse(uint8_t channel) {
  uint8_t i2c_error = I2Cwrite(TwoBytesCommds, SAMP, channel, -1, -1);
  return i2c_error;
}

uint8_t enable(uint8_t channel, int val) {
  // Serial.print("enable, status ");

  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, ENAB, channel, val, -1);
  // Serial.println(i2c_error);
  return i2c_error;
}


uint8_t startStimulation(uint8_t channel, uint8_t val) {
  uint8_t flag;
  flag = 0;

  // Serial.print("startStimulation, status ");

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

  // Serial.println(i2c_error);
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
