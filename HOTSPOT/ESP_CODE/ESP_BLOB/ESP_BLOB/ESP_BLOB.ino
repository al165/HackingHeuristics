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
#include "NoiselessTouchESP32.h"

#define VALVE_PIN 15

#define TOUCH_HISTORY_LEN 6 // history length 
#define TOUCH_HYSTERESIS 8 // this is the sensitivity
const int TOUCH_PIN[] = {T9, T8};

NoiselessTouchESP32 touchsensor1(TOUCH_PIN[0], TOUCH_HISTORY_LEN, TOUCH_HYSTERESIS);
NoiselessTouchESP32 touchsensor2(TOUCH_PIN[1], TOUCH_HISTORY_LEN, TOUCH_HYSTERESIS);
NoiselessTouchESP32 touchsensor[2] = {touchsensor1,touchsensor2}; 

int touchCount = 0;

WiFiManager wm;
AsyncUDP udp;

IPAddress MCAST_GRP(224,3,29,71);
const int UDP_PORT = 10000;

String mac;
String station;

int rampStates = 0;
int rampVals = 0;
int targetVals = 0;

const int RAMP_TIME = 300;

// timer to simplify scheduling events
Timer<4> timer;

const int MAX_AIR_TIME = 8;
const int AIR_ON_WAIT = 5;
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
  String data(reinterpret_cast<char *>(packet.data()));

  Serial.println("--------");
  Serial.println(data);

  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, data);

  if(error){
    Serial.println("Error parsing JSON");
    return;
  }

  blink();
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

  JsonObject parameters = obj[station];
  String output;
  serializeJson(parameters, output);
  Serial.println(output);

  int last_error = -1;

  if (parameters.containsKey("ampl")) {
    int val = parameters["ampl"];
    if (rampVals > 0) {
      rampStates = -1;
    } else {
      rampStates = 1;
    }
    targetVals = val;
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
  }

  if (parameters.containsKey("airon")){
    // open or close air valve
    float val = parameters["airon"];
    if(val > MAX_AIR_TIME){
      val = (float) MAX_AIR_TIME;
    }

    // if(val < 0.5 && valveState == 2){
    //   turnValveOff(0);
    // }
    if(val >= 0.5 && valveState == 0) {
      valveState = 2;
      digitalWrite(VALVE_PIN, HIGH);
      Serial.println("turnValveOn");
      
      air_timer.in((int)(airTime * 1000), turnValveOff);
    } 
  }

  return;
}


bool turnValveOff(void *){
  Serial.println("turnValveOff");
  //air_timer.cancel();
  digitalWrite(VALVE_PIN, LOW);
  valveState = 1;

  air_timer.in((int)(AIR_ON_WAIT * 1000), setValveAvaliable);
  return true;
}

bool setValveAvaliable(void*){
  valveState = 0; 
  return true;
}

bool updateTouch(void*){
  int count = 0;
  for (int i=0; i<2; i++){
    if(touchsensor[i].touching()){
      count++;
    }
  }

  if(count != touchCount){
    touchCount = count;
    udp.printf("{\"0\":{\"type\": \"sensor\", \"touchCount\": \"%d\"}}\n", touchCount);
    Serial.printf("{\"0\":{\"type\": \"sensor\", \"touchCount\": \"%d\"}}\n", touchCount);
  }
  return true;
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

  mac = getMac();

  Serial.print("MAC: ");
  Serial.println(mac);

  timer.every(10000, ping);
  timer.every(RAMP_TIME, updateRamp);
  timer.every(100, updateTouch);
  ping(0);

  udp.print("{\"server\":{\"type\": \"whoami\"}}");

}

void loop() {
  timer.tick();
  air_timer.tick();
}

bool updateRamp(void *) {

  if (rampStates == -1) {
    if (rampVals == 0) {
      rampStates = 1;
    } else {
      rampVals -= 1;
      setAmplitude(1, rampVals);
      setAmplitude(2, rampVals);

      startStimulation(1, 2);
      startStimulation(2, 2);
    }
  } else if (rampStates == 1) {
    if (rampVals >= targetVals) {
      rampStates = 0;
    } else {
      rampVals += 1;
      setAmplitude(1, rampVals);
      setAmplitude(2, rampVals);

      startStimulation(1, 2);
      startStimulation(2, 2);
    }
  } else {
    // Serial.println("no change");
    
  }

  return true;
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