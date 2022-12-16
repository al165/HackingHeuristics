#include <WebServer.h>
#include <ArduinoJson.h>

#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiManager.h>

#include <Wire.h>
#include <NeuroStimDuino.h>

WiFiManager wm;
WebServer server(80);

WiFiUDP UDP;
char packet[255];
IPAddress host(192, 168, 178, 200);
const unsigned int UDP_PORT = 4210;
const unsigned int PORT = 8080;
const unsigned int PORT_RANGE = 10;
unsigned int currentPort = PORT;

WiFiClient client;


// Stimulator loop that continuously updates the AMP value of
// the Neurostimduino
TaskHandle_t Loop2;

int rampStates[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int rampVals[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int targetVals[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


// Number of micro-seconds between ramp updates
long RAMP_TIME = 100000;  // 100ms
unsigned long last_update_time = 0;


long lastBroadcastTime = 0;
const long BROADCAST = 5000000;


void setStimulation() {
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

  JsonObject postObj = doc.as<JsonObject>();
  bool success = false;
  int last_error = 0;

  if (!postObj.containsKey("channel") || !postObj.containsKey("addr")) {
    Serial.println("incomplete JSON: must contain 'channel' and 'addr'");
    server.send(400, F("text/html"), "body must contain `channel` and 'addr'");
    return;
  }

  uint8_t channel = postObj["channel"];
  uint8_t addr = postObj["addr"];

  setAddress(addr, 0);
  server.send(201, F("text/html"), "ok");

  if (postObj.containsKey("ampl")) {
    int idx = (addr - 0x1A) * 2 + (channel - 1);
    //    Serial.print("idx: ");
    //    Serial.print(idx);
    //    Serial.print(" addr: ");
    //    Serial.print(addr);
    //    Serial.print(" channel: ");
    //    Serial.println(channel);
    //
    uint8_t ampl = postObj["ampl"];
    //    if(rampVals[idx] > ampl){
    //      rampStates[idx] = -1;
    //    } else {
    //      rampStates[idx] = 1;
    //    }

    if (rampVals[idx] > 0) {
      rampStates[idx] = -1;
    } else {
      rampStates[idx] = 1;
    }
    targetVals[idx] = ampl;

  }

  if (postObj.containsKey("freq")) {
    int val = postObj["freq"];
    last_error = setFrequency(channel, val);
    //    Serial.println(last_error);

    if (last_error == 0) {
      success = true;
    }
  }

  if (postObj.containsKey("durn")) {
    int val = postObj["durn"];
    last_error = setDuration(channel, val);
    //    Serial.println(last_error);

    if (last_error == 0) {
      success = true;
    }
  }

  if (postObj.containsKey("idly")) {
    int val = postObj["idly"];
    last_error = setInterPhaseDelay(channel, val);
    //    Serial.println(last_error);

    if (last_error == 0) {
      success = true;
    }
  }

  if (postObj.containsKey("pulse")) {
    last_error = startPulse(channel);
    //    Serial.println(last_error);
  }

  last_error = startStimulation(channel, 2);
  //  Serial.println(last_error);
}

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

  NSWire.begin();
  server.on(F("/"), []() {
    Serial.println("hello");
    server.send(200, "text/plain", "hello from esp8266!");
  });

  // init states and values:
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 2; j++) {
      rampStates[i * 2 + j] = 0;
      rampVals[i * 2 + j] = 0;
      targetVals[i * 2 + j] = 0;
    }
  }

  server.on(F("/stim"), HTTP_POST, setStimulation);
  server.begin();

  // listen for UDP packets
  UDP.begin(UDP_PORT);

  // broadcast
  for (int i = 0; i < PORT_RANGE; i++) {
    UDP.beginPacket("255.255.255.255", PORT + i);
    UDP.println("stimulator");
    UDP.endPacket();
  }

}

void loop() {
  //  server.handleClient();
  for (;;) {
    unsigned long present = micros();
    server.handleClient();


    if (present > lastBroadcastTime + BROADCAST) {
      for (int i = 0; i < PORT_RANGE; i++) {
        UDP.beginPacket("255.255.255.255", PORT + i);
        UDP.println("stimulator");
        UDP.endPacket();
      }
      lastBroadcastTime = present;
    }



    if (present < last_update_time + RAMP_TIME) {
      continue;
    }

    last_update_time = present;

    for (uint8_t i = 0; i < 8; i++) {
      for (uint8_t j = 0; j < 2; j++) {
        if (rampStates[i * 2 + j] != 0) {
          updateRamp(26 + i, j + 1);
        }
      }
    }
  }
}


void updateRamp(uint8_t addr, uint8_t channel) {
  Serial.print("updateRamp :");
  Serial.print(addr);
  Serial.print(" ");
  Serial.println(channel);

  int idx = (addr - 26) * 2 + (channel - 1);

  Serial.print("idx: ");
  Serial.print(idx);
  Serial.print(" rampStates: ");
  Serial.print(rampStates[idx]);
  Serial.print(" targetVals: ");
  Serial.print(targetVals[idx]);
  Serial.print(" rampVals: ");
  Serial.println(rampVals[idx]);

  if (rampStates[idx] == -1) {
    if (rampVals[idx] == 0) {
      rampStates[idx] = 1;
    } else {
      rampVals[idx] -= 1;
      setAddress(addr, 0);
      int i2c_error = setAmplitude(channel, rampVals[idx]);
      startStimulation(channel, 2);
    }
  } else if (rampStates[idx] == 1) {
    if (rampVals[idx] >= targetVals[idx]) {
      rampStates[idx] = 0;
    } else {
      rampVals[idx] += 1;
      setAddress(addr, 0);
      int i2c_error = setAmplitude(channel, rampVals[idx]);
      startStimulation(channel, 2);
    }
  } else {
    // No change...
    Serial.println("no change");
    return;
  }
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
