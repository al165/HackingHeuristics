#include <Wire.h>

#include <HTTP_Method.h>
#include <Uri.h>
#include <WebServer.h>

//#include <ESP8266WiFi.h>
#include <WiFi.h>

#include <WiFiClient.h>
//#include <ESP8266WebServer.h>

#include <uri/UriBraces.h>

#include <NeuroStimDuino.h>

#include <strings_en.h>
#include <WiFiManager.h>

WiFiManager wm;
//ESP8266WebServer server(80);
WebServer server(80);



void setup() {
  Serial.begin(115200);
  Serial.println();

  NSWire.begin();

  Serial.println("connecting to WIFI...");
  if (wm.autoConnect("hackingHeuristics", "12341234")) {
    Serial.println("connected!");
  }
  else {
    Serial.println("Configportal running...");
  }
  Serial.println(" done");

  server.on(F("/"), []() {
    Serial.println("hello");
    server.send(200, "text/plain", "hello from esp8266!");
  });

  server.on(UriBraces("/{}/{}/{}"), []() {
    String cmd = server.pathArg(0);
    String chan_str = server.pathArg(1);
    String val_str = server.pathArg(2);

    Serial.println(cmd + "/" + chan_str + "/" + val_str);

    uint8_t chan = chan_str.toInt();
    int val = val_str.toInt();
    uint8_t result;

    if (cmd == "ampl") {
      result = setAmplitude(chan, val);
    } else if (cmd == "freq") {
      result = setFrequency(chan, val);
    } else if (cmd == "durn") {
      result = setDuration(chan, val);
    } else if (cmd == "idly") {
      result = setInterPhaseDelay(chan, val);
    } else if (cmd == "stim") {
      result = startStimulation(chan, val);
    } else if (cmd == "enab") {
      result = enable(chan, val);
    } else {
      result = 1;
    }
    Serial.println(result);
    server.send(200, "text/plain", "done. status = " + String(result) + "\r\n");
  });

  server.begin();

  
}

void loop() {
  server.handleClient();
}


// NeuroStimDuino Stuff here ----------------------------
uint8_t setAmplitude(uint8_t channel, int val) {
  Serial.println("setAmplitude");

  if (val < AMPL_LOW_LIMIT || val > AMPL_UPP_LIMIT) {
    Serial.print("value outside of range: ");
    Serial.print(val);
    Serial.print(" not in (");
    Serial.print(AMPL_LOW_LIMIT);
    Serial.print(", ");
    Serial.print(AMPL_UPP_LIMIT);
    Serial.println(")");

    return 1;
  }

  uint8_t val_lsb = (val & 255);
  uint8_t val_msb = ((val >> 8) & 255);  
  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, AMPL, channel, val_msb, val_lsb);

  return i2c_error;
}

uint8_t setFrequency(uint8_t channel, int val) {
  Serial.println("setFrequency");

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

  uint8_t val_lsb = (val & 255);
  uint8_t val_msb = ((val >> 8) & 255);
  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, FREQ, channel, val_lsb, val_msb);

  return i2c_error;
}

uint8_t setDuration(uint8_t channel, int val) {
  Serial.println("setFrequency");

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

  return i2c_error;
}

uint8_t setInterPhaseDelay(uint8_t channel, int val) {
  Serial.println("setFrequency");

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

  uint8_t val_lsb = (val & 255);
  uint8_t val_msb = ((val >> 8) & 255);
  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, IDLY, channel, val_lsb, val_msb);

  return i2c_error;
}

uint8_t enable(uint8_t channel, int val){
  Serial.println("enable");

  uint8_t val_lsb = (val & 255);
  uint8_t val_msb = ((val >> 8) & 255);
  uint8_t i2c_error = I2Cwrite(ThreeBytesCommds, ENAB, channel, val_lsb, val_msb);

  return i2c_error;
  
}

uint8_t startStimulation(uint8_t channel, uint8_t val) {
  uint8_t flag;
  if (val == 0) {
    flag = 1;
  }

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
  Serial.print(val);
  Serial.println(flag);
  uint8_t i2c_error = I2Cwrite(FourBytesCommds, STIM, channel, 10, 0);

  return i2c_error;
}
