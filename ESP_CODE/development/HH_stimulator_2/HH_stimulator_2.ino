#include <WebServer.h>
#include <ArduinoJson.h>

#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiManager.h>

#include <Wire.h>
#include <NeuroStimDuino.h>

WiFiManager wm;
WebServer server(80);


void TCA9548A(uint8_t bus) {
  Wire.beginTransmission(0x70);  // TCA9548A address
  Wire.write(1 << bus);          // send byte to select bus
  Wire.endTransmission();
  Serial.print(bus);
}


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
    Serial.println("incomplete JSON: missing 'channel' or 'addr'");
    server.send(400, F("text/html"), "body missing `channel` or 'addr'");
    return;
  }

  uint8_t channel = postObj["channel"];
  uint8_t addr = postObj["addr"];
  
  TCA9548A(addr);

  server.send(201, F("text/html"), "ok");


  if (postObj.containsKey("ampl")) {
    int val = postObj["ampl"];
    last_error = setAmplitude(channel, val);
    Serial.println(last_error);

    if (last_error == 0) {
      success = true;
    }
  }

  if (postObj.containsKey("freq")) {
    int val = postObj["freq"];
    last_error = setFrequency(channel, val);
    Serial.println(last_error);

    if (last_error == 0) {
      success = true;
    }
  }

  if (postObj.containsKey("durn")) {
    int val = postObj["durn"];
    last_error = setDuration(channel, val);
    Serial.println(last_error);

    if (last_error == 0) {
      success = true;
    }
  }

  if (postObj.containsKey("idly")) {
    int val = postObj["idly"];
    last_error = setInterPhaseDelay(channel, val);
    Serial.println(last_error);

    if (last_error == 0) {
      success = true;
    }
  }

  enable(channel, 1);
  startStimulation(channel, 4);

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

  server.on(F("/stim"), HTTP_POST, setStimulation);
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

uint8_t enable(uint8_t channel, int val) {
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
