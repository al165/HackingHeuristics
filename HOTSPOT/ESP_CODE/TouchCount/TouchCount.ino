#include <ArduinoJson.h>

int TOUCH_PIN[] =    {T9, T8}; 
// pinouts touch pins: 32, 33, 


#define TOUCH_HISTORY_LEN 6 // history length 
#define TOUCH_HYSTERESIS 2 // this is the sensitivity

#include <NoiselessTouchESP32.h>

// change the last number called in each instance to change the sensitivity of the touch sensor
// the higher the number the less sensitive, the lower the more sensitive 
// has to be multiple of two 

NoiselessTouchESP32 touchsensor1(TOUCH_PIN[0], TOUCH_HISTORY_LEN, TOUCH_HYSTERESIS);
NoiselessTouchESP32 touchsensor2(TOUCH_PIN[1], TOUCH_HISTORY_LEN, TOUCH_HYSTERESIS);


NoiselessTouchESP32 touchsensor[2] = {touchsensor1,touchsensor2}; 


void setup() {
  Serial.begin(115200);

  
}

int counter[2];
int touch[2];
int raw_mean_value[2];
int highest[2] = {0,0};
int lowest[2] = {255,255};



void loop() {
DynamicJsonDocument doc(8192);
JsonArray countValues = doc.createNestedArray("touch counts");

  for (int i=0; i<2; i++){

  touch[i] = touchsensor[i].touching();
  raw_mean_value[i] = touchsensor[i].value_from_history();
  highest[i] = _max(highest[i], raw_mean_value[i]);
  lowest[i] = _min(lowest[i], raw_mean_value[i]);
 
  if (touch[i] == 1) {
    
    counter[i]++;
  }


  //int value = counter[i]; --- this one adds total time touched
  int value = touch[i]; // this is just present time
  countValues.add(value);
  }
   serializeJson(doc, Serial);
 Serial.println();
delay(50);

}
