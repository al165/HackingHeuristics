
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const int pinOuts[] = {26, 16, 17, 18, 19, 23};

uint8_t count = 0;

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

void setup() {
  Serial.begin(115200);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }

    for (count=0;count<6;count++) {
  pinMode( pinOuts[count], OUTPUT);
  digitalWrite( pinOuts[count], LOW);
    }
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(20, 20);
  display.println(" ..START UP..");
  display.display(); 
    delay(2000);

}

void loop() {
  for (count=0;count<6;count++) {

  display.clearDisplay();
  display.setTextSize(4);
  display.setTextColor(WHITE);
  display.setCursor(60, 20);
  display.println(count);
  display.display(); 
  
   digitalWrite( pinOuts[count], HIGH);
    delay(1000); 

  digitalWrite( pinOuts[count], LOW);
    delay(1000); 

  
}

}
