#include <OneWire.h>
#include <DallasTemperature.h>

#define PH_PIN 32
#define TDS_PIN 35
#define TURB_PIN 34
#define TEMP_PIN 4
#define FLOW_PIN 14

OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

volatile int pulseCount = 0;
float flowRate = 0.0;
unsigned long oldTime = 0;

void IRAM_ATTR pulseCounter() 
{
    pulseCount++;
}

void setup() 
{
    Serial.begin(115200);
    
    sensors.begin();
    
    // Set flow pin with internal pullup to detect the button press to GND
    pinMode(FLOW_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(FLOW_PIN), pulseCounter, FALLING);
    
    Serial.println("Water Quality Monitoring System Initialized...");
}

void loop() 
{
    // 1. Read Analog Sensors (pH, TDS, Turbidity)
    int phRaw = analogRead(PH_PIN);
    float phValue = map(phRaw, 0, 4095, 0, 1400) / 100.0;

    int tdsRaw = analogRead(TDS_PIN);
    float tdsValue = map(tdsRaw, 0, 4095, 0, 1000);

    int turbRaw = analogRead(TURB_PIN);
    float turbValue = map(turbRaw, 0, 4095, 0, 100);

    // 2. Read Temperature (DS18B20)
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);

    // 3. Calculate Flow Rate every 1 second
    if ((millis() - oldTime) > 1000) 
    {
        // YF-S201 formula: Flow rate (L/min) = Frequency / 7.5
        flowRate = ((1000.0 / (millis() - oldTime)) * pulseCount) / 7.5;
        
        // Print results to Serial Monitor
        Serial.print("pH: "); Serial.print(phValue);
        Serial.print(" | TDS: "); Serial.print(tdsValue); Serial.print(" ppm");
        Serial.print(" | Turbidity: "); Serial.print(turbValue); Serial.print("%");
        Serial.print(" | Temp: "); Serial.print(tempC); Serial.print("°C");
        Serial.print(" | Flow: "); Serial.print(flowRate); Serial.println(" L/min");

        oldTime = millis();
        pulseCount = 0;
    }
}
