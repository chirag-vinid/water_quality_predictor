#include <OneWire.h>
#include <DallasTemperature.h>

// Pin Definitions
#define PH_PIN 32
#define TDS_PIN 35
#define TURB_PIN 34
#define TEMP_PIN 4
#define FLOW_PIN 14
#define RELAY_PIN 12

// Setup Temperature Sensor
OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

// Flow calculation variables
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
    pinMode(RELAY_PIN, OUTPUT);
    pinMode(FLOW_PIN, INPUT_PULLUP);
    
    // Interrupt for flow sensor pulses
    attachInterrupt(digitalPinToInterrupt(FLOW_PIN), pulseCounter, FALLING);
}

void loop() 
{
    // 1. Read Analog Sensors
    int phRaw = analogRead(PH_PIN);
    float phValue = map(phRaw, 0, 4095, 0, 1400) / 100.0; // Map to 0-14 pH

    int tdsRaw = analogRead(TDS_PIN);
    float tdsValue = map(tdsRaw, 0, 4095, 0, 1000); // Map to 0-1000 ppm

    int turbRaw = analogRead(TURB_PIN);
    float turbValue = map(turbRaw, 0, 4095, 0, 100); // Map to 0-100% turbidity

    // 2. Read Temperature
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);

    // 3. Calculate Flow Rate every 1 second
    if ((millis() - oldTime) > 1000) 
    {
        // YF-S201 formula: Pulse frequency (Hz) = 7.5Q, Q is flow rate in L/min
        flowRate = ((1000.0 / (millis() - oldTime)) * pulseCount) / 7.5;
        oldTime = millis();
        pulseCount = 0;
    }

    // 4. Actuator Logic (Example: Shut off if pH is too acidic)
    if (phValue < 6.5) 
    {
        digitalWrite(RELAY_PIN, HIGH); // Open relay/Solenoid
    } 
    else 
    {
        digitalWrite(RELAY_PIN, LOW);
    }

    // Output to Serial Monitor
    Serial.printf("pH: %.2f | TDS: %.0f ppm | Turb: %.1f%% | Temp: %.2f C | Flow: %.2f L/min\n", 
                  phValue, tdsValue, turbValue, tempC, flowRate);

    delay(2000);
}
