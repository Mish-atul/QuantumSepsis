# QuantumSepsis Shield — Hardware Integration Guide

## Architecture Overview

```
┌──────────────────┐       WiFi / HTTP        ┌─────────────────────┐       POST /predict       ┌──────────────────────┐
│  HARDWARE LAYER  │ ──────────────────────▶  │   REACT FRONTEND    │ ──────────────────────▶   │   AWS EC2 BACKEND    │
│                  │                          │   (Vercel)          │                           │   (FastAPI + ML)     │
│  ESP32 / RPi     │       BLE / Serial       │                     │  ◀──────────────────────  │                      │
│  + Sensor Array  │ ──────────────────────▶  │   /demo route       │       JSON response       │   LSTM + XGBoost +   │
│                  │   (Web Serial / BLE API) │   DemoSimulator.tsx  │       risk, tripwires,    │   Conformal + Red    │
└──────────────────┘                          └─────────────────────┘       actions              │   Team Agent         │
                                                                                                └──────────────────────┘
```

**Data Flow:**  
`Sensors → MCU (ESP32/RPi) → React Frontend → AWS FastAPI → ML Ensemble → Response → UI Display`

---

## 1. Hardware Components

### Option A: ESP32-Based (Recommended — Cheapest, WiFi Built-in)

| Component | Model | Reads | Approx. Cost |
|---|---|---|---|
| Microcontroller | **ESP32-WROOM-32** | — | ₹400 / $5 |
| Pulse Oximeter | **MAX30102** | Heart Rate, SpO₂ | ₹150 / $3 |
| Temperature | **MLX90614** (IR, non-contact) | Temperature (°C) | ₹350 / $5 |
| Blood Pressure | **HX710B + Cuff** (analog) | SBP, DBP → MAP | ₹500 / $8 |
| Respiration | **Piezo belt sensor** or **ADXL345 accelerometer** | Resp Rate | ₹200 / $3 |
| OLED Display | **SSD1306 0.96"** (optional) | Local readout | ₹150 / $2 |

> **NOTE:** GCS, Lactate, WBC, Creatinine, and Platelets cannot be read by bedside hardware sensors. These are lab values entered manually via the UI sliders. The hardware only automates the 5 real-time vitals: **HR, SpO₂, Temperature, Resp Rate, and MAP**.

### Option B: Raspberry Pi-Based (More Processing Power)

| Component | Model | Notes |
|---|---|---|
| SBC | **Raspberry Pi 4 / Zero 2 W** | Runs Python directly, has WiFi |
| Pulse Ox | MAX30102 (same) | Via I2C |
| Temp | MLX90614 (same) | Via I2C |
| BP | USB BP Monitor (e.g., Omron) | Parse serial output |
| Resp | Piezo / Accelerometer | Via ADC (MCP3008) |

---

## 2. Wiring Diagram (ESP32 + MAX30102 + MLX90614)

```
ESP32-WROOM-32 Pin Layout
─────────────────────────

        ┌────────────────────┐
        │      ESP32         │
        │                    │
 3.3V ──┤ 3V3          GND  ├── GND (shared)
        │                    │
 SDA  ──┤ GPIO 21    GPIO 22├── SCL
        │  (I2C Data)  (I2C Clock)
        │                    │
        │             GPIO 19├── Piezo Resp Sensor (Analog In)
        │                    │
        │             GPIO 5 ├── OLED CS (optional)
        └────────────────────┘
          │           │
     ┌────┴───┐  ┌────┴───┐
     │MAX30102│  │MLX90614│
     │ SDA    │  │ SDA    │
     │ SCL    │  │ SCL    │
     │ VIN→3V3│  │ VIN→3V3│
     │ GND    │  │ GND    │
     └────────┘  └────────┘

Both sensors share the I2C bus (same SDA/SCL lines).
```

---

## 3. ESP32 Firmware (Arduino / PlatformIO)

This firmware reads all sensors and sends a JSON payload to the React frontend every 5 seconds via HTTP POST.

### `firmware.ino`

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"           // SparkFun MAX3010x library
#include "heartRate.h"
#include <Adafruit_MLX90614.h>
#include <ArduinoJson.h>

// ── WiFi Config ──────────────────────────────
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ── Target: Your React frontend or direct to AWS ──
// Option 1: Direct to AWS backend
const char* serverUrl = "http://44.220.161.215:8000/predict";
// Option 2: Through your Vercel frontend proxy
// const char* serverUrl = "https://your-vercel-app.vercel.app/api/predict";

// ── Sensor Objects ───────────────────────────
MAX30105 particleSensor;
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// ── Heart Rate Averaging ─────────────────────
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute;
int beatAvg;

// ── Respiration (piezo on GPIO 19) ───────────
const int RESP_PIN = 19;
volatile int breathCount = 0;
unsigned long lastRespCheck = 0;

// ── Reading interval ─────────────────────────
const unsigned long SEND_INTERVAL = 5000; // 5 seconds
unsigned long lastSend = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA=21, SCL=22

  // Connect WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  // Initialize MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!");
  } else {
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
  }

  // Initialize MLX90614
  if (!mlx.begin()) {
    Serial.println("MLX90614 not found!");
  }

  // Respiration piezo pin
  pinMode(RESP_PIN, INPUT);
}

void loop() {
  // ── Read Heart Rate + SpO2 ─────────────────
  long irValue = particleSensor.getIR();
  if (checkForBeat(irValue)) {
    long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60 / (delta / 1000.0);
    if (beatsPerMinute > 20 && beatsPerMinute < 255) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;
      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  // Estimate SpO2 (simplified — real clinical devices use ratio-of-ratios)
  float spo2 = 0;
  if (irValue > 50000) {
    // Rough estimation; replace with proper R-value calculation for clinical use
    spo2 = constrain(map(irValue, 50000, 150000, 90, 100), 80, 100);
  }

  // ── Read Temperature ───────────────────────
  float tempC = mlx.readObjectTempC();

  // ── Read Respiration Rate ──────────────────
  // Simple threshold-crossing detection on the piezo signal
  static bool wasAbove = false;
  int respVal = analogRead(RESP_PIN);
  bool isAbove = respVal > 2048; // midpoint of 12-bit ADC
  if (isAbove && !wasAbove) breathCount++;
  wasAbove = isAbove;

  // Calculate breaths per minute every SEND_INTERVAL
  unsigned long now = millis();
  if (now - lastSend >= SEND_INTERVAL) {
    float respRate = (breathCount * 60000.0) / (now - lastRespCheck);
    breathCount = 0;
    lastRespCheck = now;
    lastSend = now;

    // ── Build JSON Payload ─────────────────────
    StaticJsonDocument<256> doc;
    doc["heart_rate"]   = (beatAvg > 0) ? beatAvg : 75;
    doc["spo2"]         = (spo2 > 0) ? spo2 : 98;
    doc["temperature"]  = (tempC > 30 && tempC < 45) ? tempC : 37.0;
    doc["resp_rate"]    = (respRate > 5 && respRate < 60) ? respRate : 16;
    doc["map"]          = 85;         // Manual or from BP cuff
    doc["gcs_total"]    = 15;         // Manual input (clinician)
    doc["lactate"]      = 1.0;        // Lab value (manual)
    doc["wbc"]          = 8.0;        // Lab value (manual)
    doc["creatinine"]   = 0.9;        // Lab value (manual)
    doc["platelets"]    = 220;        // Lab value (manual)

    String payload;
    serializeJson(doc, payload);

    // ── Send to Backend ────────────────────────
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverUrl);
      http.addHeader("Content-Type", "application/json");
      int httpCode = http.POST(payload);

      if (httpCode > 0) {
        Serial.printf("[HTTP] POST response: %d\n", httpCode);
        String response = http.getString();
        Serial.println(response);
      } else {
        Serial.printf("[HTTP] POST failed: %s\n", http.errorToString(httpCode).c_str());
      }
      http.end();
    }
  }
}
```

### Required Arduino Libraries

```
SparkFun MAX3010x Pulse and Proximity Sensor Library
Adafruit MLX90614 Library
ArduinoJson (v6+)
WiFi (built-in for ESP32)
```

Install via Arduino Library Manager or PlatformIO.

---

## 4. Alternative: Web Serial API (Browser ↔ USB)

If the ESP32 is **plugged into the same computer** running the browser, you can use the **Web Serial API** to read sensor data directly from the browser without WiFi.

### Frontend Integration (`useSerialPort.ts`)

```typescript
// Hook to read sensor data from a serial-connected ESP32
export function useSerialPort(onData: (vitals: Partial<VitalInputs>) => void) {
  const [port, setPort] = useState<SerialPort | null>(null);
  const [connected, setConnected] = useState(false);

  const connect = async () => {
    try {
      const selectedPort = await navigator.serial.requestPort();
      await selectedPort.open({ baudRate: 115200 });
      setPort(selectedPort);
      setConnected(true);

      const reader = selectedPort.readable!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Read loop
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse newline-delimited JSON from ESP32
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line.trim());
            onData({
              heart_rate: parsed.heart_rate,
              spo2: parsed.spo2,
              temperature: parsed.temperature,
              resp_rate: parsed.resp_rate,
              map: parsed.map,
            });
          } catch { /* skip non-JSON lines */ }
        }
      }
    } catch (err) {
      console.error("Serial connection failed:", err);
    }
  };

  const disconnect = async () => {
    if (port) {
      await port.close();
      setPort(null);
      setConnected(false);
    }
  };

  return { connect, disconnect, connected };
}
```

For this mode, the ESP32 firmware should print JSON to Serial instead of using WiFi:

```cpp
// In loop(), replace the HTTP POST section with:
Serial.println(payload); // Sends JSON over USB serial
```

---

## 5. Frontend: Adding Hardware Mode to DemoSimulator

Add a new mode to the existing `DemoSimulator.tsx` to accept hardware input:

### Step 1: Add "Hardware" to the DemoMode type

```typescript
// In DemoSimulator.tsx, line ~41
type DemoMode = "instant" | "manual" | "simulated" | "hardware";
```

### Step 2: Add a Hardware connect button

```tsx
{/* Add after the existing mode selector */}
{mode === "hardware" && (
  <div className="flex gap-2">
    <Button
      className="flex-1"
      variant={connected ? "destructive" : "default"}
      onClick={connected ? disconnect : connect}
    >
      {connected ? "⏏ Disconnect" : "🔌 Connect Sensor"}
    </Button>
    {connected && (
      <Badge className="text-[10px] bg-emerald-500/20 text-emerald-400 border-emerald-500/40">
        LIVE
      </Badge>
    )}
  </div>
)}
```

### Step 3: Wire up the serial hook

```tsx
// Inside DemoSimulator component
const { connect, disconnect, connected } = useSerialPort((hwVitals) => {
  // Merge hardware readings with current vitals (preserving manual lab values)
  setVitals((prev) => ({ ...prev, ...hwVitals }));
});

// Auto-predict when hardware sends new data (already handled by instant mode useEffect)
```

### Step 4: Add the mode option to the Select dropdown

```tsx
<SelectItem value="hardware">🔌 Hardware Live</SelectItem>
```

---

## 6. Alternative: WiFi Direct (ESP32 → AWS, Display on Frontend)

If you don't want the frontend to be the middleman, the ESP32 can POST directly to `http://44.220.161.215:8000/predict` and the frontend can poll a `/latest` endpoint.

### Add to `api_server.py`:

```python
from datetime import datetime

# Store the latest hardware reading
latest_result = {}

@app.post("/hardware/ingest")
def hardware_ingest(vitals: VitalInputs):
    """Receive vitals from hardware, run prediction, store result."""
    global latest_result
    window = build_window(vitals)
    result = runtime.predict_one(window)
    # ... (same processing as /predict) ...
    latest_result = {
        "vitals": vitals.dict(),
        "prediction": processed_result,
        "timestamp": datetime.now().isoformat(),
    }
    return {"status": "ingested", "risk_score": processed_result["risk_score"]}

@app.get("/hardware/latest")
def hardware_latest():
    """Frontend polls this to display the latest hardware prediction."""
    return latest_result
```

### Frontend polling:

```typescript
// Poll every 5 seconds
useEffect(() => {
  if (mode !== "hardware") return;
  const interval = setInterval(async () => {
    const res = await fetch("/api/hardware/latest");
    if (res.ok) {
      const data = await res.json();
      if (data.prediction) setResult(data.prediction);
      if (data.vitals) setVitals(data.vitals);
    }
  }, 5000);
  return () => clearInterval(interval);
}, [mode]);
```

---

## 7. Sensor Mapping to API Schema

| API Field | Hardware Sensor | Auto? | Notes |
|---|---|---|---|
| `heart_rate` | MAX30102 | ✅ | Beats per minute from IR pulse detection |
| `spo2` | MAX30102 | ✅ | Ratio-of-ratios on RED/IR LEDs |
| `temperature` | MLX90614 | ✅ | Non-contact IR, reads forehead/wrist |
| `resp_rate` | Piezo belt / ADXL345 | ✅ | Chest expansion or accelerometer-based |
| `map` | HX710B + cuff | ⚠️ | Needs calibration; MAP = (SBP + 2×DBP) / 3 |
| `gcs_total` | — | ❌ | Clinician assessment (manual slider, 3–15) |
| `lactate` | — | ❌ | Lab test (manual input) |
| `wbc` | — | ❌ | Lab test (manual input) |
| `creatinine` | — | ❌ | Lab test (manual input) |
| `platelets` | — | ❌ | Lab test (manual input) |

---

## 8. Safety & Calibration

> **CAUTION:** This hardware integration is for **research and demonstration purposes only**. It is NOT a certified medical device. Do NOT use it for actual clinical decisions.

### Calibration Steps

1. **MAX30102 (HR/SpO₂)**: Compare readings against a certified pulse oximeter for 10 patients. Apply linear correction if needed.
2. **MLX90614 (Temp)**: Calibrate against a clinical thermometer. The sensor reads skin temperature which is typically 1-2°C lower than core body temperature — apply an offset.
3. **Respiration**: The piezo belt requires threshold tuning based on the patient's breathing depth. Adjust the `2048` threshold in firmware.
4. **Blood Pressure**: If using the analog cuff approach, calibrate against a sphygmomanometer. Consider using a validated USB BP monitor instead.

---

## 9. Bill of Materials (Complete Kit)

| Item | Qty | Cost (INR) | Cost (USD) |
|---|---|---|---|
| ESP32-WROOM-32 DevKit | 1 | ₹400 | $5 |
| MAX30102 Pulse Oximeter | 1 | ₹150 | $3 |
| MLX90614 IR Temperature | 1 | ₹350 | $5 |
| Piezo Respiration Belt | 1 | ₹200 | $3 |
| SSD1306 OLED Display | 1 | ₹150 | $2 |
| Jumper Wires + Breadboard | 1 set | ₹100 | $2 |
| USB-C Cable | 1 | ₹100 | $2 |
| **Total** | | **₹1,450** | **~$22** |

---

## 10. Quick Start Checklist

- [ ] **Order hardware** from the BOM above
- [ ] **Flash ESP32** with the firmware from Section 3
- [ ] **Update WiFi credentials** in the firmware (`ssid`, `password`)
- [ ] **Set server URL** — either direct to AWS (`http://44.220.161.215:8000/predict`) or through Vercel proxy
- [ ] **Open AWS Security Group** — port `8000` should already be open from the FastAPI setup
- [ ] **Add `"hardware"` mode** to `DemoSimulator.tsx` (Section 5)
- [ ] **Test with Serial Monitor** — verify JSON output before going wireless
- [ ] **Calibrate sensors** against clinical-grade equipment
- [ ] **Demo!** — Show real-time vitals flowing into the ML pipeline

---

## 11. Demo Script for Mentor Presentation

1. **Start with the stable baseline** — hardware reads normal vitals → green WATCH alert
2. **Simulate fever** — hold a warm object near MLX90614 → temperature spikes → AMBER alert
3. **Simulate tachycardia** — exercise briefly before placing finger on MAX30102 → HR rises → tripwires fire
4. **Show the hybrid** — hardware auto-fills HR, SpO₂, Temp, RR → manually slide Lactate to 5.0 → CRITICAL alert with full sepsis bundle actions
5. **Explain the architecture** — "Hardware reads real vitals, sends them over WiFi to our AWS backend running the LSTM + XGBoost ensemble with conformal prediction intervals and an adversarial Red Team safety agent"
