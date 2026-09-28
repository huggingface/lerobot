#include <Arduino.h>
#include <SPI.h>

// ---------------------------------------------------------------------------
// Timing & Sampling Configuration (8 Channels)
// ---------------------------------------------------------------------------
// 8 channels at 20,000 Hz each = 160,000 conversions/sec total aggregate
constexpr uint32_t NUM_CHANNELS = 8;
constexpr uint32_t CHANNEL_FS_HZ = 20000;
constexpr uint32_t AGGREGATE_SAMPLE_RATE_HZ = CHANNEL_FS_HZ * NUM_CHANNELS;
constexpr float SAMPLE_PERIOD_US = 1000000.0f / AGGREGATE_SAMPLE_RATE_HZ;

// Hardware Pins & SPI
constexpr uint8_t PIN_RST = 9;
constexpr uint8_t PIN_CS = 10;
constexpr uint32_t SPI_CLOCK_HZ = 20000000; // Increased to 20 MHz to handle 160 kHz throughput safely

// ADS8688 Commands & Registers
constexpr uint16_t CMD_NOOP     = 0x0000;
constexpr uint16_t CMD_AUTO_RST = 0xA000; // Reset auto sequence to channel 0

constexpr uint8_t REG_AUTO_SEQ_EN = 0x01; // Auto Sequence Enable register

// Range code 0x00 = +/- 2.5 * Vref = +/- 10.24 V
constexpr uint8_t RANGE_BIPOLAR_10V24 = 0x00;

// Buffering
constexpr uint32_t RING_SIZE = 16384; // Doubled ring buffer to prevent drops at higher aggregate rate
constexpr uint32_t RING_MASK = RING_SIZE - 1;
// Must be a multiple of 8 so packets never split an octet: 384 / 8 = 48 full frame sets
constexpr uint16_t TX_SAMPLES = 384;

volatile uint16_t ringBuffer[RING_SIZE];
volatile uint8_t ringPhase[RING_SIZE];
volatile uint32_t ringHead = 0;
volatile uint32_t ringTail = 0;
volatile uint32_t lostSamples = 0;
volatile uint8_t samplePhase = 0;

uint16_t txSamples[TX_SAMPLES];
uint32_t packetSequence = 0;

IntervalTimer sampleTimer;
SPISettings adsSPI(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE1);

// ---------------------------------------------------------------------------
// Binary Serialization Helpers (Little-Endian)
// ---------------------------------------------------------------------------
static inline void putU16LE(uint8_t *p, uint16_t v) {
    p[0] = v & 0xFF;
    p[1] = (v >> 8) & 0xFF;
}

static inline void putU32LE(uint8_t *p, uint32_t v) {
    p[0] = v & 0xFF;
    p[1] = (v >> 8) & 0xFF;
    p[2] = (v >> 16) & 0xFF;
    p[3] = (v >> 24) & 0xFF;
}

static inline uint16_t adsFrame(uint16_t command) {
    digitalWriteFast(PIN_CS, LOW);
    SPI.transfer((uint8_t)(command >> 8));
    SPI.transfer((uint8_t)(command & 0xFF));
    uint8_t msb = SPI.transfer(0x00);
    uint8_t lsb = SPI.transfer(0x00);
    digitalWriteFast(PIN_CS, HIGH);
    return ((uint16_t)msb << 8) | lsb;
}

static void adsWriteRegister(uint8_t address, uint8_t value) {
    uint16_t command = ((uint16_t)(address & 0x7F) << 9) | 0x0100 | value;
    digitalWriteFast(PIN_CS, LOW);
    SPI.transfer((uint8_t)(command >> 8));
    SPI.transfer((uint8_t)(command & 0xFF));
    SPI.transfer(0x00);
    digitalWriteFast(PIN_CS, HIGH);
}

// ---------------------------------------------------------------------------
// ISR: Fired at 160 kHz (Outputs CH0 through CH7 sequentially)
// ---------------------------------------------------------------------------
void sampleISR() {
    uint16_t sample = adsFrame(CMD_NOOP);
    uint32_t head = ringHead;
    uint32_t next = (head + 1) & RING_MASK;

    if (next == ringTail) {
        lostSamples++;
        return;
    }

    ringBuffer[head] = sample;
    ringPhase[head] = samplePhase;
    samplePhase = (samplePhase + 1) & 0x07;
    ringHead = next;
}

// ---------------------------------------------------------------------------
// Hardware Init
// ---------------------------------------------------------------------------
void initializeADS8688() {
    pinMode(PIN_CS, OUTPUT);
    digitalWriteFast(PIN_CS, HIGH);

    pinMode(PIN_RST, OUTPUT);
    digitalWriteFast(PIN_RST, HIGH);

    SPI.begin();
    SPI.beginTransaction(adsSPI);

    delay(10);

    // Hardware reset
    digitalWriteFast(PIN_RST, LOW);
    delayMicroseconds(2);
    digitalWriteFast(PIN_RST, HIGH);

    // Internal reference settling delay
    delay(20);

    // Configure Range (+/- 10.24 V) for all channels (Registers 0x05 to 0x0C)
    for (uint8_t reg = 0x05; reg <= 0x0C; reg++) {
        adsWriteRegister(reg, RANGE_BIPOLAR_10V24);
    }

    // Enable auto-scan on all 8 channels (binary 0xFF enables bits 0-7)
    adsWriteRegister(REG_AUTO_SEQ_EN, 0xFF);

    // Reset sequence to Channel 0 and enter Auto-Scan mode
    adsFrame(CMD_AUTO_RST);

    // Discard the first conversion latency frame
    adsFrame(CMD_NOOP);
}

void setup() {
    Serial.begin(2000000);
    initializeADS8688();

    if (!sampleTimer.begin(sampleISR, SAMPLE_PERIOD_US)) {
        while (true) {}
    }

    sampleTimer.priority(64);
}

void loop() {
    if (!Serial) {
        ringTail = ringHead;
        return;
    }

    uint32_t head = ringHead;
    uint32_t tail = ringTail;

    while (tail != head && ringPhase[tail] != 0) {
        tail = (tail + 1) & RING_MASK;
    }

    uint32_t available = (head - tail) & RING_MASK;

    if (available < TX_SAMPLES) return;

    for (uint16_t i = 0; i < TX_SAMPLES; i++) {
        txSamples[i] = ringBuffer[tail];
        tail = (tail + 1) & RING_MASK;
    }

    ringTail = tail;

    uint8_t header[14];
    header[0] = 'A';
    header[1] = '8';
    header[2] = '6';
    header[3] = '8';
    putU32LE(&header[4], packetSequence);
    putU16LE(&header[8], TX_SAMPLES);
    putU32LE(&header[10], lostSamples);

    Serial.write(header, sizeof(header));
    Serial.write((const uint8_t *)txSamples, TX_SAMPLES * sizeof(uint16_t));

    packetSequence++;
}