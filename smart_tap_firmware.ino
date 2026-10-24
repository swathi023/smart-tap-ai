#include <SPI.h>
#include <Wire.h>
#include <MFRC522.h>
#include <LiquidCrystal_I2C.h>

// =============== PIN CONFIGURATION ===============

#define RFID_SS_PIN 5
#define RFID_RST_PIN 4
#define LCD_SDA_PIN 21
#define LCD_SCL_PIN 22
#define LCD_ADDR 0x27

#define MAX_CARDS 5
#define CARD_COOLDOWN_MS 2000

MFRC522 rfid(RFID_SS_PIN, RFID_RST_PIN);
LiquidCrystal_I2C lcd(LCD_ADDR, 16, 2);

// =============== CARD TRIP DATA ===============
// Each card has its OWN tap-in location, tap-out location and fare.
// 1st tap of a card -> TAP IN at the start location
// 2nd tap of a card -> TAP OUT at the end location, fare is charged
// Cards work BOTH WAYS: trip 1 goes In -> Out, trip 2 goes Out -> In,
// trip 3 goes In -> Out again, and so on (same fare each way).
// EDIT THE TABLE BELOW with your real locations and fares.
// Keep names within 13 characters so they fit on the 16x2 LCD.
// Fares are DEMONSTRATION values, not official BMTC fares.

struct CardTrip {
  const char* inLocation;
  const char* outLocation;
  float fare;
};
CardTrip trips[MAX_CARDS] = {
  // inLocation      outLocation      fare (Rs.)
  {"Majestic",       "Bagaluru",      39.0},  // Card 1
  {"Indiranagar",    "Majestic",      35.0},  // Card 2
  {"Hebbal",         "Silk Board",    45.0},  // Card 3
  {"Majestic",       "Malleshwaram",  18.0},  // Card 4
  {"Shivajinagar",   "Yelahanka",     35.0}   // Card 5
};

// =============== CARD REGISTRATION ===============
// The first 5 different cards tapped are saved as Card 1..5 in order.
// (Resets when the ESP32 restarts.)

String cardUIDs[MAX_CARDS];
bool onBoard[MAX_CARDS] = {false, false, false, false, false};
bool reverseTrip[MAX_CARDS] = {false, false, false, false, false};  // false: In->Out, true: Out->In
int registeredCount = 0;

String lastUID = "";
unsigned long lastReadTime = 0;

// =============== LCD ===============

// Centre a text on a 16-character line (padded with spaces).
// Overwriting full lines instead of calling lcd.clear() removes the
// flicker, so the text looks steadier and easier to read.
String centre16(String text) {
  text = text.substring(0, 16);
  int left = (16 - text.length()) / 2;
  String out = "";
  for (int i = 0; i < left; i++) out += " ";
  out += text;
  while (out.length() < 16) out += " ";
  return out;
}

void showLCD(String line1, String line2) {
  lcd.setCursor(0, 0);
  lcd.print(centre16(line1));
  lcd.setCursor(0, 1);
  lcd.print(centre16(line2));
}

// =============== READ CARD UID ===============

String readUID() {
  String uid = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) uid += "0";
    uid += String(rfid.uid.uidByte[i], HEX);
  }
  uid.toUpperCase();
  return uid;
}

// =============== FIND REGISTERED CARD ===============

int findCard(String uid) {
  for (int i = 0; i < registeredCount; i++) {
    if (cardUIDs[i] == uid) return i;
  }
  return -1;
}

// =============== TAP IN / TAP OUT ===============

// Start and end location for the card's CURRENT trip (handles both directions).
const char* startOf(int c) {
  return reverseTrip[c] ? trips[c].outLocation : trips[c].inLocation;
}

const char* endOf(int c) {
  return reverseTrip[c] ? trips[c].inLocation : trips[c].outLocation;
}

void tapIn(int c, const String &uid) {
  onBoard[c] = true;

  Serial.println("TRIP_START");

  Serial.print("TAP,");
  Serial.print(uid);
  Serial.print(",");
  Serial.println(startOf(c));

  Serial.print("START,");
  Serial.println(startOf(c));

  Serial.print("Card ");
  Serial.print(c + 1);
  Serial.print(" TAP IN at: ");
  Serial.println(startOf(c));

  showLCD("TAP IN", startOf(c));
}

void tapOut(int c, const String &uid) {
  float fare = trips[c].fare;
  Serial.print("TAP,");
  Serial.print(uid);
  Serial.print(",");
  Serial.println(endOf(c));

  Serial.println("\n======= JOURNEY =======");
  Serial.print("Card: ");
  Serial.println(c + 1);
  Serial.print("From: ");
  Serial.println(startOf(c));
  Serial.print("To: ");
  Serial.println(endOf(c));
  Serial.print("Fare: Rs. ");

  Serial.println(fare, 2);
  Serial.println("=======================");

  // Machine-readable line for the Python bridge.
  Serial.print("FARE,");
  Serial.print(startOf(c));
  Serial.print(",");
  Serial.print(endOf(c));
  Serial.print(",");
  Serial.println(fare, 2);

  showLCD("FARE Rs." + String(fare, 0), endOf(c));

  onBoard[c] = false;                 // card is ready for the next trip
  reverseTrip[c] = !reverseTrip[c];   // next trip runs the opposite way

  Serial.println("Journey completed.");
}

// =============== RFID PROCESSING ===============

void processCard() {
  if (!rfid.PICC_IsNewCardPresent()) return;
  if (!rfid.PICC_ReadCardSerial()) return;

  String uid = readUID();

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  unsigned long now = millis();

  // Ignore repeated reads of the same card held on the reader.
  if (uid == lastUID && now - lastReadTime < CARD_COOLDOWN_MS) return;

  lastUID = uid;
  lastReadTime = now;

  Serial.print("Detected UID: ");
  Serial.println(uid);
  int c = findCard(uid);

  // Auto-register new cards (up to 5).
  if (c == -1) {
    if (registeredCount >= MAX_CARDS) {
      Serial.println("ERROR,Registration full");
      showLCD("Cards full", "5 cards saved");
      return;
    }

    c = registeredCount;
    cardUIDs[c] = uid;
    registeredCount++;

    Serial.print("REGISTERED,");
    Serial.print(uid);
    Serial.print(",Card ");
    Serial.println(c + 1);
  }

  // Same card: first tap = IN, second tap = OUT.
  if (!onBoard[c]) {
    tapIn(c, uid);
  } else {
    tapOut(c, uid);
  }
}

// =============== SETUP ===============

void setup() {
  Serial.begin(115200);

  Wire.begin(LCD_SDA_PIN, LCD_SCL_PIN);
  lcd.init();
  lcd.backlight();

  SPI.begin(18, 19, 23, RFID_SS_PIN);
  rfid.PCD_Init();

  delay(500);

  Serial.println("\nSMART TAP AI");
  Serial.println("Same card: tap once to TAP IN, again to TAP OUT.");

  showLCD("Smart Tap AI", "Tap your card");
}

// =============== MAIN LOOP ===============

void loop() {
  processCard();
}
