// Philosophy here is simple: we only code "safety" related features, enabling expansion
// from the client side as needed without recompillation here. We should not be preventing
// expansion here, and should rather be enabling it without recompillation.
//
// Commands: 
//  "o[PIN]" open relay or off for high side switches.
//  "c[PIN]" close relay or on for high side switches.
//  "g[PIN]" get value for a pin. Mostly relevant to determine what a pin was set to.
//  "i[PIN]" Analog input for a pin.
//  "a[PIN] [VALUE]" Analog out for a pin via MCP4728
//  "m[PIN] [VALUE]" PWM modulated analog out for a pin.
//  "p[index] [mPIN] [iPIN] [setpoint] [k_prop] [k_int] [m_min]" Set a PI loop - all terms integers.
//      Setting gain to 0 turns of, and setting integral term to 0 resets the integral term.
//  "q" Quit this client. Can start another! (only 1 at a time)
//
//  For the PI loops, if a range of +/- 128 on  the input would mean full range on the output
//  this is a gain of 1. To take full advantage of 32 bit integers, we divide by 32 prior to
//  multiplying by the gain. For the integral term (in milli-seconds), we divide by 65536, so
//  that an offset of 1 DAC unit could be un-noticeable for 1 minute for the lowest setting.

// !!! For multiple ethernet connections, the while (client.connected())becomes while (client.available()),
// and each loop the "client" variable is over-written. as "new" isn't used, it is probably OK without
// memory leaks!

#include <Controllino.h>
#include <SPI.h>
#include <Ethernet.h>
#include <Adafruit_MCP4728.h>

// #define ANALOG_I2C_ADDR 40 // Not needed???

// Enter a MAC address and IP address for your controller below.
// The IP address will be dependent on your local network.
// gateway and subnet are optional:
byte mac[] = {0x50, 0xD7, 0x53, 0x00, 0xEB, 0xA7};    // MAC address (can be found on the Controllino)
IPAddress ip(172,16,8,200);                           // IP address (arbitrarily choosen)
EthernetServer server(23);                            // HTTP port
Adafruit_MCP4728 mcp;
int next_str_ix;  // The next index in the string we're passing (saves passing back and forth)

#define MIN_9V_PIN 3
#define MAX_9V_PIN 10

#define MAX_SERVOS 4
#define MAX_INTEGRAL 5120000 //An offset of 512 for 10000 milli-seconds
struct PIParams {
  int k_prop;
  int k_int;
  int m_pin;
  int i_pin;
  int setpoint;
  int m_min;
  long integral;
  unsigned long last_msec;
};

struct PIParams pi_params[MAX_SERVOS];
int s_ix=0; // Servo index
bool mcp_init=false; //Is the MCP initialised? This can be checked on start-up.

void setup() {
  // Ethernet initialization
  Serial.begin(9600);
  Ethernet.begin(mac, ip);

  // Start with all servos zeroed.
  for (int i=0; i<MAX_SERVOS;i++){
      pi_params[i].k_prop = 0;
      pi_params[i].integral = 0;
  }

  // Check for Ethernet hardware present
  if (Ethernet.hardwareStatus() == EthernetNoHardware) {
    Serial.println("Ethernet shield was not found.  Sorry, can't run without hardware. :(");
    while (true) {
      delay(1); // do nothing, no point running without Ethernet hardware
    }
  }
  if (Ethernet.linkStatus() == LinkOFF) {
    Serial.println("Ethernet cable is not connected.");
  }

  // Start listening for clients
  server.begin();

  Serial.print("Chat server address: ");
  Serial.println(Ethernet.localIP());
}

void loop() {
  char c = ' ';
  // Listen resquests
  EthernetClient client = server.available();
  
  if (client) {
    // Wait for the request
    String request = "";
    
    // Read the input string up until a newline character. Remaining characters will be left
    // in the buffer.
    while (client.available() && c != '\n') {
      c = client.read();
      request += c;
    }
    // Check we have a validly terminated message.
    if (c != '\n') {
      Serial.println("Erroneous Request (no newline):");
      Serial.println(request);
      return failure(client);
    }
    Serial.println("New Request:");
    Serial.print(request);

    // Parse the request ------------------------------------------------

    char c = request[0];
    // First, deal with any single character requests.
    if (c=='q'){
      client.stop();
      return;
    } 
    // Get the "pin" value (the first integer)
    next_str_ix = 1;
    int pin = get_value(request);
    if (pin == -1){
      Serial.println("Invalid integer pin number.");
      return;
    }
    // Now the commands which use 1 or more arguments
    if (c == 'o') {
      digitalWrite(pin, HIGH);
      return success(client);
    } else if (c == 'm'){
      // SAFETY: Limit this explicitly
      int value = get_value(request);
      if (value >= 0){
        Serial.println(String(value)); // Bugshooting
        if (pin >= MIN_9V_PIN && pin <= MAX_9V_PIN)
          if (value > int(255*9/24)) value = int(255*9/24);
        analogWrite(pin, value);
        return success(client);
      }
      else failure(client);
    } else if (c == 'c') {
      digitalWrite(pin, LOW);
      return success(client);
    } else if (c == 'g') {
      client.println(String(digitalRead(pin)));
    } else if (c == 'i') {
      client.println(String(analogRead(pin)));
    } else if (c == 'a') {
      // Analog ouput through the DAC.
      int value = get_value(request);
      Serial.println(String(value)); // Bugshooting
      /// Sanity check the output value.
      if (value < 0 || value > 4095) 
        return failure(client);
      // If we aren't initialised, initialise!
      if (!mcp_init){
        if (!mcp.begin()){
          return failure(client);
        } else mcp_init=true;
      }
      // Set the ADC value, catching an error if it occurs.
      if (mcp.setChannelValue(pin, value)){
        return success(client);
      } else {
        mcp_init=false;
        return failure(client);
      }
    } else if (c == 'p') {
      //p[index] [mPIN] [iPIN] [setpoint] [k_prop] [k_int] [m_min]
      pi_params[pin].m_pin = get_value(request);
      pi_params[pin].i_pin = get_value(request);
      pi_params[pin].setpoint = get_value(request);
      pi_params[pin].k_prop = get_value(request);
      pi_params[pin].k_int = get_value(request);
      pi_params[pin].m_min = get_value(request);
      // Sanity check and disable if a problem.
      if (pi_params[pin].m_pin < 0 || pi_params[pin].i_pin < 0 || pi_params[pin].setpoint < 0) pi_params[pin].k_prop=0;
      // Reset the integral if the integral term is reset.
      if (pi_params[pin].k_int == 0) pi_params[pin].integral=0;
      // Reset the time of last iteration (important if this is the first activation)
      pi_params[pin].last_msec = millis();
    } else {
      Serial.println("Invalid command character.");
      return failure(client);
    }
  }
  else {
    // Now lest run a servo loop!
    if (pi_params[s_ix].k_prop != 0){
      // Read the analog input and find the error.
      int error = analogRead(pi_params[s_ix].i_pin) - pi_params[s_ix].setpoint;
      
      // Compute the integral of the error. The integral has to be bounded.
      unsigned long now = millis();
      int dt = now - pi_params[s_ix].last_msec;
      pi_params[s_ix].last_msec = now;
      pi_params[s_ix].integral += dt * error;
      if (pi_params[s_ix].integral > MAX_INTEGRAL) pi_params[s_ix].integral = MAX_INTEGRAL;
      if (pi_params[s_ix].integral < -MAX_INTEGRAL) pi_params[s_ix].integral = -MAX_INTEGRAL;

      // Do the PI servo math
      int output = (pi_params[s_ix].m_min + 256)/2 + 
        pi_params[s_ix].k_prop * error / 32 + 
        pi_params[s_ix].k_int * pi_params[s_ix].integral/65536;
      
      // Make sure that we are within range for the putout, and write it.
      if (output < pi_params[s_ix].m_min) output = pi_params[s_ix].m_min;
      if (output > 255) output = 255;
      analogWrite(pi_params[s_ix].m_pin, output);
    }
    s_ix++; // move on to the next servo when we get the next chance. 
  }
}

int get_value(String request){
  int i = next_str_ix;
  String svalue;
  int value;
  char command = request[0];

  if (sizeof(request) > 1 && isdigit(request[i])) {
    // Read the value at the end of the command
    while (isdigit(request[i]) && i < request.length()) {
        svalue += request[i];
        i++;
    }
    value = svalue.toInt();

    // Initial command - check that the pin number isn't silly.
    // We can do more sophisticated checking later!
    if (next_str_ix == 1 && value > 80) value=-1;
   
    //Increment the place of the next number.
    next_str_ix += i;
    return value;
  } else {
    return -1; // No value
  }
}

// Failure and success as functions so behaviour is easy to read and easily changed.
void failure(EthernetClient client){
  client.println("0");
}
void success(EthernetClient client){
  client.println("1");
}