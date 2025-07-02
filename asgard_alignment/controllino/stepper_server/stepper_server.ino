// Philosophy here is simple: we only code "safety" related features, which includes
// maximising the voltage of the PWM output.
//
// Commands: 
//  "r[MOTOR] [STEPS]" Relative motor move
//  "s[MOTOR] [STEPS]" Move to a fixed number of steps from zero.
//  "h[MOTOR]" Home a motor, but moving backwards until the home sensor is found.
//  "w[MOTOR]" Find the position of a motor.
//  "z[MOTOR]" Find if the motor is homed.
//  "?" Ping
//  "q" Quit this client. Can start another! (only 1 at a time)
//
//  For the PI loops, if a range of +/- 128 on  the input would mean full range on the output
//  this is a gain of 1. To take full advantage of 32 bit integers, we divide by 32 prior to
//  multiplying by the gain. For the integral term (in milli-seconds), we divide by 65536, so
//  that an offset of 1 DAC unit could be un-noticeable for 1 minute for the lowest setting.

// !!! For multiple ethernet connections, the while (client.connected())becomes while (client.available()),
// and each loop the "client" variable is over-written. as "new" isn't used, it is probably OK without
// memory leaks!

#include "Controllino.h"
#include <SPI.h>
#include <Ethernet.h>

// #define ANALOG_I2C_ADDR 40 // Not needed???

// Enter a MAC address and IP address for your controller below.
// The IP address will be dependent on your local network.
// gateway and subnet are optional:
byte mac[] = {0x50, 0xD7, 0x53, 0x00, 0xEB, 0x9F};    // MAC address (can be found on the Controllino)
IPAddress ip(192,168,100,12);                           // IP address (arbitrarily choosen)
EthernetServer server(23);                            // HTTP port
int next_str_ix;  // The next index in the string we're passing (saves passing back and forth)

#define MAX_MOTORS 3
#define START_PIN 2
#define MOTOR_HIGH 100 
#define MOTOR_AB MOTOR_HIGH*5/7
int current_pos[3] = {0,0,0};
int target_pos[3] = {0,0,0};
int zero_pins[3] = {54,55,56};
const unsigned int STEPS[8][4] = {{MOTOR_AB,MOTOR_AB,0,0},
  {0,MOTOR_HIGH,0,0},
  {0,MOTOR_AB,MOTOR_AB,0},
  {0,0,MOTOR_HIGH,0},
  {0,0,MOTOR_AB,MOTOR_AB},
  {0,0,0,MOTOR_HIGH},
  {MOTOR_AB,0,0,MOTOR_AB},
  {MOTOR_HIGH,0,0,0}};
unsigned int *this_step=STEPS[0];
bool looking_for_home[3]={false,false,false};
bool found_home[3]={false,false,false};

void setup() {
  // Ethernet initialization
  Serial.begin(9600);
  Ethernet.begin(mac, ip);
  for (int i=0;i<MAX_MOTORS;i++){
    pinMode(zero_pins[i],INPUT);
  }

  // Try to move the first motor 10000 steps
  //looking_for_home[1] = true;
  //looking_for_home[1] = false;
  //target_pos[1] = -10000;

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
    if (c=='?'){
      return success(client);
    }
    // Get the "pin" value (the first integer)
    next_str_ix = 1;
    int pin = get_value(request);
    if (pin == -1){
      Serial.println("Invalid integer pin number.");
      return;
    }
    // Now the commands which use 1 or more arguments
    if (c == 'r'){
      int value = get_value(request);
      // Check we have the right motor.
      if (pin >= MAX_MOTORS) return failure(client);
      target_pos[pin] += value;
      return success(client);
    }
    else if (c == 's') {
      int value = get_value(request);
      // Check we have the right motor.
      if (pin >= MAX_MOTORS) return failure(client);
      target_pos[pin] = value;
      return success(client);
    } else if (c=='h'){
      if (pin >= MAX_MOTORS) return failure(client);
      looking_for_home[pin] = true;
      found_home[pin] = false;
      target_pos[pin] = -24000;
      return success(client);
    } else if (c=='w'){
      if (pin >= MAX_MOTORS) return failure(client);
      else client.println(String(current_pos[pin]));
    } else if (c=='z'){ //Has zero been found?
      if (pin >= MAX_MOTORS) return failure(client); //!!! This just returns "0" which isn't great.
      else client.println(String(int(found_home[pin])));
    } else {
      Serial.println("Invalid command character.");
      return failure(client);
    }
  }
  else {
    // Now lets move the motor if requested!
    for (int i=0;i<MAX_MOTORS;i++){
      if (looking_for_home[i]){
        if (digitalRead(zero_pins[i])){
          target_pos[i]=0;
          current_pos[i]=0;
          looking_for_home[i]=false;
          found_home[i]=true;
        }
      }
      if (target_pos[i] == current_pos[i]){
        analogWrite(i*4 + START_PIN, 0);
        analogWrite(i*4 + START_PIN+1, 0); 
        analogWrite(i*4 + START_PIN+2, 0); 
        analogWrite(i*4 + START_PIN+3, 0); 
      } else {
        if (target_pos[i] > current_pos[i]) current_pos[i] += 1;
        if (target_pos[i] < current_pos[i]) current_pos[i] -= 1;
        int mod_result = current_pos[i] % 8;
        if (mod_result < 0) mod_result += 8;
        this_step = STEPS[mod_result];
        analogWrite(i*4 + START_PIN, this_step[0]);
        analogWrite(i*4 + START_PIN+1, this_step[1]); 
        analogWrite(i*4 + START_PIN+2, this_step[2]); 
        analogWrite(i*4 + START_PIN+3, this_step[3]);
      } 
    }
    delay(5); // If not looking for the home sensor.
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
  client.println("F");
}
void success(EthernetClient client){
  client.println("S");
}