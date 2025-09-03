// Philosophy here is simple: we only code "safety" related features, which includes
// maximising the voltage of the PWM output.
//
// Commands: 
//  "e[MOTOR]" Enable a motor
//  "d[MOTOR]" Disable a motor.
//  "r[STEPS]" Relative motor move for all enabled motors
//  "h[MOTOR]" Home a motor, but moving backwards until the home sensor is found.
//  "w[MOTOR]" Find the position of a motor.
//  "z[MOTOR]" Find if the motor is homed.
//  "s[MOTOR] [STEPS]" Move to a fixed number of steps from zero. 
//  "?" Ping
//  "q" Quit this client. Can start another! (only 1 at a time)
//
//  Philisophy: 
//    For all moves, we only change the step once, but update the postition for all motors.
//    For all moves other than a relative move, we move 1 motor at a time.

#include <SPI.h>
#include <NativeEthernet.h>
#define BAD_INT 32767

// Enter a MAC address and IP address for your controller below.
// The IP address will be dependent on your local network.
// gateway and subnet are optional:
byte mac[] = {0x50, 0xD7, 0x53, 0x00, 0xEB, 0x9E};    // MAC address (Make this up??? Just subtracted 1 from the controllino)
IPAddress ip(192,168,100,12);                           // IP address (arbitrarily choosen)
EthernetServer server(23);                            // HTTP port
int next_str_ix;  // The next index in the string we're passing (saves passing back and forth)

#define MAX_MOTORS 12
#define STEP_PIN 27
#define DIR_PIN 28
#define MS1 30
#define MS2 31
#define MS3 32
int current_pos[MAX_MOTORS] = {0,0,0,0,0,0,0,0,0,0,0,0};
int target_pos[MAX_MOTORS] = {0,0,0,0,0,0,0,0,0,0,0,0};
int zero_pins[MAX_MOTORS] = {20,21,22,23,16,17,18,19,3,2,1,0};
int enable_pins[MAX_MOTORS] = {33,34,35,36,37,38,39,40,41,13,14,15};
bool looking_for_home[MAX_MOTORS]={false,false,false,false,false,false,false,false,false,false,false,false};
bool found_home[MAX_MOTORS]=      {false,false,false,false,false,false,false,false,false,false,false,false};
bool enable_motors[MAX_MOTORS]=    {false,false,false,false,false,false,false,false,false,false,false,false};
EthernetClient clients[8];
int stepit=0;

void setup() {
  // Ethernet initialization
  Serial.begin(9600);
  Ethernet.begin(mac, ip);
  for (int i=0;i<MAX_MOTORS;i++){
    pinMode(zero_pins[i],INPUT);
    pinMode(enable_pins[i],OUTPUT);
    digitalWrite(enable_pins[i],HIGH);
  }
  pinMode(STEP_PIN,OUTPUT);
  pinMode(DIR_PIN,OUTPUT);
  pinMode(MS1,OUTPUT);
  pinMode(MS2,OUTPUT);
  pinMode(MS3,OUTPUT);
  digitalWrite(MS1,0);
  digitalWrite(MS2,0);
  digitalWrite(MS3,0);
  

  // Check for Ethernet hardware present
  if (Ethernet.hardwareStatus() == EthernetNoHardware) {
    Serial.println("Ethernet hardware was not found.  Sorry, can't run without hardware. :(");
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
  // Listen resquests. !!! This fails when a client is already connected
  EthernetClient client = server.accept();
  if (client) {
    for (byte i = 0; i < 8; i++) {
      if (!clients[i]) {
        Serial.println("New client connected.");
        // Once we "accept", the client is no longer tracked by EthernetServer
        // so we must store it into our list of clients
        clients[i] = client;
        break;
      }
    }
  }
  
  // check for incoming data from all clients
  for (byte i = 0; i < 8; i++) {
    if  (clients[i] && clients[i].available() > 0) { 
      // We would wait for the request here. However, we know packets are very small, so
      // try no waiting.
      String request = "";
  
      // Read the input string up until a newline character. Remaining characters will be left
      // in the buffer.
      while (clients[i].available() && c != '\n') {
        c = clients[i].read();
        request += c;
      }
      // Check we have a validly terminated message.
      if (c != '\n') {
        Serial.println("Erroneous Request (no newline):");
        Serial.println(request);
        return failure(clients[i]);
      }
      Serial.print("New Request from client ");
      Serial.print(i);
      Serial.print(": ");
      Serial.print(request);
  
      // Parse the request ------------------------------------------------
  
      char c = request[0];
      // First, deal with any single character requests.
      if (c=='q'){
        clients[i].stop();
        return;
      } 
      if (c=='?'){
        return success(clients[i]);
      }
      // Get the "pin" value (the first integer)
      next_str_ix = 1;
      int pin = get_value(request);
      if (pin == BAD_INT){
        Serial.println("Invalid integer pin number.");
        return failure(clients[i]);
      } else Serial.println(pin);
      // Now the commands which use 1 or more arguments
      if (c == 'r'){
        // Add this movement to all active motors. Here "pin" is the 
        // distance to move XXX
        for (int i=0;i<MAX_MOTORS;i++){
          if (enable_motors[i]) target_pos[i] += pin;
        }
        return success(clients[i]);
      }
      else if (c == 'e'){
        enable_motors[pin]=true;
        digitalWrite(enable_pins[pin], LOW);
        return success(clients[i]);
      }
      else if (c == 'd'){
        enable_motors[pin]=false;
        digitalWrite(enable_pins[pin], HIGH);
        return success(clients[i]);
      }
      else if (c == 's') {
        int value = get_value(request);
        // Check we have the right motor.
        if (pin >= MAX_MOTORS) return failure(clients[i]);
        // Make sure this motor ie enabled
        if (!enable_motors[pin]){
          return failure(clients[i]);
        }
        target_pos[pin] = value;
        return success(clients[i]);
      } else if (c=='h'){
        if (pin >= MAX_MOTORS) return failure(clients[i]);
        looking_for_home[pin] = true;
        found_home[pin] = false;
        target_pos[pin] = -24000;
        return success(clients[i]);
      } else if (c=='w'){
        if (pin >= MAX_MOTORS) return failure(clients[i]);
        else {
           clients[i].println(String(current_pos[pin]));
           return;
        }
      } else if (c=='z'){ //Has zero been found?
        if (pin >= MAX_MOTORS) return failure(clients[i]); //!!! This just returns "0" which isn't great.
        else clients[i].println(String(int(found_home[pin])));
      } else {
        Serial.println("Invalid command character.");
        return failure(clients[i]);
      }
    }
  }

  // stop any clients which disconnect
  for (byte i = 0; i < 8; i++) {
    if (clients[i] && !clients[i].connected()) {
      Serial.print("Client disconnected. Number: ");
      Serial.println(i);
      clients[i].stop();
    }
  }

  // Now lets move the motors!!!
  for (int i=0;i<MAX_MOTORS;i++){
    if (looking_for_home[i]){
      if (digitalRead(zero_pins[i])){
        target_pos[i]=0;
        current_pos[i]=0;
        looking_for_home[i]=false;
        found_home[i]=true;
      }
    }
    if ((target_pos[i] == current_pos[i]) || !enable_motors[i]){
      continue;
     // Do nothing.
    } else {
      // Print the current steps.
      Serial.print(i);
      Serial.print(" ");
      Serial.print(target_pos[i]);
      Serial.print(" ");
      Serial.print(current_pos[i]);
      Serial.print(" ");
      Serial.print(stepit);
      Serial.println();
      if (stepit){
        if (target_pos[i] > current_pos[i]) {
          digitalWrite(DIR_PIN,HIGH); //!!! Need to do this only for 1 of the motors.XXX
          digitalWrite(STEP_PIN,HIGH);
          current_pos[i] += 1;
        }
        if (target_pos[i] < current_pos[i]) {
          digitalWrite(DIR_PIN,LOW);
          digitalWrite(STEP_PIN,HIGH);
          current_pos[i] -= 1;
        }
      } else {
        digitalWrite(STEP_PIN,LOW);
      }
      stepit = (stepit + 1) % 2;
    } 
  }
  delay(5); // If not looking for the home sensor.
}

int get_value(String request){
  int i = next_str_ix;
  String svalue;
  int value;
  char command = request[0];

  if (sizeof(request) > 1 && (isdigit(request[i])) || (request[i] == '-')) {
    // Read the value at the end of the command
    while ((isdigit(request[i]) || (request[i] == '-')) && i < request.length()) {
        svalue += request[i];
        i++;
    }
    value = svalue.toInt();

    // Initial command - check that the pin number isn't silly.
    // We can do more sophisticated checking later!
    if (next_str_ix == 1 && value > 80) value=BAD_INT;
   
    //Increment the place of the next number.
    next_str_ix += i;
    return value;
  } else {
    return BAD_INT; // No value
  }
}

// Failure and success as functions so behaviour is easy to read and easily changed.
void failure(EthernetClient client){
  client.println("F");
  // As Native Ethernet doesn't work... let's flush
}
void success(EthernetClient client){
  client.println("S");
  // As Native Ethernet doesn't work... let's flush
}
