# Controllino code for Asgard

## Common functionality
Communication is with the Multi-Device Server (MDS) via ethernet. Serial debugging through USB can provide additional information. All commands are indexed by a single character, with the command set common to all controllino instances. All code is asynchronous and functionality is executed every loop, subject to no new commands.

## Libraries
Uses Adafruit_MCP4728 and Controllino libraries, installed using the arduino library manager.

## Controllino 0: Main controller
The code is found in "controllino_server". This controls temperature, power and the piezo motors.

## Controllino 1: Stepper controller
Code found in "stepper_server". Controls stepper motors with some custom electronics. 
