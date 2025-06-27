## MUST BE RUN IN BASE ENVIRONMENT (check >conda env list). Otherwise wont import _sardine.

import numpy as np
import json
import sys
import time
import logging
from threading import Thread, Event

sys.path.insert(1, "/home/heimdallr/Documents/rtc-example/python/")
# sys.path.insert(1, "/home/heimdallr/Documents/rtc-example/python/baldr/")
# sys.path.insert(1, "/home/heimdallr/Documents/rtc-example/python/sardine/")
# sys.path.insert(1, "/home/heimdallr/Documents/rtc-example/")
from baldr import _baldr as ba
from baldr import sardine as sa

sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
import bmc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
DM_COMMAND_SIZE = 140
CONFIG_FILE = "/home/asg/Progs/repos/asgard-alignment/config_files/dm_shared_memory_config.json"


class DMController(Thread):
    def __init__(
        self, dm_index, shared_memory_url, DM_serial_number, update_interval=0.1
    ):
        super().__init__()
        self.dm_index = dm_index
        self.shared_memory_url = shared_memory_url
        self.DM_serial_number = DM_serial_number
        self.update_interval = update_interval
        self.running = Event()
        self.running.set()
        self.last_command = None
        self.shm = None
        self.dm = None
        self.init_shared_memory()
        self.init_dm()

    def init_shared_memory(self):
        """Initialize shared memory."""
        try:
            self.shm = sa.from_url(np.ndarray, self.shared_memory_url)
            logging.info(
                f"DM {self.dm_index}: Connected to shared memory at {self.shared_memory_url}"
            )
        except Exception as e:
            logging.error(
                f"DM {self.dm_index}: Failed to connect to shared memory: {e}"
            )
            self.shm = None

    def init_dm(self):
        """Initialize the DM."""
        try:
            self.dm = bmc.BmcDm()  # Create DM object
            dm_err_flag = self.dm.open_dm(self.DM_serial_number)  # Open DM connection
            if dm_err_flag != 0:
                raise RuntimeError(
                    f"DM {self.dm_index}: Error opening DM with serial {self.DM_serial_number}"
                )
            logging.info(
                f"DM {self.dm_index}: Successfully initialized DM with serial {self.DM_serial_number}"
            )
        except Exception as e:
            logging.error(f"DM {self.dm_index}: Failed to initialize DM: {e}")
            self.dm = None

    def read_shared_memory(self):
        """Read command from shared memory."""
        if self.shm is not None:
            try:
                return np.copy(self.shm)  # Read the shared memory
            except Exception as e:
                logging.error(f"DM {self.dm_index}: Error reading shared memory: {e}")
        return None

    def send_command_to_dm(self, command):
        """Send command to the DM."""
        if self.dm is not None:
            try:
                self.dm.send_data(command)  # Send the command to the DM
                logging.info(f"DM {self.dm_index}: Command sent successfully.")
            except Exception as e:
                logging.error(f"DM {self.dm_index}: Failed to send command to DM: {e}")
        else:
            logging.warning(
                f"DM {self.dm_index}: DM not initialized; command not sent."
            )

    def run(self):
        """Thread loop to monitor shared memory and update the DM."""
        while self.running.is_set():
            command = self.read_shared_memory()
            if command is not None and not np.array_equal(command, self.last_command):
                self.send_command_to_dm(command)
                self.last_command = command
            time.sleep(self.update_interval)

    def stop(self):
        """Stop the thread."""
        self.running.clear()
        if self.dm is not None:
            self.dm.close_dm()  # Cleanly close the DM connection
            logging.info(f"DM {self.dm_index}: Closed DM connection.")


class DMControllerManager:
    def __init__(self, shared_memory_urls, DM_serial_numbers, update_interval=0.1):
        self.dm_count = len(shared_memory_urls)
        self.controllers = [
            DMController(
                dm_index=i,
                shared_memory_url=url,
                DM_serial_number=serial,
                update_interval=update_interval,
            )
            for i, (url, serial) in enumerate(
                zip(shared_memory_urls, DM_serial_numbers)
            )
        ]

    def start(self):
        """Start all DM controller threads."""
        logging.info("Starting DM controllers...")
        for controller in self.controllers:
            controller.start()

    def stop(self):
        """Stop all DM controller threads."""
        logging.info("Stopping DM controllers...")
        for controller in self.controllers:
            controller.stop()
        for controller in self.controllers:
            controller.join()


def setup_shared_memory(num_dms):
    """Set up shared memory regions and write their URLs to a JSON file."""
    commands_dict = {}
    commands_url_dict = {}

    for beam in range(num_dms):
        commands_dict[beam] = sa.region.host.open_or_create(
            f"beam{beam}_commands", shape=[DM_COMMAND_SIZE], dtype=np.double
        )
        commands_url_dict[beam] = sa.url_of(commands_dict[beam])
        logging.info(
            f"DM {beam}: Shared memory created at {commands_url_dict[beam].geturl()}."
        )

    # Save shared memory URLs to JSON
    save_shared_memory_urls(commands_url_dict)

    return commands_url_dict


def save_shared_memory_urls(commands_url_dict):
    """Save shared memory URLs to a JSON file."""
    config_data = {
        "commands_urls": {beam: url.geturl() for beam, url in commands_url_dict.items()}
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)
    logging.info(f"Shared memory URLs saved to {CONFIG_FILE}.")


def read_shared_memory_config(config_file):
    """Read shared memory configuration from JSON file."""
    with open(config_file, "r") as f:
        config_data = json.load(f)
    logging.info(f"Loaded shared memory configuration from {config_file}.")
    return config_data


if __name__ == "__main__":
    # Number of DMs
    NUM_DMS = 4

    DM_serial_numbers = ["17DW019#113", "17DW019#053", "17DW019#093", "17DW019#122"]

    # Step 1: Set up shared memory
    commands_url_dict = setup_shared_memory(NUM_DMS)

    # Step 2: Create DMControllerManager and start controllers
    shared_memory_urls = [commands_url_dict[beam].geturl() for beam in range(NUM_DMS)]
    manager = DMControllerManager(
        shared_memory_urls=shared_memory_urls,
        DM_serial_numbers=DM_serial_numbers,
        update_interval=0.1,
    )

    try:
        manager.start()
        logging.info("DM controllers are running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("Stopping DM controllers.")
        manager.stop()


"""

# example to open and edit shared memory somewhere else 
import numpy as np
from baldr import _baldr as ba
from baldr import sardine as sa
import json
#ff = "/home/asg/Progs/repos/asgard-alignment/config_files/dm_shared_memory_config.json"
ff = "dm_shared_memory_config.json"
with open(ff, "r") as f:
    config_data = json.load(f)
    for beam in [0,1,2,3]:
    
        url_dict[beam] = config_data['commands_urls'][f'{beam}']

        dm_dict[beam] = sa.from_url(np.ndarray, url_dict[beam])

## DOES NOT WORK IF YOU ASSIGN VALUE (a=X). But relative shifts work (a+=)




"""
