import argparse
import zmq

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ZeroMQ Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--timeout", type=int, default=5000, help="Response timeout in milliseconds")
    args = parser.parse_args()

    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a socket to communicate with the server
    socket = context.socket(zmq.REQ)

    # Set the receive timeout
    socket.setsockopt(zmq.RCVTIMEO, args.timeout)

    # Connect to the server
    server_address = f"tcp://{args.host}:{args.port}"
    socket.connect(server_address)

    while True:
        # Prompt the user for a message
        message = input("Enter message to send to the server (or 'exit' to quit): ")

        if message.lower() == "exit":
            print("Exiting...")
            break

        # Send the message to the server
        print(f"Sending message to server: {message}")
        socket.send_string(message)

        try:
            # Wait for a response from the server
            response = socket.recv_string()
            print(f"Received response from server: {response}")
        except zmq.Again as e:
            print("No response from server, request timed out")

if __name__ == "__main__":
    main()