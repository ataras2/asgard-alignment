import zmq


def send_and_receive(socket, msg):
    print(f"Sending message: {msg}")
    socket.send_json(msg)
    try:
        reply = socket.recv_json()
        print(f"Received reply: {reply}")
        return reply
    except zmq.Again as e:
        print(f"Error: No response from server within timeout period. {e}")
    return None


def compare_reply(reply, expected_reply):
    # compare everything recuresively except the time field
    reply_copy = reply.copy()
    expected_reply_copy = expected_reply.copy()
    if "time" in reply_copy["reply"]:
        del reply_copy["reply"]["time"]
    if "time" in expected_reply_copy["reply"]:
        del expected_reply_copy["reply"]["time"]
    if reply_copy == expected_reply_copy:
        print("Test passed: Reply matches expected reply.")
    else:
        print("Test failed: Reply does not match expected reply.")
        print(f"Expected: {expected_reply_copy}")
        print(f"Received: {reply_copy}")
