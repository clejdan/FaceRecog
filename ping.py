import zmq
import numpy as np
import cv2
import time
import os

BIND_ADDR = "tcp://0.0.0.0:5555"

def main():
    OUTPUT_DIR = "ReceivedImages"
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND_ADDR)
    print(f"[B] REP listening on {BIND_ADDR} ...")
    
    while True:
        jpg_bytes = sock.recv()  # blocks until message arrives
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("[B] Got bytes but failed to decode.")
            sock.send_string("DECODE_FAIL")
            continue
        
        h, w = img.shape[:2]
        print(f"[B] Received image {w}x{h}, bytes={len(jpg_bytes)} @ {time.strftime('%H:%M:%S')}")
        
        # Save the received image
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{OUTPUT_DIR}/img_{int(time.time())}.jpg"
        cv2.imwrite(filename, img)
        print(f"[B] Saved to {filename}")
        
        sock.send_string("OK")  # ACK back to sender

if __name__ == "__main__":
    main()