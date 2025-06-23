#!/usr/bin/env python3
"""
Node Selector Network Topology Implementation
N1 increments a value 100 times and sends to N2 or N3 randomly.
N2 sends ACK when receiving multiples of 3.
N3 sends ACK when receiving multiples of 5.
Uses UDP sockets.
"""

import socket
import threading
import time
import random

class ReceiverNode:
    def __init__(self, node_id, listen_ip, listen_port, sender_ip, sender_port, multiple):
        self.node_id = node_id
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.sender_ip = sender_ip
        self.sender_port = sender_port
        self.multiple = multiple  # 3 for N2, 5 for N3
        self.running = True
        self.socket = None
        
    def start_server(self):
        """Start UDP server to receive messages"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.listen_ip, self.listen_port))
        print(f"Node {self.node_id} (N{self.node_id}) listening on {self.listen_ip}:{self.listen_port}")
        print(f"Node {self.node_id} will ACK multiples of {self.multiple}")
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                value = int(data.decode())
                print(f"N{self.node_id} received value: {value} from {addr}")
                
                # Check if value is multiple of our number
                if value % self.multiple == 0:
                    print(f"N{self.node_id}: {value} is multiple of {self.multiple}, sending ACK")
                    self.send_ack(value)
                else:
                    print(f"N{self.node_id}: {value} is not multiple of {self.multiple}, no ACK")
                    
            except socket.error as e:
                if self.running:
                    print(f"N{self.node_id} socket error: {e}")
                    
    def send_ack(self, original_value):
        """Send ACK back to sender"""
        try:
            ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ack_message = f"ACK_{original_value}_from_N{self.node_id}"
            ack_socket.sendto(ack_message.encode(), (self.sender_ip, self.sender_port))
            ack_socket.close()
            print(f"N{self.node_id} sent ACK for value {original_value}")
        except Exception as e:
            print(f"N{self.node_id} failed to send ACK: {e}")
    
    def stop(self):
        """Stop the node"""
        self.running = False
        if self.socket:
            self.socket.close()

class SenderNode:
    def __init__(self, node_id, listen_ip, listen_port, n2_ip, n2_port, n3_ip, n3_port):
        self.node_id = node_id
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.n2_address = (n2_ip, n2_port)
        self.n3_address = (n3_ip, n3_port)
        self.running = True
        self.ack_socket = None
        
    def start_ack_listener(self):
        """Start listening for ACK messages"""
        self.ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ack_socket.bind((self.listen_ip, self.listen_port))
        print(f"N{self.node_id} ACK listener started on {self.listen_ip}:{self.listen_port}")
        
        while self.running:
            try:
                data, addr = self.ack_socket.recvfrom(1024)
                ack_message = data.decode()
                print(f"N{self.node_id} received: {ack_message} from {addr}")
            except socket.error as e:
                if self.running:
                    print(f"N{self.node_id} ACK listener error: {e}")
    
    def send_values(self):
        """Send 100 incremented values randomly to N2 or N3"""
        time.sleep(1)  # Wait for servers to start
        send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        for i in range(1, 101):
            # Randomly choose N2 or N3
            target = random.choice([self.n2_address, self.n3_address])
            target_name = "N2" if target == self.n2_address else "N3"
            
            try:
                send_socket.sendto(str(i).encode(), target)
                print(f"N{self.node_id} sent value {i} to {target_name} ({target[0]}:{target[1]})")
                time.sleep(0.1)  # Small delay to see the communication clearly
            except Exception as e:
                print(f"N{self.node_id} failed to send value {i}: {e}")
        
        send_socket.close()
        print(f"N{self.node_id} finished sending all 100 values")
        
        # Wait a bit for remaining ACKs
        time.sleep(2)
        self.running = False
    
    def stop(self):
        """Stop the sender node"""
        self.running = False
        if self.ack_socket:
            self.ack_socket.close()

def main():
    # Create nodes
    n1 = SenderNode(1, "127.0.0.1", 1234, "127.0.0.2", 2345, "127.0.0.3", 3456)
    n2 = ReceiverNode(2, "127.0.0.2", 2345, "127.0.0.1", 1234, 3)  # ACK multiples of 3
    n3 = ReceiverNode(3, "127.0.0.3", 3456, "127.0.0.1", 1234, 5)  # ACK multiples of 5
    
    # Start receiver nodes
    n2_thread = threading.Thread(target=n2.start_server)
    n2_thread.daemon = True
    n2_thread.start()
    
    n3_thread = threading.Thread(target=n3.start_server)
    n3_thread.daemon = True
    n3_thread.start()
    
    # Start N1's ACK listener
    n1_ack_thread = threading.Thread(target=n1.start_ack_listener)
    n1_ack_thread.daemon = True
    n1_ack_thread.start()
    
    # Start N1's sending process
    n1_send_thread = threading.Thread(target=n1.send_values)
    n1_send_thread.start()
    
    try:
        # Wait for N1 to finish
        n1_send_thread.join()
        
        # Wait a bit more for any remaining ACKs
        time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nStopping node selector simulation...")
    
    # Clean up
    n1.stop()
    n2.stop()
    n3.stop()
    
    print("Node selector simulation completed!")

if __name__ == "__main__":
    main()
