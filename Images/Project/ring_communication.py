#!/usr/bin/env python3
"""
Ring Communication Network Topology Implementation
Three computers communicate in a single direction creating a loop.
Each device increments the received value and sends it to the next device.
Communication ends when the payload reaches value '100'.
"""

import socket
import threading
import time
import sys

class RingNode:
    def __init__(self, node_id, listen_ip, listen_port, next_ip, next_port):
        self.node_id = node_id
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.next_ip = next_ip
        self.next_port = next_port
        self.running = True
        self.server_socket = None
        
    def start_server(self):
        """Start the server to listen for incoming connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.listen_ip, self.listen_port))
            self.server_socket.listen(1)
            print(f"Node {self.node_id} listening on {self.listen_ip}:{self.listen_port}")
            
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"Node {self.node_id} received connection from {addr}")
                    
                    # Receive data
                    data = client_socket.recv(1024)
                    if data:
                        value = int(data.decode())
                        print(f"Node {self.node_id} received value: {value}")
                        
                        client_socket.close()
                        
                        if value >= 100:
                            print(f"Node {self.node_id} received final value {value}. Communication complete!")
                            self.running = False
                            break
                        
                        # Increment and forward
                        new_value = value + 1
                        print(f"Node {self.node_id} forwarding value: {new_value}")
                        self.send_to_next(new_value)
                    
                except socket.error as e:
                    if self.running:
                        print(f"Node {self.node_id} socket error: {e}")
                        
        except Exception as e:
            print(f"Node {self.node_id} server error: {e}")
    
    def send_to_next(self, value):
        """Send value to the next node in the ring"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.next_ip, self.next_port))
            client_socket.send(str(value).encode())
            client_socket.close()
            print(f"Node {self.node_id} sent {value} to {self.next_ip}:{self.next_port}")
        except Exception as e:
            print(f"Node {self.node_id} failed to send to next node: {e}")
    
    def initiate_communication(self, initial_value=1):
        """Start the ring communication (only called by the first node)"""
        time.sleep(2)  # Wait for all servers to start
        print(f"Node {self.node_id} initiating communication with value: {initial_value}")
        self.send_to_next(initial_value)
    
    def stop(self):
        """Stop the node"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

def main():
    # Define the ring topology: Node1 -> Node2 -> Node3 -> Node1
    nodes = [
        RingNode(1, "127.0.0.1", 1234, "127.0.0.2", 2345),
        RingNode(2, "127.0.0.2", 2345, "127.0.0.3", 3456),
        RingNode(3, "127.0.0.3", 3456, "127.0.0.1", 1234)
    ]
    
    # Start server threads for each node
    threads = []
    for node in nodes:
        thread = threading.Thread(target=node.start_server)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Let servers start up
    time.sleep(1)
    
    # Node1 initiates the communication
    initiator_thread = threading.Thread(target=nodes[0].initiate_communication)
    initiator_thread.start()
    
    try:
        # Wait for communication to complete
        while any(node.running for node in nodes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping ring communication...")
    
    # Clean up
    for node in nodes:
        node.stop()
    
    print("Ring communication simulation completed!")

if __name__ == "__main__":
    main()
