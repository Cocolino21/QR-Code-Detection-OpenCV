#!/usr/bin/env python3
"""
Relay Nodes Network Topology Implementation
Sender transmits 100 packets to D1, D2, or D3 randomly.
Each packet must pass through intermediate nodes to reach destination.
Topology: Sender -> D1 -> D2 -> D3
"""

import socket
import threading
import time
import random
import struct

class RelayNode:
    def __init__(self, node_id, node_ip, listen_port, next_hop_ip=None, next_hop_port=None):
        self.node_id = node_id
        self.node_ip = node_ip
        self.listen_port = listen_port
        self.next_hop_ip = next_hop_ip
        self.next_hop_port = next_hop_port
        self.running = True
        self.server_socket = None
        
    def start_server(self):
        """Start server to listen for incoming packets"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.node_ip, self.listen_port))
            self.server_socket.listen(5)
            print(f"{self.node_id} listening on {self.node_ip}:{self.listen_port}")
            
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    
                    # Receive payload: target_ip (4 bytes) + value (4 bytes)
                    data = client_socket.recv(8)
                    if len(data) == 8:
                        # Unpack the data: target IP (as 4 bytes) + value (as 4 bytes)
                        target_ip_bytes = data[:4]
                        value_bytes = data[4:]
                        
                        # Convert IP bytes back to string
                        target_ip = '.'.join(str(b) for b in target_ip_bytes)
                        value = struct.unpack('>I', value_bytes)[0]
                        
                        print(f"{self.node_id} received packet: target={target_ip}, value={value}")
                        
                        # Check if this packet is for us
                        if target_ip == self.node_ip:
                            print(f"{self.node_id} - Packet reached destination! Value: {value}")
                        else:
                            # Forward to next hop
                            if self.next_hop_ip and self.next_hop_port:
                                print(f"{self.node_id} forwarding packet to {self.next_hop_ip}:{self.next_hop_port}")
                                self.forward_packet(data)
                            else:
                                print(f"{self.node_id} - No next hop configured, packet dropped")
                    
                    client_socket.close()
                    
                except socket.error as e:
                    if self.running:
                        print(f"{self.node_id} socket error: {e}")
                        
        except Exception as e:
            print(f"{self.node_id} server error: {e}")
    
    def forward_packet(self, packet_data):
        """Forward packet to next hop"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.next_hop_ip, self.next_hop_port))
            client_socket.send(packet_data)
            client_socket.close()
        except Exception as e:
            print(f"{self.node_id} failed to forward packet: {e}")
    
    def stop(self):
        """Stop the relay node"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

class SenderNode:
    def __init__(self, sender_ip, d1_ip, d1_port):
        self.sender_ip = sender_ip
        self.d1_ip = d1_ip
        self.d1_port = d1_port
        self.destinations = ["127.0.0.2", "127.0.0.3", "127.0.0.4"]  # D1, D2, D3
        
    def create_packet(self, target_ip, value):
        """Create packet with target IP and value"""
        # Convert target IP to 4 bytes
        ip_parts = target_ip.split('.')
        target_ip_bytes = bytes([int(part) for part in ip_parts])
        
        # Convert value to 4 bytes (big-endian unsigned int)
        value_bytes = struct.pack('>I', value)
        
        return target_ip_bytes + value_bytes
    
    def send_packets(self):
        """Send 100 packets to random destinations"""
        time.sleep(2)  # Wait for all servers to start
        
        for i in range(1, 101):
            # Randomly select destination
            target_ip = random.choice(self.destinations)
            
            # Create packet
            packet = self.create_packet(target_ip, i)
            
            try:
                # Send to D1 (first hop)
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((self.d1_ip, self.d1_port))
                client_socket.send(packet)
                client_socket.close()
                
                print(f"Sender sent packet {i} targeting {target_ip}")
                time.sleep(0.1)  # Small delay for visibility
                
            except Exception as e:
                print(f"Sender failed to send packet {i}: {e}")
        
        print("Sender finished sending all 100 packets")

def main():
    # Create the relay topology: Sender -> D1 -> D2 -> D3
    d1 = RelayNode("D1", "127.0.0.2", 2345, "127.0.0.3", 3456)  # D1 forwards to D2
    d2 = RelayNode("D2", "127.0.0.3", 3456, "127.0.0.4", 4567)  # D2 forwards to D3
    d3 = RelayNode("D3", "127.0.0.4", 4567)  # D3 is final destination (no next hop)
    
    sender = SenderNode("127.0.0.1", "127.0.0.2", 2345)
    
    # Start relay nodes
    nodes = [d1, d2, d3]
    threads = []
    
    for node in nodes:
        thread = threading.Thread(target=node.start_server)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Start sender
    sender_thread = threading.Thread(target=sender.send_packets)
    sender_thread.start()
    
    try:
        # Wait for sender to finish
        sender_thread.join()
        
        # Wait a bit more for packets to propagate
        time.sleep(3)
        
    except KeyboardInterrupt:
        print("\nStopping relay nodes simulation...")
    
    # Clean up
    for node in nodes:
        node.stop()
    
    print("Relay nodes simulation completed!")

if __name__ == "__main__":
    main()
