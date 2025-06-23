#!/usr/bin/env python3
"""
Main script to run all network topology simulations
Choose which simulation to run from the menu
"""

import subprocess
import sys
import os

def run_simulation(script_name):
    """Run a specific simulation script"""
    try:
        print(f"\n{'='*50}")
        print(f"Running {script_name}")
        print(f"{'='*50}")
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode != 0:
            print(f"Error running {script_name}")
        else:
            print(f"\n{script_name} completed successfully!")
            
    except FileNotFoundError:
        print(f"Error: {script_name} not found!")
    except KeyboardInterrupt:
        print(f"\n{script_name} interrupted by user")
    except Exception as e:
        print(f"Error running {script_name}: {e}")

def main():
    """Main menu for selecting simulations"""
    simulations = {
        '1': ('ring_communication.py', 'Ring Communication Topology'),
        '2': ('node_selector.py', 'Node Selector Topology'),
        '3': ('relay_nodes.py', 'Relay Nodes Topology'),
        '4': ('all', 'Run All Simulations')
    }
    
    while True:
        print("\n" + "="*60)
        print("NETWORK PROGRAMMING LAB - SOCKET SIMULATIONS")
        print("="*60)
        print("Select a simulation to run:")
        print()
        
        for key, (filename, description) in simulations.items():
            print(f"{key}. {description}")
        
        print("0. Exit")
        print()
        
        choice = input("Enter your choice (0-4): ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
        elif choice in simulations:
            filename, description = simulations[choice]
            
            if choice == '4':  # Run all
                print("\nRunning all simulations sequentially...")
                for sim_choice in ['1', '2', '3']:
                    sim_file, sim_desc = simulations[sim_choice]
                    print(f"\n--- Starting {sim_desc} ---")
                    run_simulation(sim_file)
                    input("\nPress Enter to continue to next simulation...")
            else:
                run_simulation(filename)
            
            input("\nPress Enter to return to menu...")
        else:
            print("Invalid choice! Please enter 0-4.")

if __name__ == "__main__":
    print("Network Programming Lab - Socket Programming Simulations")
    print("Make sure you have Wireshark running to capture traffic on loopback interface!")
    print("\nNote: Run each simulation in a separate terminal or use this menu.")
    main()
