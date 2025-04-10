from pymavlink import mavutil

connection = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
connection.wait_heartbeat()
print("Heartbeat from system (system_id %u component_id %u)" % (connection.target_system, connection.target_component))