from pymavlink import mavutil
from time import sleep

leo = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600) # Setup connection and wait for heartbeat
leo.wait_heartbeat()
print("Heartbeat from system (system_id %u component_id %u)" % (leo.target_system, leo.target_component))

leo.arducopter_arm() 
leo.motors_armed_wait()
leo.set_mode_manual()
sleep(2)
for i in range (1,4): # Tests connection by starting/stoping each motor
    leo.set_servo(i,1250)
    sleep(2)
    leo.set_servo(i,1000)
mode_id = leo.mode_mapping().get('STABILIZE') # Sets mode to stabilize
leo.set_mode(mode_id)