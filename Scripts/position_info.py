from pymavlink import mavutil
import time

CONNECTION_PORT = '/dev/serial/by-id/usb-ArduPilot_fmuv2_31003D001751383435323530-if00'
CONNECTION_RATE = 115200

delayTime = 1

def main():
    leo = mavutil.mavlink_connection(CONNECTION_PORT, baud=CONNECTION_RATE)
    leo.wait_heartbeat()
    print('Connected (system %u component %u)' % (leo.target_system, leo.target_component))
    print(leo.wait_heartbeat())
    print(leo.recv_match(type="ATTITUDE"))
    leo.mav.message_interval_send(
        mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,
        10000000
    )

    response = leo.recv_match(type='COMMAND_ACK', blocking=False)
    if response and response.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL and response.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print('Battery status command accepted')
    else:
        print('Battery status command failed')

    # Send attitude message interval command directly
    leo.mav.message_interval_send(
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        15000
    )

    response = leo.recv_match(type='COMMAND_ACK', blocking=False)
    if response and response.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL and response.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print('Attitude command accepted')
    else:
        print('Attitude command failed')
    
    roll = pitch = yaw = 0.0
    battery = 0

    sTime = time.time()
    tTime = 0

    roll = pitch = yaw = 0.0
    battery = 0

    msg_pos = leo.recv_match(type='ATTITUDE', blocking=False)
    if msg_pos:
        roll = msg_pos.roll
        pitch = msg_pos.pitch
        yaw = msg_pos.yaw

    msg_btry = leo.recv_match(type='BATTERY_STATUS', blocking=False)
    if msg_btry:
        battery = msg_btry.battery_remaining

    while True:
        print(leo.recv_match(type="ATTITUDE", blocking=False))
        time.sleep(0.5)


if __name__ == '__main__':
    main()
