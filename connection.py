import curses
import time
from pymavlink import mavutil


def main(stdscr):
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.addstr("Press 's' to STABILIZE, 'q' to DISARM and quit...\n")

    # Connect to Pixhawk
    leo = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
    leo.wait_heartbeat()
    stdscr.addstr("Connected to Pixhawk\n")
    stdscr.refresh()

    # Arm motors and set manual mode
    leo.arducopter_arm()
    leo.motors_armed_wait()
    leo.set_mode_manual()
    time.sleep(2)

    # Test motors
    for i in range(1, 5):
        leo.set_servo(i, 1250)
        time.sleep(2)
        leo.set_servo(i, 1000)

    mode_id = leo.mode_mapping().get('STABILIZE')

    # Position estimation vars
    velocity = [0.0, 0.0, 0.0]  # vx, vy, vz in m/s
    position = [0.0, 0.0, 0.0]  # x, y, z in meters
    last_time = time.time()

    while True:
        key = stdscr.getch()
        if key != -1:
            try:
                char = chr(key)
                stdscr.addstr(f"Battery: {leo.battery.voltage}V\n")
                stdscr.addstr(f"Mode: {leo.flightmode}\n")
                stdscr.addstr(f"Key pressed: {char}\n")
                stdscr.refresh()

                if char == 's' and mode_id is not None:
                    leo.set_mode(mode_id)
                    stdscr.addstr("Switched to STABILIZE mode\n")
                elif char == 'q':
                    leo.set_mode_manual()
                    for i in range(1, 4):
                        leo.set_servo(i, 1000)
                    leo.arducopter_disarm()
                    stdscr.addstr("Motors disarmed. Exiting...\n")
                    break

            except ValueError:
                pass

        # Position estimation from accelerometers
        now = time.time()
        dt = now - last_time
        last_time = now

        imu = leo.recv_match(type='SCALED_IMU2', blocking=False)
        if imu:
            ax = imu.xacc * 9.81 / 1000  # Convert from mg to m/s^2
            ay = imu.yacc * 9.81 / 1000
            az = imu.zacc * 9.81 / 1000

            # Subtract gravity from Z (simple method)
            az -= 9.81

            # Integrate acceleration to velocity
            velocity[0] += ax * dt
            velocity[1] += ay * dt
            velocity[2] += az * dt

            # Integrate velocity to position
            position[0] += velocity[0] * dt
            position[1] += velocity[1] * dt
            position[2] += velocity[2] * dt

        stdscr.addstr(f"Estimated Position (m): X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}\n")
        stdscr.addstr("Loop tick...\n")
        stdscr.clear()
        stdscr.refresh()
        time.sleep(0.1)


if __name__ == "__main__":
    curses.wrapper(main)
