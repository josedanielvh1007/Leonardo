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

    while True:
        key = stdscr.getch()
        if key != -1:
            try:
                char = chr(key)
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

        stdscr.addstr("Loop tick...\n")
        stdscr.refresh()
        time.sleep(0.5)


if __name__ == "__main__":
    curses.wrapper(main)
