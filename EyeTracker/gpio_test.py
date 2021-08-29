import RPi.GPIO as GPIO
import time

pin_audio = 9
pin_red = 3
pin_yellow = 27
pin_green = 11
pin_sw = 26
pin_btn = 2

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_audio, GPIO.OUT, initial=0)
GPIO.setup(pin_red, GPIO.OUT, initial=0)
GPIO.setup(pin_green, GPIO.OUT, initial=1)
GPIO.setup(pin_yellow, GPIO.OUT, initial=0)
GPIO.setup(pin_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(pin_sw, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.output(pin_red, GPIO.LOW)
GPIO.output(pin_green, GPIO.LOW)
GPIO.output(pin_yellow, GPIO.LOW)
GPIO.output(pin_audio, GPIO.LOW)

time.sleep(5)

GPIO.output(pin_red, GPIO.HIGH)
GPIO.output(pin_green, GPIO.HIGH)
GPIO.output(pin_yellow, GPIO.HIGH)
GPIO.output(pin_audio, GPIO.HIGH)

time.sleep(5)

GPIO.cleanup()