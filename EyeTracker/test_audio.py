import RPi.GPIO as GPIO
import time

pin_audio = 19
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_audio, GPIO.OUT, initial=0)


GPIO.output(pin_audio, GPIO.LOW)
time.sleep(6)
print('done')
GPIO.output(pin_audio, GPIO.HIGH)
time.sleep(6)
GPIO.output(pin_audio, GPIO.LOW)
time.sleep(6)
GPIO.cleanup()