import psutil
import sys

def check_cpu_temp(threshold=76):
    temperature = psutil.sensors_temperatures().get('cpu_thermal')
    if temperature:
        current_temperature = temperature[0].current
        if current_temperature > threshold:
            print(f'CPU temperature is too high ({current_temperature}°C). Exiting program.')
            sys.exit(1)
        else:
            print('CPU temp is safe: '+str(current_temperature))
    else:
        print('Failed to retrieve CPU temperature information.')

#check_cpu_temp()
