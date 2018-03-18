import subprocess
import os

output = subprocess.check_output("th driver.lua -dataDir ./eval -modelPath ./model.th7", shell=True)

print(output.decode('UTF-8'))