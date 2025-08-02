#!/bin/bash

echo "Running noiseless-AZ-Angle.py with -l 10..."
python3 noiseless-AZ-Angle.py -l 10

echo "Running noiseless-AZ-Angle.py with -l 50..."
python3 noiseless-AZ-Angle.py -l 50

echo "Running noiseless-AZ-Amplitude.py with -l 50..."
python3 noiseless-AZ-Amplitud.py -l 50


echo "Done."
