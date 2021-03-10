import torch
import models
import os
import subprocess
import datetime
import logging
import sys

# Experiment parameters
USE_GPU = True
NUM_JOINTS = 17
