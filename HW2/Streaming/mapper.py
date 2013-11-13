#!/usr/bin/env python

import sys
import math

for line in sys.stdin:
    nums = line.split()
    lowerX = math.floor(float(nums[0])*10)/10
    upperX = math.ceil(float(nums[0])*10)/10
    lowerY = math.floor(float(nums[1])*10)/10
    upperY = math.ceil(float(nums[1])*10)/10
    print '%s,%s,%s,%s\t%s' % (lowerX,upperX,lowerY,upperY,1)
