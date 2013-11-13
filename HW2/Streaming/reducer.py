#!/usr/bin/env python

import sys
import csv

current_bin = None
current_count = 0
c = csv.writer(open('test.csv', "wb"),delimiter=',',quoting=csv.QUOTE_MINIMAL)

for line in sys.stdin:
    line = line.strip()
    bbin,count = line.split("\t",1)
    count = int(count)

    if current_bin == bbin:
        current_count += count
    else:
         if current_bin:
		print '%s,%s' % (current_bin,current_count)

         current_count = count
         current_bin = bbin

if current_bin == bbin:
	print '%s,%s' % (current_bin,current_count)