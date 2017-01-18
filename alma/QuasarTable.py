"""
match also to the archive listing:

alma_aq_data_20161117.fits


"""
from __future__ import print_function, unicode_literals

# system functions
import sys

# 3rd functions
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from astropy.table import Table, hstack

# local functions
sys.path.append('/home/rgm/soft/python/lib/')
from librgm.plotid import plotid
from librgm.xmatch import xmatch_cat
from librgm.xmatch import xmatch_cat_join

infile = 'QuasarTable.csv'

# Read csv file into a Pandas DataFrame and then convert to an Astropy table
dataframe = pd.read_csv(infile, usecols=list(range(18)),
                        nrows=16)

print(dataframe)

# convert to Astropy table
table1 = Table.from_pandas(dataframe)
table1.info()
table1.info('stats')


print('Name of  firt object in the table:', table1[0]['NAME'])
print('Name of last object in the table:', table1[-1]['NAME'])


infile2 = 'NonCalibratorSource_xVeron2010_sources20162911_109.fits'

table2 = Table.read(infile2)
table2.info('stats')

checkplot=False
idxmatch, dr = xmatch_cat(
   table1=table1,
   table2=table2,
   colnames_radec1=['RA', 'DEC'],
   colnames_radec2=['ALPHA_J2000', 'DELTA_J2000'],
   checkplot=checkplot)

print('len(table1):', len(table1))
print('len(table2):', len(table2))
print('len(idxmatch)', len(idxmatch))

result = xmatch_cat_join(table1=table1,
                         table2=table2,
                         idxmatch=idxmatch,
                         dr=dr, radius_arcsec=10.0)
result.info('stats')

print('result[0]', result[0])

for irow, row in enumerate(result):
    print(irow+1, row['NAME'], row['FLUX_AUTO'], row['FLUXERR_AUTO'])

print('Now match sources back to observation log')
key=raw_input("Enter any key to continue: ")


# now match to alma_aq_data_20161117.fits

infile3 = 'alma_aq_data_20161117.fits'

table3 = Table.read(infile3)
table3.info('stats')


checkplot = True
idxmatch, dr = xmatch_cat(
   table1=table1,
   table2=table3,
   colnames_radec1=['RA', 'DEC'],
   colnames_radec2=['RA', 'Dec'],
   checkplot=checkplot)


result = xmatch_cat_join(table1=table1,
                         table2=table3,
                         idxmatch=idxmatch,
                         dr=dr, radius_arcsec=10.0)


result.info('stats')

result.write('tmp.fits')
result.write('tmp.csv')
