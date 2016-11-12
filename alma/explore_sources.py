"""

http://docs.astropy.org/en/stable/coordinates/matchsep.html#matching-catalogs


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# standard library functions
import inspect
import os
import sys
import time


# 3rd party functions
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from table_stats import *


# local functions
import explore_alma_archive as eaa

if __name__ == "__main__":
    # hundreds of lines of code

    import argparse
    import ConfigParser

    from astropy.table import Table
    from astropy.io import ascii
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import match_coordinates_sky
    from astropy import units as u


    infile = 'sources300916.txt'
    infile = '/home/rgm/soft/obelics/bitbucket/results/sources031116.txt'
    infile = '/home/rgm/soft/obelics/bitbucket/results/sources111116.txt'


    # infile = 'sources300916_test2.txt'
    # infile = 'test.txt'
    # infile = 'sources300916_new.txt'


    print('Reading: ', infile)

    alma = ascii.read(infile, guess=False)
    alma.write('sources.fits', overwrite=True)

    print('Numer of rows:', len(alma))
    print()
    print('Number of columns:', len(alma.colnames))
    print('Number of data fields in row 1:', len(alma[0]))
    print(alma[0])


    # table = ascii.read(infile, header_start=0, data_start=1)
    # print()
    # print(table[0])


    # table = Table.read(infile)

    table_stats('sources.fits', debug=True)
    alma.info()
    alma.info('stats')

def sources_stats(data):
    """


    """




    return

    limitflux = False
    if limitflux:
        itest = (alma['FLUX_AUTO'] > 0.003) & (alma['FLUX_AUTO'] < 0.030) & \
                (alma['FLUX_AUTO']/alma['FLUXERR_AUTO'] > 5.0) & \
                (alma['FLUX_AUTO']/alma['FLUXERR_AUTO'] < 50.0)

        alma = alma[itest]

    xdata = alma['FLUX_AUTO']
    ydata = alma['FLUXERR_AUTO']
    itest = (xdata > 0) & (xdata < 1.0)
    xdata = xdata[itest]
    ydata = ydata[itest]
    plt.suptitle(infile)
    plt.xlabel('FLUX_AUTO')
    plt.ylabel('FLUXERR_AUTO')
    plt.plot(xdata, ydata, '.', label=len(xdata))
    plt.legend(fontsize='medium')
    plt.show()
    plt.close()



    xdata = alma['FLUX_AUTO']
    ydata = alma['FLUXERR_AUTO']
    itest = (xdata > 0) & (xdata < 1.0)
    xdata = xdata[itest]
    ydata = ydata[itest]
    ydata = xdata / ydata
    itest = (ydata < 100)
    xdata = xdata[itest]
    ydata = ydata[itest]
    plt.suptitle(infile)
    plt.xlabel('FLUX_AUTO (Jy)')
    plt.ylabel('S/N')
    plt.plot(xdata, ydata, '.', label=len(xdata))
    plt.legend(fontsize='medium')
    plt.show()
    plt.close()


    xdata = alma['FLUX_AUTO']
    ydata = alma['FLUXERR_AUTO']
    itest = (xdata > 0) & (xdata < 1.0)
    xdata = xdata[itest]
    ydata = ydata[itest]
    ydata = xdata / ydata
    itest = (ydata < 100)
    xdata = xdata[itest]
    ydata = ydata[itest]
    itest = (xdata > 0.0001) & (xdata < 0.010)
    xdata = xdata[itest] * 1000.0
    ydata = ydata[itest]
    plt.suptitle(infile)
    plt.xlabel('FLUX_AUTO (mJy)')
    plt.ylabel('S/N')
    plt.plot(xdata, ydata, '.', label=len(xdata))
    plt.legend(fontsize='medium')
    plt.show()
    plt.close()


    xdata = alma['ALPHA_J2000']
    ydata = alma['DELTA_J2000']

    plt.plot(xdata, ydata, '.')
    # plt.show()
    plt.close()

    ra = alma['ALPHA_J2000']
    dec = alma['DELTA_J2000']

    radec_alma = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')

    # newtable = Table([ra, dec], names=('ra', 'dec'))
    # outfile = 'tmp.vot'
    # newtable.write(outfile, table_id='alma_data', format='votable')


    config_file_eaa = 'explore_alma_archive.cfg'
    config = eaa.rd_config(config_file=config_file_eaa)

    infile_Veron2010 = config.get("defaults", "infile_Veron2010")
    print('Reading:', infile_Veron2010)
    Veron2010 = Table.read(infile_Veron2010)
    Veron2010.info('stats')

    ra = Veron2010['RAJ2000']
    dec = Veron2010['DEJ2000']
    radec_veron = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')


    print('Using: match_to_catalog_sky')
    idxmatch, d2d, d3d = radec_alma.match_to_catalog_sky(radec_veron)
    print('len(idxmatch):', len(idxmatch), min(idxmatch), max(idxmatch))
    print('len(radec_alma):', len(radec_alma))
    print('len(radec_veron):', len(radec_veron))

    veron_xmatch_alma = Veron2010[idxmatch]
    radec_veron_xmatch_alma = radec_veron[idxmatch]

    alma.info('stats')
    veron_xmatch_alma.info('stats')

    separation = radec_alma.separation(radec_veron_xmatch_alma)
    median_separation = np.median(separation).arcsec
    print('Median separation (arc seconds):', median_separation)

    dra, ddec = radec_alma.spherical_offsets_to(radec_veron_xmatch_alma)
    print('dRa range:', np.min(dra), np.max(dra))
    print('dDEC range:', np.min(ddec), np.max(ddec))

    xdata = dra.arcsec
    ydata = ddec.arcsec

    range = 4.0
    itest = (abs(xdata) < range) & (abs(ydata) < range)
    xdata = xdata[itest]
    ydata = ydata[itest]

    plt.plot(xdata, ydata, '.', label=len(xdata))
    plt.axis()
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend()
    plt.show()
