"""

http://astroquery.readthedocs.io/en/latest/


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import socket
import sys
import time

import numpy as np
print('numpy.__version__:', np.__version__)
import matplotlib as mpl
import matplotlib.pyplot as plt
print('maplotlib.__version__:', mpl.__version__)

import astropy
print('astropy.__version__:', astropy.__version__)

from astropy import coordinates
from astropy import coordinates as coord
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.coordinates import SkyCoord


from astropy import units as u
from astropy.table import Table
from astropy.table import vstack

from astroquery.alma import Alma


def rd_config(config_file=None):

    # ConfigParser is renamed configparser in Python 3 and also 2.7.? or
    # maybe case insensitive MacOS can read either
    try:
        import configparser
    except:
        import ConfigParser as configparser

    print('__file__:', __file__)
    print('__name__: ', __name__)

    # could get this from __file__ search in a few default places
    config_file = 'explore_alma_archive.cfg'

    config = configparser.RawConfigParser()
    config.read(config_file)

    print(config)

    return config

def query_archive(wget=False, test=False):
    """

    """

    t0 = time.time()

    help_info = False
    if help_info:
        help(Alma)
        Alma.help()

    if test:

        ra = 180.0
        dec = -60.0
        radec = coordinates.SkyCoord(180.0, -60.0, unit=(u.degree, u.degree),
                                 frame='icrs')
        radius = 20.0
        obs_list = Alma.query_region(radec, radius*u.deg)
        print(len(obs_list))
        obs_list.info('stats')

        print(i, ra, dec, radius, len(obs_list), len(result))
        print('Elapsed time(secs):', time.time() - t0)


        sys.exit()


    i=-1
    radius = 5.1
    for ra in np.arange(5, 365, 10):
        for dec in np.arange(-85, 85, 10):
            i = i + 1

            print(i, ra, dec, radius)

            radec = coordinates.SkyCoord(ra, dec,
                                     unit=(u.degree, u.degree),
                                     frame='icrs')

            obs_list = Alma.query_region(radec, radius*u.deg)

            if i == 0:
                result = obs_list
            if i != 0:
                result = vstack([result, obs_list])

            print(i, ra, dec, radius, len(obs_list), len(result))
            print('Elapsed time(secs):', time.time() - t0)

    print(radius, len(result))

    result.info('stats')

    # now remove the duplicates caused by the overlapping circular search

    result.write('results.fits')

    # probably better to remove the dupes from a temporary file


def parse_args(version=None):
    """Parse the command line arguments

    Returns the args as an argparse.Namespace object

    """
    import sys
    import argparse

    description = '''Explore the ALMA Science'''

    epilog = "This is just an example of an epilog description"

    # use formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # so that --help lists the defaults

    parser = argparse.ArgumentParser(
        description=description, epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--version', action='store_const', default=False, const=True,
        help='show the version')

    parser.add_argument("--debug", action='store_true', default=False,
                        dest='debug', help="debug option; logging level DEBUG")

    parser.add_argument(
        '--ramax', action='store', default=24.0,
        help='Tile centre maximum  RA')

    parser.add_argument(
        '--read_csv', action='store_true', default=False,
        help='Read the csv file')

    parser.add_argument(
        '--convert_csv', action='store_true', default=False,
        help='Convert csv format file to fits format file')

    parser.add_argument(
        '--write_fits', action='store_true', default=False,
        help='Write fits format ALMA archive file')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """

    default activity could be to look for fits archive file and
    then csv file in that order and print out the help info

    """


    # could part of debug option
    print('numpy.__version__:', np.__version__)
    print('maplotlib.__version__:', mpl.__version__)
    print('astropy.__version__:', astropy.__version__)

    pid = os.getpid()
    print('Current working directory: %s' % (os.getcwd()))
    print('User: ', os.getenv('USER'))
    print('Hostname: ', os.getenv('HOSTNAME'))
    print('Host:     ', os.getenv('HOST'))
    print('Hostname: ', socket.gethostname())
    print()
    print('__file__: ', __file__)
    print('__name__: ', __name__)

    t0 = time.time()

    args = parse_args(version=None)
    config = rd_config()

    debug = args.debug
    read_csv = args.read_csv
    write_fits = args.write_fits
    read_fits = True

    rd_config(config_file=None)

    if read_csv:
        infile = 'alma_aq_data.csv'
        print('Read:', infile)

        table = Table.read(infile, format='ascii')
        table.info('stats')

    if write_fits:
        table.write('alma_aq_data.fits', overwrite=True)


    if read_fits:
        infile = 'alma_aq_data.fits'
        table = Table.read(infile)
        print('Elapsed time(secs):', time.time() - t0)
        table.info('stats')
        colnames = table.colnames
        ncolumns = len(table.colnames)
        print('Number of columns:', ncolumns)
        for i in range(0, ncolumns):
            data_min = np.min(np.array(table.field(i), dtype=object))
            data_max = np.max(np.array(table.field(i), dtype=object))
            # data_median = np.median(np.array(table.field(i), dtype=object))
            print(i+1, colnames[i], data_min, data_max)

        if debug:
            key=raw_input("Enter any key to continue: ")

    xdata = table['RA']
    ydata = table['Dec']

    print(np.min(xdata), np.max(xdata), len(xdata))
    print(np.min(ydata), np.max(ydata), len(ydata))
    ndata = len(xdata)
    plt.title(infile)
    plt.scatter(xdata, ydata, s=2, edgecolor='none', label=str(ndata))

    plt.xlim(0.0, 360.0)
    plt.ylim(-90.0, 60.0)
    plt.legend()

    plt.savefig(infile + '_fig1.png')
    plt.clf()

    # plt.show()

    # read and match to Veron+2010 quasar catalogue
    infile_Veron2010 = config.get("defaults", "infile_Veron2010")
    print('Reading:', infile_Veron2010)
    Veron2010 = Table.read(infile_Veron2010)
    if debug:
        Veron2010.info('stats')

    ra = Veron2010['RAJ2000']
    dec = Veron2010['DEJ2000']
    radec_veron = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')

    ra= table['RA']
    dec = table['Dec']
    radec_alma = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')

    # See http://docs.astropy.org/en/stable/coordinates/matchsep.html

    # match_coordinates_sky(matchcoord, catalogcoord,
    #                       nthneighbor=1, storekdtree='_kdtree_sky')
    # Finds the nearest on-sky matches of a coordinate or coordinates in
    # a set of catalog coordinates.
    idx, d2d, d3d = match_coordinates_sky(radec_alma, radec_veron)

    print()
    print('idxmatch range:', np.min(idx), np.max(idx))
    idx_unique, idx_unique_indices = np.unique(idx, return_index=True)
    print('Number of unique sources:', len(idx_unique), len(idx))
    print('idxmatch range:', np.min(idx_unique_indices),
          np.max(idx_unique_indices))
    print()

    # match_to_catalog_sky(catalogcoord[, nthneighbor])
    # Finds the nearest on-sky matches of this coordinate in a set
    # of catalog coordinates.
    idx, d2d, d3d = radec_veron.match_to_catalog_sky(radec_alma)

    print('idxmatch range:', np.min(idx), np.max(idx))
    idx_unique, idx_unique_indices = np.unique(idx, return_index=True)
    print('Number of unique sources:', len(idx_unique), len(idx))
    print('idxmatch range:', np.min(idx_unique_indices),
          np.max(idx_unique_indices))
    print()

    print(len(d2d), len(idx_unique), len(idx_unique_indices))

    veron_alma = Veron2010[idx_unique_indices]
    xdata = d2d[idx_unique_indices].arcsec
    itest = xdata  < 30.0

    veron_alma = veron_alma[itest]
    ndata = len(veron_alma)
    print('Number of Veron+2010 Quasars within 30" match radius:', ndata)
    print()

    xdata = xdata[itest]
    ndata = len(xdata)
    print('Number of Veron+2010 Quasars within 30" match radius:', ndata)

    plt.hist(xdata, bins=60, label=str(ndata))
    plt.grid()
    plt.legend()
    plt.title(infile)
    plt.xlabel('Separation (")')
    plt.ylabel('Frequency per bin')

    plotfile = infile + '_fig2.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()

    veron_alma.info('stats')

    plt.title(infile_Veron2010)
    xdata = veron_alma['z']
    ydata = veron_alma['Vmag']
    plt.scatter(xdata, ydata, s=4, edgecolor='none', label=str(ndata))

    plt.xlim(0.0, 7.5)
    plt.ylim(25.00, 12.0)
    plt.legend()

    plotfile = infile + '_fig3.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()


    xdata = veron_alma['F6cm']
    ndata = len(xdata)
    print(min(xdata), max(ydata))
    print(xdata)
    # get rid of NANs since they cause hist to crash
    xdata = np.log10(xdata[~np.isnan(xdata)])
    ndata = len(xdata)
    plt.hist(xdata, bins=50, label=str(ndata))
    plt.grid()
    plt.legend()
    plt.title(infile)
    plt.xlabel('Log10(F6cm (Jy))')
    plt.ylabel('Frequency per bin')

    plotfile = infile + '_fig4.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()


    xdata = veron_alma['F20cm']
    ndata = len(xdata)
    print(min(xdata), max(ydata))
    print(xdata)
    # get rid of NANs since they cause hist to crash
    xdata = np.log10(xdata[~np.isnan(xdata)])
    ndata = len(xdata)
    plt.hist(xdata, bins=50, label=str(ndata))
    plt.grid()
    plt.legend()
    plt.title(infile)
    plt.xlabel('Log10(F20cm (Jy))')
    plt.ylabel('Frequency per bin')

    plotfile = infile + '_fig5.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()

    ndata = len(veron_alma)
    veron_alma.info('stats')

    igood = 0
    for i in range(0, ndata):
        if (veron_alma[i]['z'] >= 0.3 and
            (np.isnan(veron_alma[i]['F6cm']) or veron_alma[i]['F6cm'] <= 0.1)
            and (np.isnan(veron_alma[i]['F20cm']) or
                 veron_alma[i]['F20cm'] <= 0.1)):
            igood = igood +1
            print(igood, i+1, veron_alma[i]['Name'],
                  veron_alma[i]['z'], veron_alma[i]['Vmag'],
                  veron_alma[i]['F6cm'], veron_alma[i]['F20cm'])


    # read and match to SDSS DR7QSO
