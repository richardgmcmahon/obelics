"""

astroquery did not work
http://astroquery.readthedocs.io/en/latest/

so used alma_aq_data.csv via wget



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

#from astropy import __version__
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

# from astroquery.alma import Alma

def wget_archivelog():
    """
    get a full list of everything ALMA has observed as csv file

    How can I get a full list of everything ALMA has observed?
    https://help.almascience.org/snoopi/index.php?/Knowledgebase/Article/View/308

    wget -O alma_aq_data.csv --no-check-certificate "http://almascience.org/aq/?result_view=raw&science_observations=&download=true&format=CSV"

    """

    import os
    import time

    now = time.localtime(time.time())
    datestamp = time.strftime("%Y%m%d",now)

    output = 'alma_aq_data_' + datestamp + '.csv'
    wget_options = ' --no-check-certificate '

    url_prefix = 'http://almascience.org/'

    url_suffix = 'aq/?result_view=raw&science_observations=&download=true&format=CSV'

    wget_url = url_prefix + url_suffix

    cmd = 'wget ' + wget_options + ' "' + wget_url + '" ' + ' -O ' + output

    print('Executing:', cmd)
    os.system(cmd)


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
    if config_file is None:
        config_file = 'explore_alma_archive.cfg'

    config = configparser.RawConfigParser()
    config.read(config_file)

    print(config)

    return config

def ScanIntent(data, debug=False):

    return


def Integration(data, debug=False):
    """


    """

    xdata = data

    ndata = len(xdata)

    print('ndata:', ndata)

    data_min = min(xdata)
    data_max = max(xdata)

    print('min:', data_min)
    argsort_data = np.argsort(xdata)
    print(' 0.1%:', xdata[argsort_data[int(ndata*0.001)]])
    print(' 0.5%:', xdata[argsort_data[int(ndata*0.005)]])
    print(' 1.0%:', xdata[argsort_data[int(ndata*0.01)]])
    print(' 5.0%:', xdata[argsort_data[int(ndata*0.05)]])
    print('50.0%:', xdata[argsort_data[int(ndata*0.50)]])
    print('95.0%:', xdata[argsort_data[int(ndata*0.95)]])
    print('99.0%:', xdata[argsort_data[int(ndata*0.99)]])
    print('99.5%:', xdata[argsort_data[int(ndata*0.995)]])
    print('99.9%:', xdata[argsort_data[int(ndata*0.999)]])
    print('max:', data_max)

    label = str(ndata)
    # + '\n' + str(data_min) + '\n' + str(data_max)
    plt.hist(xdata, bins=100, label=label)
    plt.grid()
    plt.legend()
    plt.xlabel('Integration')
    plt.ylabel('Frequency per bin')

    plotfile = infile + '_Integration_fig1.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()

    itest = xdata < 3600
    xdata = xdata[itest]

    ndata = len(xdata)
    label = str(ndata)
    # + '\n' + str(data_min) + '\n' + str(data_max)
    plt.hist(xdata, bins=100, label=label)
    plt.grid()
    plt.legend()
    plt.xlabel('Integration')
    plt.ylabel('Frequency per bin')

    plotfile = infile + '_Integration_fig2.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()

    return


def Largest_angular_scale(data):



    return


def Frequency_support(data, debug=False, infile=None):
    """

    decode the frequency metadata

    The frequency info is stored in a string with a arbitrary number of
    segments of the form:
         [330.55..330.60GHz,0.06kHz,null]

    """
    # help(data)


    print('__file__:', __file__)
    print('__name__: ', __name__)

    print('data.shape:', data.shape)
    print('data.dtype:', data.dtype)

    print('Some info for first and last data record')
    print(data[0])
    print(data[0].shape)
    print(data[0].dtype)
    print(len(data[0]))
    print()

    print(data[-1])
    print(data[-1].shape)
    print(data[-1].dtype)
    print(len(data[-1]))
    print()

    irec = -1
    nvalue_max = -1
    nvalues = []
    freq_observed = []
    for record in data:
        irec = irec + 1
        if debug:
            print(irec, record)

        values = record.split('[')
        if debug:
            print(irec, record.dtype, record.shape,
                  len(record), len(values),
                  len(values[0]), len(values[-1]))

        nvalue = len(values)
        nvalues.append(nvalue)
        nvalue_max = max(nvalue_max, nvalue)

        # parse value[1:] since value[0] is the data before first '['
        for cell in values[1:]:
            if debug:
                print('cell:', cell)
            freq_cells = cell.split(',')
            if debug:
                print(len(freq_cells))
                print('freq_cells[0]:', freq_cells[0])
            freq = freq_cells[0].split('..')
            itest = freq[1].find('Hz')
            if debug:
                print('Units:', freq[1][itest-1:])
            if freq[1][itest-1:itest] != 'G':
                sys.exit()
            if debug:
                print(len(freq))
                print(freq[0], freq[1])
            # assume always GHz for now
            freq_observed.append(float(freq[0]))

        values = record.split(']')
        if debug:
            print('Number of values:', len(values),
                  len(values[0]), len(values[-1]))

    print('Maximum Frequency range:', nvalue_max)

    ndata = len(freq_observed)
    label = str(ndata)
    xdata = freq_observed
    plt.hist(xdata, bins=100, label=label)
    plt.grid()
    plt.legend()
    plt.xlabel('Observed Frequency (Ghz)')
    plt.ylabel('Frequency per bin')
    plt.title(infile)

    plotfile = infile + '_Frequency_support_fig1.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()



    xdata = 300.0/np.asarray(xdata)
    plt.hist(xdata, bins=100, label=label)
    plt.grid()
    plt.legend()
    # plt.xlabel('Observed Frequency (Ghz)')
    plt.ylabel('Frequency per bin')
    plt.title(infile)

    # axes = plt.gca()
    # xdata = np.asarray(axes.get_xlim())
    # ydata = np.asarray(axes.get_ylim())
    # print(xdata)
    # print(ydata)

    # plt.twiny()
    plt.xlabel("Observed Wavelength (mm)")
    # plt.plot(300.0/xdata, ydata)

    plotfile = infile + '_Frequency_support_fig2.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.clf()

    return

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

    parser.add_argument("--verbose", action='store_true', default=False,
                        dest='verbose', help="verbose option; logging level DEBUG")

    parser.add_argument(
        '--read_fits', action='store_true', default=True,
        help='Read the fits format file')

    parser.add_argument(
        '--read_csv', action='store_true', default=False,
        help='Read the csv format file')

    parser.add_argument(
        '--convert_csv', action='store_true', default=False,
        help='Convert csv format file to fits format file')

    parser.add_argument(
        '--write_fits', action='store_true', default=False,
        help='Write fits format ALMA archive file after ')

    parser.add_argument(
        '--wget_csvfile', action='store_true', default=False,
        help='run wget to get new copy of observation log in csv format')


    args = parser.parse_args()

    return args


def plot_radec(ra, dec, title=None, suffix=None, infile=None,
               filelabel=None):
    """

    """

    xdata = ra
    ydata = dec

    print('xdata range:', np.min(xdata), np.max(xdata), len(xdata))
    print('ydata range:', np.min(ydata), np.max(ydata), len(ydata))
    ndata = len(xdata)
    plt.title(infile)
    plt.scatter(xdata, ydata, s=2, edgecolor='none', label=str(ndata))

    plt.xlim(0.0, 360.0)
    plt.ylim(-90.0, 60.0)
    plt.legend()

    if title is not None: plt.title(title)
    plotfile = 'radec.png'
    if filelabel is None:
        filelabel = ''
    if filelabel is not None:
        filelabel = '_' + filelabel
    if infile is not None: plotfile = infile + '_radec' + filelabel + '.png'
    plt.savefig(plotfile)
    plt.clf()

    return


def drqso_analysis(radec_alma=None, dr='dr7'):
    """


    """
    if dr == 'dr12':
        infile = config.get("defaults", "infile_dr12qso")
    if dr == 'dr7':
        infile = config.get("defaults", "infile_dr7qso")

    print('Reading:', infile)
    drqso = Table.read(infile)
    debug = True
    if debug:
        drqso.info('stats')

    ra = drqso['RA']
    dec = drqso['DEC']
    radec_drqso = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')


    print('Using: match_coordinates_sky')
    idx, d2d, d3d = match_coordinates_sky(radec_alma, radec_drqso)

    print('len(idx):', len(idx))
    print('idxmatch range:', np.min(idx), np.max(idx))
    idx_unique, idx_unique_indices = np.unique(idx, return_index=True)
    print('Number of unique sources:', len(idx_unique), len(idx))
    print('idxmatch range:', np.min(idx_unique_indices),
          np.max(idx_unique_indices))
    print()

    # match_to_catalog_sky(catalogcoord[, nthneighbor])
    # Finds the nearest on-sky matches of this coordinate in a set
    # of catalog coordinates.
    print('Using: match_to_catalog_sky')
    idx, d2d, d3d = radec_drqso.match_to_catalog_sky(radec_alma)

    print('len(idx):', len(idx))
    print('idxmatch range:', np.min(idx), np.max(idx))
    idx_unique, idx_unique_indices = np.unique(idx, return_index=True)
    print('Number of unique sources:', len(idx_unique), len(idx))
    print('idxmatch range:', np.min(idx_unique_indices),
          np.max(idx_unique_indices))
    print()

    print(len(d2d), len(idx_unique), len(idx_unique_indices))

    # create table of Veron quasars that have ALMA observations
    drqso_alma = drqso[idx_unique_indices]
    xdata = d2d[idx_unique_indices].arcsec
    itest = xdata  < 15.0

    drqso_alma = drqso_alma[itest]
    ndata = len(drqso_alma)
    print('Number of Quasars within 15" match radius:', ndata)
    print()

    if dr == 'dr7':
        for isource, source in enumerate(drqso_alma):
            print(isource + 1, source['SDSSJ'],
                  source['z'], source['RMAG'],
                  source['FIRSTMAG'], source['ONAME'])


    if dr == 'dr12':
        for isource, source in enumerate(drqso_alma):
            print(isource + 1, source['SDSS_NAME'],
                  source['Z_VI'], source['PSFMAG'][2],
                  source['FIRST_FLUX'])


    return drqso_alma


def veron2010_analysis(radec_alma=None):
    """
    read and match to Veron+2010 quasar catalogue
    """
    infile_Veron2010 = config.get("defaults", "infile_Veron2010")
    print('Reading:', infile_Veron2010)
    Veron2010 = Table.read(infile_Veron2010)
    if debug:
        Veron2010.info('stats')

    ra = Veron2010['RAJ2000']
    dec = Veron2010['DEJ2000']
    radec_veron = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')


    # See http://docs.astropy.org/en/stable/coordinates/matchsep.html

    # match_coordinates_sky(matchcoord, catalogcoord,
    #                       nthneighbor=1, storekdtree='_kdtree_sky')
    # Finds the nearest on-sky matches of a coordinate or coordinates in
    # a set of catalog coordinates.
    print('Using: match_coordinates_sky')
    idx, d2d, d3d = match_coordinates_sky(radec_alma, radec_veron)

    print('len(idx):', len(idx))
    print('idxmatch range:', np.min(idx), np.max(idx))
    idx_unique, idx_unique_indices = np.unique(idx, return_index=True)
    print('Number of unique sources:', len(idx_unique), len(idx))
    print('idxmatch range:', np.min(idx_unique_indices),
          np.max(idx_unique_indices))
    print()

    # match_to_catalog_sky(catalogcoord[, nthneighbor])
    # Finds the nearest on-sky matches of this coordinate in a set
    # of catalog coordinates.
    print('Using: match_to_catalog_sky')
    idx, d2d, d3d = radec_veron.match_to_catalog_sky(radec_alma)

    print('len(idx):', len(idx))
    print('idxmatch range:', np.min(idx), np.max(idx))
    idx_unique, idx_unique_indices = np.unique(idx, return_index=True)
    print('Number of unique sources:', len(idx_unique), len(idx))
    print('idxmatch range:', np.min(idx_unique_indices),
          np.max(idx_unique_indices))
    print()

    print(len(d2d), len(idx_unique), len(idx_unique_indices))

    # create table of Veron quasars that have ALMA observations
    veron_alma = Veron2010[idx_unique_indices]
    xdata = d2d[idx_unique_indices].arcsec
    itest = xdata  < 15.0

    veron_alma = veron_alma[itest]
    ndata = len(veron_alma)
    print('Number of Veron+2010 Quasars within 15" match radius:', ndata)
    print()

    if debug:
        key=raw_input("Enter any key to continue: ")

    xdata = xdata[itest]
    ndata = len(xdata)
    print('Number of Veron+2010 Quasars within 15" match radius:', ndata)

    plt.hist(xdata, bins=60, label=str(ndata))
    plt.grid()
    plt.legend()
    plt.title(infile)
    plt.xlabel('Separation (")')
    plt.ylabel('Frequency per bin')

    plotfile = infile + '_Separation_hist.png'
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

    # limit to redshift > 0.3 and 20cm/6cm flux less that 10mJy or NAN
    print(len(veron_alma['z']))
    isnotnan = ~np.isnan(veron_alma['z'])
    print(len(isnotnan))
    itest = veron_alma['z'] >= 0.3
    print(len(itest))
    print(np.min(veron_alma['z']), np.max(veron_alma['z']))
    print(np.nanmin(veron_alma['z']), np.nanmax(veron_alma['z']))
    print(len(~np.isnan(veron_alma['F6cm'])))
    itest = (~np.isnan(veron_alma['F6cm']))
    print(len(itest))
    itest = (veron_alma['z'] >= 0.3) & (np.isnan(veron_alma['F6cm']))
    print(len(itest))

    itest = (veron_alma['F6cm'] <= 0.01)
    print(len(itest))

    itest = (~np.isnan(veron_alma['z'])) & (veron_alma['z'] >= 0.3)
    print(len(itest))

    itest = (np.isnan(veron_alma['F6cm'])) | (veron_alma['F6cm'] <= 0.01)
    print(len(itest))

    itest = (np.isnan(veron_alma['F20cm'])) | (veron_alma['F20cm'] <= 0.01)
    print(len(itest))


    itest_rqq = (
        ((~np.isnan(veron_alma['z'])) & (veron_alma['z'] >= 0.3)) &
        ((np.isnan(veron_alma['F6cm'])) | (veron_alma['F6cm'] <= 0.01)) &
        ((np.isnan(veron_alma['F20cm'])) | (veron_alma['F20cm'] <= 0.01)))
    print(len(itest_rqq))
    print(len(veron_alma[itest_rqq]))

    veron_alma[itest_rqq].info('stats')
    ndata = len(veron_alma)

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

    #
    veron_alma[itest_rqq].info('stats')
    xdata = veron_alma[itest_rqq]['RAJ2000']
    ydata = veron_alma[itest_rqq]['DEJ2000']

    plt.close()
    plot_radec(xdata, ydata,
               title=alma.meta['file'], infile=infile,
               filelabel='rqq')

    return veron_alma


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

    read_csv = args.read_csv
    read_fits = args.read_fits

    write_fits = args.write_fits
    convert_csv = args.convert_csv
    wget_csvfile = args.wget_csvfile

    debug = args.debug

    config = rd_config(config_file=None)
    config = rd_config()

    infile_prefix_aa = config.get("defaults", "infile_prefix_aa")
    print('infile_prefix_aa:', infile_prefix_aa)

    if wget_csvfile:
        wget_archivelog()
        sys.exit()

    if read_csv or convert_csv:
        infile = infile_prefix_aa + '.csv'
        print('Read:', infile)

        alma = Table.read(infile, format='ascii')
        alma.meta['file'] = infile
        alma.info('stats')

    if write_fits or convert_csv:
        outfile = infile_prefix_aa + '.fits'
        alma.write(outfile, overwrite=True)

    if read_fits:
        infile = infile_prefix_aa + '.fits'
        print('Reading:', infile)
        alma = Table.read(infile)
        alma.meta['file'] = infile
        print('Elapsed time(secs):', time.time() - t0)
        alma.info('stats')
        colnames = alma.colnames
        ncolumns = len(alma.colnames)
        print('Number of columns:', ncolumns)
        for i in range(0, ncolumns):
            data_min = np.min(np.array(alma.field(i), dtype=object))
            data_max = np.max(np.array(alma.field(i), dtype=object))
            # data_median = np.median(np.array(table.field(i), dtype=object))
            print(i+1, colnames[i], data_min, data_max)

        if debug:
            key=raw_input("Enter any key to continue: ")

    ra = alma['RA']
    dec = alma['Dec']
    plot_radec(ra, dec,
               title = alma.meta['file'], infile=infile)

    Largest_angular_scale(data=alma['Largest angular scale'])

    if debug:
        key=raw_input("Enter any key to continue: ")

    Frequency_support(data=alma['Frequency support'], infile=infile,
                      debug=debug)

    Integration(data=alma['Integration'])

    # plt.show()
    # create astropy RA, Dec 'object' thing
    radec_alma = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')

    # read and match to Veron+2010 quasars
    veron2010_alma = veron2010_analysis(radec_alma=radec_alma)

    # read and match to SDSS DR7QSO quasars
    dr7qso_alma = drqso_analysis(radec_alma=radec_alma)

    dr12qso_alma = drqso_analysis(radec_alma=radec_alma, dr='dr12')

    # make a html table with NED and SDSS DR7 and DR12 links

    # match back to alma observataions
