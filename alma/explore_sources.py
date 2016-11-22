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

from astropy.table import Table, Column


# local functions
import explore_alma_archive as eaa
from plotid import *
from table_stats import *
from write_ds9_regionfile import *

def make_ds9regionfile(data):
    """

    """

    unique_files, index, counts = np.unique(
        data['FILE'], return_index=True, return_counts=True)
    print('Number of unique source files:', len(unique_files))

    nchars_max = 0
    for ifile, file, in enumerate(unique_files):
        print(ifile+1, ifile, counts[ifile], file)
        # parse the project info from file name
        file_parsed = file.split("/")
        project = file_parsed[1]
        nchars = len(project)
        nchars_max = max(nchars_max, nchars)
        print('Project:', project, len(project))
        print('Image filename:', file_parsed[-1])

    print('Maximum string length:', nchars_max)
    nrows = len(data)
    print('Number of rows:', nrows)

    # create empty PROJECT column
    dtype = 'S' + str(nchars_max)
    print('dtype:', dtype)
    project = Column(length=nrows, name='PROJECT', dtype=dtype)
    data.add_column(project, index=1)

    for ifile, file, in enumerate(data['FILE']):
        # parse the project info from file name
        file_parsed = file.split("/")
        project = file_parsed[1]
        nchars = len(project)
        nchars_max = max(nchars_max, nchars)
        data['PROJECT'][ifile] = project

    data.info()

    unique_projects, index, counts = np.unique(
        data['PROJECT'], return_index=True, return_counts=True)
    print('Number of unique projects:', len(unique_projects))

    for iproject, project, in enumerate(unique_projects):
        print(iproject, iproject+1, project, counts[iproject])

    for ifile, file, in enumerate(unique_files):
        itest = (data['FILE'] == file)
        print(ifile, file, len(data[itest]))
        ralist = data['ALPHA_J2000'][itest]
        declist = data['DELTA_J2000'][itest]
        file_parsed = file.split("/")
        filename_ds9 = file_parsed[-1] + '.ds9.reg'
        comments = '# Project: ' + data['PROJECT'][itest[0]] + '\n' \
            + '# ' + data['FILE'][itest[0]] + '\n'
        write_ds9_regionfile(ralist, declist,
                             filename=filename_ds9, symbol='circle',
                             color='green', comments=comments)

    return data

def source_stats(data, filename=None, showplot=False, saveplot=True):
    """analyze the sources file


    """
    print('Number of sources:', len(data))

    save = data

    unique_files, index, counts = np.unique(
        data['FILE'], return_index=True, return_counts=True)
    print('Number of unique source files:', len(unique_files))

    for ifile, file, in enumerate(unique_files):
        print(ifile+1, ifile, counts[ifile], file)


    # create data array with one row per file
    data_test = data[index]

    unique_naxis3, index, counts = np.unique(
        data_test['NAXIS3'], return_index=True, return_counts=True)
    print('Number of unique NAXIS3:',
          len(unique_naxis3), np.sum(unique_naxis3))

    for idata, naxis3, in enumerate(unique_naxis3):
        print(idata+1, idata, counts[idata], naxis3)

    suptitle =''
    if filename is not None:
        suptitle = filename
    title = str(len(unique_files)) + ' files' + ' :NAXIS3 frequency'

    nbins = np.max(counts) + 1
    print('nbins:', nbins)
    xdata = data_test['NAXIS3']
    print('Data range:', np.min(xdata), np.max(xdata))
    nbins = np.max(xdata) + 1
    range = (0, np.max(xdata))
    n, bins, patches = plt.hist(xdata, nbins, range=range,
                                label = str(len(counts)),
                                facecolor='green', alpha=0.75)


    if title is not None: plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend()
    plt.xlabel('NAXI3')
    plt.ylabel('Frequency')
    plotfile = 'sources_histogram_NAXIS3.png'
    plotid()
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    #  key=raw_input("Enter any key to continue: ")

    title = str(len(data)) + ' sources' + ' versus NAXIS3 value'
    n, bins, patches = plt.hist(data['NAXIS3'], nbins, range=range,
                                label = str(len(counts)),
                                facecolor='green', alpha=0.75)


    if title is not None: plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend()
    plt.xlabel('NAXI3')
    plt.ylabel('Frequency')
    plotfile = 'sources_histogram_sources_ByNAXIS3.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()


    # key=raw_input("Enter any key to continue: ")

    # the histogram of the source count data
    unique_files, index, counts = np.unique(
        data['FILE'], return_index=True, return_counts=True)
    print('Number of unique source files:', len(unique_files))

    print('Median:', np.median(counts))
    nbins = np.max(counts) + 1
    nbins=100
    title = str(len(data)) + ' sources'
    n, bins, patches = plt.hist(counts, nbins, label = str(len(counts)),
                                facecolor='green', alpha=0.75)

    if title is not None: plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend()
    plt.xlabel('Number of sources per image file')
    plt.ylabel('Frequency')
    plotfile = 'sources_histogram_ByFile.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()

    nbins = np.max(counts) + 1
    range = [np.min(counts), np.max(counts)+1]
    print('range:', range)
    n, bins, patches = plt.hist(counts, bins=nbins,
                                label = 'files:' + str(len(counts)),
                                range=range,
                                facecolor='green', alpha=0.75,
                                histtype='step', cumulative=True)

    # Overlay a reversed cumulative histogram.
    plt.hist(counts, bins=nbins, histtype='step', cumulative=-1,
             range=range)

    plt.xlim(range)
    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='center right')
    plt.xlabel('Number of sources per image file')
    plt.ylabel('Cumulative Frequency')
    plotfile = 'sources_cumulative_ByFile.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()

    isort = np.argsort(counts)
    print(len(isort))
    cumulative = np.cumsum(counts[isort])
    plt.plot(cumulative, label='files:' + str(len(cumulative)))
    if title is not None: plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='center left')
    plt.grid()
    plt.xlabel('image file sorted by number of increasing source count')
    plt.ylabel('Cumulative total number of source ')
    plotfile = 'sources_cumulative_ByFileSourceumber.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    # plot histogram of frequency in Hz per image

    # limit to source with CUNIT3 == 'Hz' and CTYPE3 == 'FREQ'
    itest = (data['CTYPE3'] == 'FREQ') & (data['NAXIS3'] == 1)
    data = data[itest]
    print('Number of sources with CTYPE3 = FREQ:', len(data))

    unique_files, index, counts = np.unique(
        data['FILE'], return_index=True, return_counts=True)
    print('Number of unique source files:', len(unique_files))

    data = data[index]
    xdata = data['CRVAL3'] / 1e9
    range = [np.min(xdata), np.max(xdata)]
    print('data range:', range)
    nbins = 100
    plt.hist(xdata, bins=nbins, histtype='step', label=str(len(xdata)))

    title = str(len(xdata)) + ' image files'
    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='center right')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Number')
    plotfile = 'sources_histogram_images_ByFreq.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    # wavelength in mm
    xdata = 300.0 / xdata
    plt.hist(xdata, bins=nbins, histtype='step', label=str(len(xdata)))

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend()
    plt.xlabel('Wavelength (mm)')
    plt.ylabel('Number')
    plotfile = 'sources_histogram_images_ByWavelenth.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    xdata = 300.0e9 / data['CRVAL3']
    ydata = counts
    plt.plot(xdata, ydata, '.', label=str(len(xdata)))

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend()
    plotid()
    plt.xlabel('Wavelength (mm)')
    plt.ylabel('Number of sources per image')

    plotfile = 'sources_NumberPerImage_ByWavelenth.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    # look at the shape in terms of the beam
    itest = (data['BMAJ'] == 'NA')
    print("BMAJ == 'NA':", len(data[itest]))
    itest = (data['BMAJ'] == 'NA')
    print("BMIN == 'NA':", len(data[itest]))

    itest = (data['BMAJ'] != 'NA') & (data['BMIN'] != 'NA')
    data = data[itest]
    xdata = data['BMAJ'].astype('f16')*3600.0
    ydata = data['BMIN'].astype('f16')*3600.0

    xrange = [0.0, np.max(xdata)]
    yrange = [0.0, np.max(ydata)]
    xrange[1] = max(xrange[1], yrange[1])
    yrange[1] = max(xrange[1], yrange[1])

    plt.plot(xdata, ydata, '.', label=str(len(xdata)))

    plt.axes().set_aspect('equal')
    plt.xlim(xrange)
    plt.ylim(yrange)

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='upper left')
    plotid()
    plt.xlabel('BMAJ (arcsec)')
    plt.ylabel('BMIN (arcsec)')

    plotfile = 'sources_BMAJ_v_BMIN_arcsec.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    # look at the  image pixel size
    xdata = data['BMAJ'].astype('f16')/data['CDELT2']
    ydata = data['BMIN'].astype('f16')/data['CDELT2']

    xrange = [0.0, np.max(xdata)]
    yrange = [0.0, np.max(ydata)]
    xrange[1] = max(xrange[1], yrange[1])
    yrange[1] = max(xrange[1], yrange[1])

    plt.plot(xdata, ydata, '.', label=str(len(xdata)))

    plt.axes().set_aspect('equal')
    plt.xlim(xrange)
    plt.ylim(yrange)


    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='upper left')
    plotid()
    plt.xlabel('BMAJ/CDELT2')
    plt.ylabel('BMIN/CDELT2')

    plotfile = 'sources_BMAJ_v_BMIN_pixels.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    # look at the BEAM Sampling
    xdata = data['CDELT1']*3600.0
    ydata = data['CDELT2']*3600.0

    plt.plot(xdata, ydata, '.', label=str(len(xdata)))

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='upper left')
    plotid()
    plt.xlabel('CDELT1 (arcsec)')
    plt.ylabel('CDELT2 (arcsec)')

    plotfile = 'sources_CDELT1_v_CDELT2.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    # look at Kron radius versus sqrt(BMAJ^2 + BMIN^2)
    data = save
    itest = (data['CTYPE3'] == 'FREQ') & (data['NAXIS3'] == 1)
    data = data[itest]

    itest = (data['BMAJ'] != 'NA') & (data['BMIN'] != 'NA')
    data = data[itest]

    xdata = np.sqrt(np.square(data['BMAJ'].astype('f16')) + \
                    np.square(data['BMIN'].astype('f16'))) / data['CDELT2']
    ydata = data['KRON_RADIUS']

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5, label=str(len(xdata)))

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='upper right')
    plotid()
    plt.xlabel('<BEAM> pixels')
    plt.ylabel('KRON_RADIUS pixels')

    plotfile = 'sources_BeamPixels_v_KronRadius.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    xdata = np.sqrt(np.square(data['BMAJ'].astype('f16')) + \
                    np.square(data['BMIN'].astype('f16'))) / data['CDELT2']
    ydata = data['FWHM_IMAGE']

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5, label=str(len(xdata)))

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='upper right')
    plotid()
    plt.xlabel('<BEAM> pixels')
    plt.ylabel('FWHM_IMAGE pixels')

    plt.show()

    unique_flags, index, counts = np.unique(
        data['FLAGS'], return_index=True, return_counts=True)
    print('Number of unique source FLAGS:', len(unique_flags))
    for iflag, flag, in enumerate(unique_flags):
        print(iflag + 1, iflag, counts[iflag], flag)


    range = [0, np.max(unique_flags)+1]
    nbins = range[1]
    xdata = data['FLAGS']
    plt.hist(xdata, bins=nbins, histtype='step', range=range,
             label=str(len(xdata)))

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend()
    plotid()
    plt.xlabel('FLAGS')
    plt.ylabel('Number')
    plt.show()
    plt.close()


    return


def Calibrators(alma=None, calibrators=None, plottitle='',
                showplot=True):
     """

     """
     # read in calibrators
     calibrators = Table.read('calibrators_20161117.fits')
     ra = calibrators['RA']
     dec = calibrators['Dec']
     radec_calibrators = SkyCoord(ra, dec,
                                  unit=(u.degree, u.degree), frame='icrs')

     # assign ALMA sources ra, dec arrays
     ra = alma['ALPHA_J2000']
     dec = alma['DELTA_J2000']
     radec_alma = SkyCoord(ra, dec,
                           unit=(u.degree, u.degree), frame='icrs')

     print('Using: match_to_catalog_sky')
     idxmatch, d2d, d3d = radec_alma.match_to_catalog_sky(radec_calibrators)

     print('len(idxmatch):', len(idxmatch))
     print('idxmatch range:', np.min(idxmatch), np.max(idxmatch))
     print('len(radec_calibrators):', len(radec_calibrators))
     print('len(radec_alma):', len(radec_alma))
     print('d2d range:', np.min(d2d), np.max(d2d))
     print('d2d range:', np.min(d2d).arcsec, np.max(d2d).arcsec)


     xdata = d2d.arcsec
     itest = (xdata <= 30.0)
     xdata = xdata[itest]
     nbins = 60
     range = (0.0, 30.0)
     n, bins, patches = plt.hist(xdata, nbins, range=range,
                            label = str(len(xdata)),
                            facecolor='green', alpha=0.75)
     plt.xlim(range)
     if plottitle is not None:
         title = plottitle
         plt.title(title , fontsize='medium')
     plt.legend()
     plt.xlabel('d2d (arcsec)')
     plt.ylabel('Frequency')
     plotfile = 'sources_histogram_dr_calibrators.png'
     plotid()
     plt.savefig(plotfile)
     if showplot:
         plt.show()
     plt.close()

     return


def SourceDistribution(alma=None, plottitle='', xyfilter=False):
    """

    """
    title = plottitle

    showplot = True
    plt.close()
    xdata = alma['X_IMAGE']
    ydata = alma['Y_IMAGE']
    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))

    plt.xlabel('X_IMAGE')
    plt.ylabel('Y_IMAGE')

    plt.legend(fontsize='medium')
    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_xydistribution.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    showplot = True
    plt.close()
    xdata = alma['X_IMAGE']
    ydata = alma['Y_IMAGE']
    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.legend(fontsize='medium')
    plotid()
    plt.xlim((0, 400.0))
    plt.ylim((0, 400.0))
    plt.axes().set_aspect('equal')
    plotfile = 'sources_xydistribution_zoom.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    showplot = True
    plt.close()
    xdata = alma['X_IMAGE']/alma['NAXIS1']
    ydata = alma['Y_IMAGE']/alma['NAXIS2']
    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.title(title, fontsize='medium')
    plt.legend(fontsize='medium')
    plt.xlabel('NAXIS1/X_IMAGE')
    plt.ylabel('NAXIS2/Y_IMAGE')

    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_xydistribution_ratio.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    showplot = True
    xdata = alma['X_IMAGE']/alma['NAXIS1']
    ydata = alma['Y_IMAGE']/alma['NAXIS2']

    itest = (xdata > 0.45) & (xdata < 0.55) & (ydata > 0.45) & (ydata < 0.55)

    xdata = xdata[itest]
    ydata = ydata[itest]

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.grid()

    plt.title(title, fontsize='medium')
    plt.legend(fontsize='medium')
    plt.xlabel('NAXIS1/X_IMAGE')
    plt.ylabel('NAXIS2/Y_IMAGE')
    plt.xlim((0.45, 0.55))
    plt.ylim((0.45, 0.55))

    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_xydistribution_ratio_zoom.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()



    showplot = True
    xdata = (alma['X_IMAGE']-1.0)/alma['NAXIS1']
    ydata = (alma['Y_IMAGE']-1.0)/alma['NAXIS2']

    itest = (xdata > 0.495) & (xdata < 0.505) & (ydata > 0.495) \
        & (ydata < 0.505)

    xdata = xdata[itest]
    ydata = ydata[itest]

    xmedian = np.median(xdata)
    ymedian = np.median(ydata)
    print('Median (xdata):', xmedian)
    print('Median (ydata):', ymedian)

    dr = np.sqrt(np.square(xdata - xmedian) + np.square(ydata - ymedian))

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.grid()

    plt.title(title, fontsize='medium')
    plt.legend(fontsize='medium')
    plt.xlabel('NAXIS1/X_IMAGE')
    plt.ylabel('NAXIS2/Y_IMAGE')
    plt.xlim((0.495, 0.505))
    plt.ylim((0.495, 0.505))

    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_xydistribution_ratio_zoom2.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    xdata = xdata - xmedian
    ydata = ydata - ymedian

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.grid()

    plt.title(title, fontsize='medium')
    plt.legend(fontsize='medium')
    plt.xlabel('Relative NAXIS1/X_IMAGE')
    plt.ylabel('Relative NAXIS2/Y_IMAGE')
    plt.xlim((-0.005, 0.005))
    plt.ylim((-0.005, 0.005))

    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_xydistribution_ratio_zoom3.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    xdata = dr
    nbins = 50
    range = (0, np.max(xdata))
    n, bins, patches = plt.hist(xdata, nbins,
                                label = str(len(xdata)),
                                facecolor='green', alpha=0.75)

    if title is not None: plt.title(title , fontsize='medium')
    plt.title(title)
    plt.legend()
    plt.xlabel('dr (normalised)')
    plt.ylabel('Frequency')
    plotfile = 'sources_histogram_dr.png'
    plotid()
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    # filter the data
    xdata = (alma['X_IMAGE']-1.0)/alma['NAXIS1']
    ydata = (alma['Y_IMAGE']-1.0)/alma['NAXIS2']

    # limit to central region in x and y
    itest = (xdata > 0.495) & (xdata < 0.505) & (ydata > 0.495) \
        & (ydata < 0.505)

    xdata = xdata[itest]
    ydata = ydata[itest]

    xmedian = np.median(xdata)
    ymedian = np.median(ydata)
    print('Median (xdata):', xmedian)
    print('Median (ydata):', ymedian)

    # now apply test to whole file
    xdata = (alma['X_IMAGE']-1.0)/alma['NAXIS1']
    ydata = (alma['Y_IMAGE']-1.0)/alma['NAXIS2']
    dr = np.sqrt(np.square(xdata - xmedian) + np.square(ydata - ymedian))

    itest = (dr <= 0.001)

    result = alma[itest]

    return result

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

    Calibrators(alma=alma, calibrators=None, plottitle=infile)

    # extract the project ID from the file
    project = []

    # table = ascii.read(infile, header_start=0, data_start=1)
    # print()
    # print(table[0])


    # table = Table.read(infile)

    table_stats('sources.fits', debug=False)
    alma.info()
    alma.info('stats')

    result = SourceDistribution(alma=alma, plottitle=infile, xyfilter=True)
    print('Number of rows in xy filtered source file:', len(result))
    result.write('result.fits', overwrite=True)

    sys.exit()

    # make_ds9regionfile(alma)
    # sys.exit()

    sources_stats(alma, filename=infile)

    print('Number of sources:', len(alma))
    # limit to source with NAXIS3 == 1
    itest = (alma['NAXIS3'] == 1)
    alma = alma[itest]
    print('Number of sources:', len(alma))





    # limit to source with CUNIT3 == 'Hz' and CTYPE3 == 'FREQ'
    itest = (alma['CTYPE3'] == 'FREQ')
    alma = alma[itest]
    print('Number of sources:', len(alma))

    itest = (alma['CUNIT3'] == 'Hz')
    alma = alma[itest]
    print('Number of sources:', len(alma))

    print('Check for FLAGS != 0')
    itest = (alma['FLAGS'] == 0)
    alma = alma[itest]
    print('Number of sources with FLAGS ==0 :', len(alma))

    itest = (np.char.find(alma['FILE'],'spw') == -1)
    alma = alma[itest]
    print('Number of sources not in spw files:', len(alma))

    # sys.exit()

    showplot = True
    limitflux = True
    if limitflux:
        itest = (alma['FLUX_AUTO'] > 0.002) & (alma['FLUX_AUTO'] < 0.030) & \
                (alma['FLUX_AUTO']/alma['FLUXERR_AUTO'] > 5.0) & \
                (alma['FLUX_AUTO']/alma['FLUXERR_AUTO'] < 50.0)

        alma = alma[itest]

    xdata = alma['FLUX_AUTO']
    ydata = alma['FLUXERR_AUTO']
    itest = (xdata > 0) & (xdata < 1.0)
    xdata = xdata[itest]
    ydata = ydata[itest]
    plt.suptitle(infile)
    plt.xlabel('FLUX_AUTO (Jy)')
    plt.ylabel('FLUXERR_AUTO (Jy)')
    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.legend(fontsize='medium')
    plotid()
    plotfile = 'sources_flux_fluxerr.png'
    plt.savefig(plotfile)
    if showplot:
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
    plt.plot(xdata, ydata, '.', alpha=0.5, label=len(xdata))
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

    range = 10.0
    itest = (abs(xdata) < range) & (abs(ydata) < range)
    xdata = xdata[itest]
    ydata = ydata[itest]
    plt.plot(xdata, ydata, '.', ms=4.0, alpha=0.5, label=len(xdata))
    # plt.axis('equal')
    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)

    plt.axes().set_aspect('equal')
    plt.xlabel('Delta RA (arcsec)')
    plt.ylabel('Delta Dec (arcsec)')
    plt.xlim((-1.0*range, range))
    plt.ylim((-1.0*range, range))

    plt.grid()
    plt.legend()
    plotid()
    plt.show()


    xdata = separation.arcsec
    itest = (abs(xdata) < 20.0)
    xdata = xdata[itest]
    nbins = 20
    plt.hist(xdata, bins=nbins, histtype='step',
             label=str(len(xdata)))

    plt.suptitle(infile)
    plt.suptitle(infile)
    plt.xlabel('Radial offset (arcsec)')
    plt.ylabel('Frequency')
    plt.legend()
    plotid()
    plt.show()
