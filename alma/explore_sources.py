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
        print('Project:', project, len(project))
        print('Image filename:', file_parsed[-1])


    print('Maximum string length:', nchars_max)
    nrows = len(data)
    print('Number of rows:', nrows)


   # add project column
   # determine the length of the PROJECT string
    nchars_max = 0
    for ifile, file, in enumerate(data['FILE']):
        # parse the project info from file name
        file_parsed = file.split("/")
        project = file_parsed[1]
        nchars = len(project)
        nchars_max = max(nchars_max, nchars)


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


def SourceStats(data, verbose=False):
    """analyze the sources file

    """
    print('Number of sources:', len(data))

    unique_files, index, counts = np.unique(
        data['FILE'], return_index=True, return_counts=True)
    print('Number of unique source files:', len(unique_files))

    unique_files, index, counts = np.unique(
        data['PROJECTCODE'], return_index=True, return_counts=True)
    print('Number of unique Project codes:', len(unique_files))


def Image_SourceStats(data, filename=None, showplot=False, saveplot=True,
                      verbose=False):
    """analyze the sources file


    """

    print('Total number of sources:', len(data))

    save = data

    unique_files, index, counts = np.unique(
        data['FILE'], return_index=True, return_counts=True)
    print('Number of unique source image files:', len(unique_files))

    print('Source counts per image range:', np.min(counts), np.max(counts))
    # sort by source counts per image
    isort = np.argsort(counts)
    if verbose:
        for ifile, file, in enumerate(unique_files[isort]):
            print(ifile+1, ifile, counts[isort][ifile], file)
            print(ifile+1, ifile, data['FILE'][index[isort][ifile]])
            print(ifile+1, ifile, data['NAXIS3'][index[isort][ifile]])
            print(ifile+1, ifile, data['CTYPE3'][index[isort][ifile]])

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

    # the histogram of the source counts per image
    unique_files, index, counts = np.unique(
        data['FILE'], return_index=True, return_counts=True)
    print('Number of unique source image files:', len(unique_files))

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
    plotfile = 'sources_histogram_ByImageFile.png'
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


    ydata = data['FLUX_RADIUS']

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5, label=str(len(xdata)))

    if title is not None:
        plt.title(title , fontsize='medium')
    plt.suptitle(suptitle)
    plt.legend(loc='upper right')
    plotid()
    plt.xlabel('<BEAM> pixels')
    plt.ylabel('FLUX_RADIUS pixels')

    plotfile = 'sources_BeamPixels_v_FluxRadius.png'
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


def xmatch_calibrators(alma=None, calibrators=None,
                filter_radec=True, filter_radec_rmax=2.0,
                plottitle='',
                showplot=True):
    """


    """
    # read in calibrators
    infile = 'calibrators_20161117.fits'
    calibrators = Table.read(infile)
    print('Number of rows in:', infile, len(calibrators))
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
    dra, ddec = radec_alma.spherical_offsets_to(radec_calibrators[idxmatch])
    separation = radec_alma.separation(radec_calibrators[idxmatch])

    print('len(idxmatch:', len(idxmatch))
    print('idxmatch range:', np.min(idxmatch), np.max(idxmatch))
    print('len(radec_calibrators):', len(radec_calibrators))
    print('len(radec_alma):', len(radec_alma))
    print('d2d range:', np.min(d2d), np.max(d2d))
    print('d2d range:', np.min(d2d).arcsec, np.max(d2d).arcsec)

    print('Radial separation range:', np.min(separation).arcsec,
          np.max(separation).arcsec)

    print('dRA range:', np.min(dra).arcsec, np.max(dra).arcsec)
    print('dDec range:', np.min(ddec).arcsec, np.max(ddec).arcsec)


    itest = (d2d.arcsec < 1.0)
    print('Number of sources with calibrator within 1 arcsec:', len(d2d[itest]))
    itest = (d2d.arcsec < 1.414)
    print('Number of sources with calibrator within 1.414 arcsec:', len(d2d[itest]))
    itest = (d2d.arcsec < 2.0)
    print('Number of sources with calibrator within 2 arcsec:', len(d2d[itest]))
    itest = (d2d.arcsec < 3.0)
    print('Number of sources with calibrator within 3 arcsec:', len(d2d[itest]))
    itest = (d2d.arcsec < 4.0)
    print('Number of sources with calibrator within 4 arcsec:', len(d2d[itest]))
    itest = (d2d.arcsec < 6.0)
    print('Number of sources with calibrator within 6 arcsec:', len(d2d[itest]))
    itest = (d2d.arcsec < 8.0)
    print('Number of sources with calibrator within 8 arcsec:', len(d2d[itest]))

    key=raw_input("Enter any key to continue: ")

    rmax = 90
    xdata = d2d.arcsec
    itest = (xdata <= 120.0)
    ndata_all = len(xdata)
    xdata = xdata[itest]
    nbins = int(rmax)
    range = (0.0, rmax)
    ndata = len(xdata)
    n, bins, patches = plt.hist(xdata, nbins, range=range,
                            label = str(ndata) + ' from ' + str(ndata_all),
                            facecolor='green', alpha=0.75)
    plt.xlim(range)
    if plottitle is not None:
        title = plottitle
        plt.title(title , fontsize='medium')
    plt.legend()
    suptitle = 'pairwise radial distance for source to nearest calibrator'
    plt.suptitle(suptitle)
    plt.xlabel('d2d (arcsec)')
    plt.ylabel('Frequency')
    plotfile = 'explore_sources_xmatch_calibrators_histogram_dr.png'
    plotid()
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    print('dRa range:', np.min(dra), np.max(dra))
    print('dDEC range:', np.min(ddec), np.max(ddec))


    xdata = dra.arcsec
    ydata = ddec.arcsec
    itest = (abs(xdata) < rmax) & (abs(ydata) < rmax)

    xdata = xdata[itest]
    ydata = ydata[itest]
    ndata = len(xdata)
    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label = str(ndata) + ' from ' + str(ndata_all))

    plt.xlim(-1.0*rmax, rmax)
    plt.ylim(-1.0*rmax, rmax)
    if plottitle is not None:
        title = plottitle
        plt.title(title , fontsize='medium')
    plt.legend()
    suptitle = 'pairwise separation for source to nearest calibrator'
    plt.suptitle(suptitle)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec')
    plotfile = 'explore_sources_xmatch_calibrators_dra_ddec.png'
    plotid()
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    rmax = 5.0
    xdata = dra.arcsec
    ydata = ddec.arcsec
    itest = (abs(xdata) < rmax) & (abs(ydata) < rmax)

    xdata = xdata[itest]
    ydata = ydata[itest]
    ndata = len(xdata)
    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label = str(ndata) + ' from ' + str(ndata_all))

    plt.xlim(-1.0*rmax, rmax)
    plt.ylim(-1.0*rmax, rmax)
    if plottitle is not None:
        title = plottitle
        plt.title(title , fontsize='medium')
    plt.legend()
    suptitle = 'pairwise separation for source to nearest calibrator'
    plt.suptitle(suptitle)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec')
    plotfile = 'explore_sources_xmatch_calibrators_dra_ddec_zoom.png'
    plotid()
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    if filter_radec:
        itest = (d2d.arcsec > filter_radec_rmax)
        alma = alma[itest]

    return alma


def SourceDistribution(alma=None, plottitle='',
                       xyfilter=False,
                       radecfilter=True,
                       radecfilter_radius=2.0):
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

    itest = (xdata > 0.475) & (xdata < 0.525) & (ydata > 0.475) \
        & (ydata < 0.525)

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
    plt.xlim((0.475, 0.525))
    plt.ylim((0.475, 0.525))

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
    plt.xlim((-0.01, 0.01))
    plt.ylim((-0.01, 0.01))

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
    plotfile = 'sources_histogram_dr_xy.png'
    plotid()
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    # filter the data
    xdata = (alma['X_IMAGE']-1.0)/alma['NAXIS1']
    ydata = (alma['Y_IMAGE']-1.0)/alma['NAXIS2']

    # limit to central region in x and y
    itest = (xdata > 0.49) & (xdata < 0.51) & (ydata > 0.49) \
        & (ydata < 0.51)

    xdata = xdata[itest]
    ydata = ydata[itest]

    xmedian = np.median(xdata)
    ymedian = np.median(ydata)
    print('Median (xdata):', xmedian)
    print('Median (ydata):', ymedian)


    # now apply test to whole file in arc seconds
    xdata = ((alma['X_IMAGE']-1.0)-(0.5*alma['NAXIS1']))*alma['CDELT1']
    ydata = ((alma['Y_IMAGE']-1.0)-(0.5*alma['NAXIS2']))*alma['CDELT2']

    xdata = xdata * 3600
    ydata = ydata * 3600

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.grid()
    plt.axes().set_aspect('equal')

    plt.title(title, fontsize='medium')
    plt.legend(fontsize='medium')
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')

    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_skydistribution_relative_zoom1.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    # limit to central region in arcsec
    itest = (np.abs(xdata) < 5.0) & (np.abs(ydata) < 5.0)

    xdata = xdata[itest]
    ydata = ydata[itest]

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.grid()

    plt.title(title, fontsize='medium')
    plt.legend(fontsize='medium')
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')

    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_skydistribution_relative_zoom2.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()

    # limit to central region in arcsec
    itest = (np.abs(xdata) < 2.0) & (np.abs(ydata) < 2.0)

    xdata = xdata[itest]
    ydata = ydata[itest]

    plt.plot(xdata, ydata, '.', ms=2.0, alpha=0.5,
             label='Sources:' + str(len(xdata)))
    plt.grid()

    plt.title(title, fontsize='medium')
    plt.legend(fontsize='medium')
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')

    plotid()
    plt.axes().set_aspect('equal')
    plotfile = 'sources_skydistribution_relative_zoom3.png'
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()


    xdata = ((alma['X_IMAGE']-1.0)-(0.5*alma['NAXIS1']))*alma['CDELT1']
    ydata = ((alma['Y_IMAGE']-1.0)-(0.5*alma['NAXIS2']))*alma['CDELT2']

    xdata = xdata * 3600
    ydata = ydata * 3600

    xmedian = np.median(xdata)
    ymedian = np.median(ydata)

    dr = np.sqrt(np.square(xdata - xmedian) + np.square(ydata - ymedian))

    itest = (dr < 4.0)
    ndata_all = len(dr)
    xdata = dr[itest]
    nbins = 40
    range = (0, np.max(xdata))
    n, bins, patches = plt.hist(
        xdata, nbins,
        label = str(len(xdata)) + '/' +  str(ndata_all),
        facecolor='green', alpha=0.75)

    if title is not None: plt.title(title , fontsize='medium')
    # if suptitle is not None: plt.suptitle(suptitle , fontsize='medium')
    plt.legend()
    plt.xlabel('dr (arcsec)')
    plt.ylabel('Frequency')
    plotfile = 'sources_histogram_skydistribution_relative_dr_arcsecs.png'
    plotid()
    plt.savefig(plotfile)
    if showplot:
        plt.show()
    plt.close()



    if xyfilter:
    # now xy apply test to whole file
        xdata = (alma['X_IMAGE']-1.0)/alma['NAXIS1']
        ydata = (alma['Y_IMAGE']-1.0)/alma['NAXIS2']

        dr = np.sqrt(np.square(xdata - xmedian) + np.square(ydata - ymedian))

        itest = (dr <= 0.001)


    if radecfilter:
        # now apply radec test to whole file in arc seconds
        xdata = ((alma['X_IMAGE']-1.0)-(0.5*alma['NAXIS1']))*alma['CDELT1']
        ydata = ((alma['Y_IMAGE']-1.0)-(0.5*alma['NAXIS2']))*alma['CDELT2']

        xdata = xdata * 3600
        ydata = ydata * 3600

        xmedian = np.median(xdata)
        ymedian = np.median(ydata)

        dr = np.sqrt(np.square(xdata - xmedian) + np.square(ydata - ymedian))

        itest = (dr <= radecfilter_radius)


    result = alma[itest]
    print('RADEC filter radius:', radecfilter_radius)
    print('Number remaining after xy/radec filter:', len(alma[itest]))

    key=raw_input("Enter any key to continue: ")

    return result

def parse_file(alma=None, debug=False):
    """

    example file
    /./2013.1.01307.S/science_goal.uid___A001_X13b_X1a4/group.uid___A001_X13b_X1a5/member.uid___A001_X13b_X1a6/product/ngc1068_continuum.image.pbcor.fits.raw.fits

    """
    # add PROJECT CODE column extracted from the FILE column
    # need to determine the length of the PROJECT string
    nchars_max = 0
    project = []
    for ifile, file, in enumerate(alma['FILE']):
        # parse the project info from file name
        file_parsed = file.split("/")
        project_new = file_parsed[2]
        # determine max string length
        nchars = len(project_new)
        nchars_max = max(nchars_max, nchars)
        project.append(project_new)

    # convert to numpy string array
    project = np.asarray(project)

    print('Number of projects:', len(project))
    print('nchars_max:', nchars_max)
    print('type(project):', type(project))
    print(project.shape)
    print(project.dtype)
    if debug:
        key=raw_input("Enter any key to continue: ")

    # create empty PROJECT CODE column
    nrows = len(alma)
    dtype = 'S' + str(nchars_max)
    print('dtype:', dtype)
    new_column = Column(length=nrows, name='PROJECTCODE', dtype=dtype)
    alma.add_column(new_column, index=1)

    print(project[0])
    print(project[1])
    alma['PROJECTCODE'] = project

    print("type(alma['PROJECTCODE'])", type(alma['PROJECTCODE']))
    print("type(alma['PROJECTCODE'][0])", type(alma['PROJECTCODE'][0]))
    print("len(alma['PROJECTCODE'])", len(alma['PROJECTCODE']))
    print("len(alma['PROJECTCODE'][0])", len(alma['PROJECTCODE'][0]))
    print("len(alma['PROJECTCODE'][1])", len(alma['PROJECTCODE'][1]))
    print(alma['PROJECTCODE'][0])
    print(alma['PROJECTCODE'][1])

    if debug:
        key=raw_input("Enter any key to continue: ")

    # add SOURCE NAME column
    # determine the length of the string
    # could look for directory called product
    nchars_max = 0
    sourcename = []
    for ifile, file, in enumerate(alma['FILE']):
        # parse the project info from file name
        print(ifile, file)
        file_parsed = file.split("/")
        product_filename = file_parsed[-1]
        print(product_filename)
        sourcename_parsed = product_filename.split(".")
        sourcename_next = sourcename_parsed[0]
        nchars = len(sourcename_next)
        nchars_max = max(nchars_max, nchars)
        sourcename.append(sourcename_next)

    # convert to numpy string array
    sourcename = np.asarray(sourcename)

    # create empty SOURCE NAME column
    nrows = len(alma)
    dtype = 'S' + str(nchars_max)
    print('dtype:', dtype)
    new_column = Column(length=nrows, name='SOURCENAME', dtype=dtype)
    alma.add_column(new_column, index=2)

    alma['SOURCENAME'] = sourcename
    print(alma['SOURCENAME'][0])
    print(alma['SOURCENAME'][1])
    print(alma['SOURCENAME'][-1])


    if debug:
        key=raw_input("Enter any key to continue: ")

    # add PRODUCT FILENAME column
    # determine the length of the string
    # could look for directory called product
    nchars_max = 0
    product_filename = []
    for ifile, file, in enumerate(alma['FILE']):
        # parse the project info from file name
        print(ifile, file)
        file_parsed = file.split("/product/")
        product_filename_next = file_parsed[-1]
        print(product_filename_next)
        nchars = len(product_filename_next)
        nchars_max = max(nchars_max, nchars)
        product_filename.append(product_filename_next)

    # convert to numpy string array
    product_filename = np.asarray(product_filename)

    # create empty SOURCE NAME column
    nrows = len(alma)
    dtype = 'S' + str(nchars_max)
    print('dtype:', dtype)
    new_column = Column(length=nrows, name='PRODUCTFILENAME', dtype=dtype)
    alma.add_column(new_column, index=3)

    alma['PRODUCTFILENAME'] = product_filename
    print(alma['PRODUCTFILENAME'][0])
    print(alma['PRODUCTFILENAME'][1])
    print(alma['PRODUCTFILENAME'][-1])

    if debug:
        key=raw_input("Enter any key to continue: ")


    alma.info('stats')

    if debug:
        key=raw_input("Enter any key to continue: ")

    return alma



def xmatch_veron2010(alma=None, range=15.0,
                     filter_radec=True, filter_radec_rmax=2.0,
                     title=None, suptitle=None,
                     plotfile_prefix=None,
                     plotfile_suffix=None):
    """


    """

    ndata_all = len(alma)

    ra = alma['ALPHA_J2000']
    dec = alma['DELTA_J2000']

    radec_alma = SkyCoord(ra, dec,unit=(u.degree, u.degree), frame='icrs')

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

    itest = (abs(xdata) < range) & (abs(ydata) < range)

    xdata = xdata[itest]
    ydata = ydata[itest]
    plt.plot(xdata, ydata, '.', ms=4.0, alpha=0.5,
             label=str(len(xdata)) + '/' + str(ndata_all))
    # plt.axis('equal')
    if title is not None:
        plt.title(title, fontsize='medium')

    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.axes().set_aspect('equal')
    plt.xlabel('Delta RA (arcsec)')
    plt.ylabel('Delta Dec (arcsec)')
    plt.xlim((-1.0*range, range))
    plt.ylim((-1.0*range, range))

    plt.grid()
    plt.legend()
    plotid()

    plotfile = 'explore_sources_xmatch_Veron2010_radec.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.show()


    range1 = 5.0

    xdata = dra.arcsec
    ydata = ddec.arcsec
    itest = (abs(xdata) < range1) & (abs(ydata) < range1)
    xdata = xdata[itest]
    ydata = ydata[itest]
    plt.plot(xdata, ydata, '.', ms=4.0, alpha=0.5, label=len(xdata))
    # plt.axis('equal')
    if title is not None:
        plt.title(title, fontsize='medium')

    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.axes().set_aspect('equal')
    plt.xlabel('Delta RA (arcsec)')
    plt.ylabel('Delta Dec (arcsec)')
    plt.xlim((-1.0*range1, range1))
    plt.ylim((-1.0*range1, range1))

    plt.grid()
    plt.legend()
    plotid()

    plotfile = 'explore_sources_xmatch_Veron2010_radec_zoom1.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.show()

    xdata = separation.arcsec
    itest = (abs(xdata) < range)
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
    plotfile = 'explores_xmatch_Veron2010_rhist.png'
    print('Saving:', plotfile)
    plt.savefig(plotfile)
    plt.show()

    if filter_radec:
        itest = (separation.arcsec <= filter_radec_rmax)
        alma = alma[itest]
        print('RA Dec Separation range:', np.min(separation.arcsec),
              np.max(separation.arcsec))
        print('RA Dec Median:', np.median(separation.arcsec))
        print('radec_filter rmax:', filter_radec_rmax)
        print('Number of sources after radec filters:', len(alma))

    return alma


if __name__ == "__main__":
    """


    """
    import argparse
    import ConfigParser

    from astropy.table import Table
    from astropy.io import ascii
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import match_coordinates_sky
    from astropy import units as u

    infile = './sources300916.txt'
    infile = '/home/rgm/soft/obelics/bitbucket/results/sources031116.txt'
    infile = '/home/rgm/soft/obelics/bitbucket/results/sources111116.txt'

    # New version with SEMODE and full filename
    infile_path = '/data/vault/rgm/Projects/Obelics/ALMA/sourcelists/'

    #filename = 'sources221116.txt'

    filename = 'sources291116.txt'

    infile = infile_path +  filename

    # infile = 'sources300916_test2.txt'
    # infile = 'test.txt'
    # infile = 'sources300916_new.txt'

    print('Reading ALMA source list: ', infile)
    alma = ascii.read(infile, guess=False)
    alma.write('sources.fits', overwrite=True)

    print(alma[0])
    print('Number of rows:', len(alma))
    print()
    print('Number of columns:', len(alma.colnames))
    print('Number of data fields in row 1:', len(alma[0]))
    alma.info('stats')

    itest = (alma['CTYPE3'] == 'FREQ') & (alma['NAXIS3'] == 1)
    alma = alma[itest]
    print('Number of sources with CTYPE3 = FREQ:', len(alma))
    key=raw_input("Enter any key to continue: ")

    title = 'xmatch: Veron+2010'
    suptitle = infile
    alma = xmatch_veron2010(alma=alma,
                            title=title,
                            suptitle=suptitle)


    ProjectCode = '2011.0.00725.S'
    ProjectCode = '2012.1.00604.S'
    ProjectCode = '2013.1.00884.S'
    print('Project code:', ProjectCode)
    count = 0
    for irow, row in enumerate(alma):
        if row['FILE'].find(ProjectCode) >= 0:
            count = count + 1
            print(irow+1, count, row)

    print('count:', count)

    parse_file(alma=alma, debug=False)
    print('Source name and Project code parsed')

    SourceStats(alma, verbose=False)
    key=raw_input("Enter any key to continue: ")

    # sys.exit()

    #
    filter_radec = True
    filter_radec_rmax = 3.0
    result = xmatch_calibrators(alma=alma, calibrators=None,
                              filter_radec=filter_radec,
                              filter_radec_rmax=filter_radec_rmax,
                              plottitle=infile)
    print('Number of sources remaining after xmatch_calibrators:', len(result))
    SourceStats(result, verbose=False)
    print('Save as result1.fits')
    result.write('results1.fits', overwrite=True)

    sys.exit()


    # extract the project ID from the file
    project = []

    # table = ascii.read(infile, header_start=0, data_start=1)
    # print()
    # print(table[0])


    # table = Table.read(infile)

    table_stats('sources.fits', debug=False)
    alma.info()
    alma.info('stats')

    Image_SourceStats(alma, filename=infile, verbose=False)

    print(len(alma), 'Sources')
    SourceStats(alma, verbose=False)
    # limit to source with CUNIT3 == 'Hz' and CTYPE3 == 'FREQ'
    itest = (alma['CTYPE3'] == 'FREQ') & (alma['NAXIS3'] == 1)
    alma = alma[itest]
    print('Number of sources with CTYPE3 = FREQ:', len(alma))

    SourceStats(alma, verbose=False)
    key=raw_input("Enter any key to continue: ")

    Image_SourceStats(alma, filename=infile, verbose=True)

    # sys.exit()

    print(len(alma), 'Sources')
    SourceStats(alma, verbose=False)
    key=raw_input("Enter any key to continue: ")

    result = SourceDistribution(alma=alma, plottitle=infile,
                                xyfilter=False,
                                radecfilter=True,
                                radecfilter_radius=2.0)
    print('Number of rows in xy filtered source file:', len(result))
    result.write('result.fits', overwrite=True)

    xmatch_calibrators(alma=result, calibrators=None, plottitle=infile)

    SourceStats(result, verbose=False)

    sys.exit()

    make_ds9regionfile(alma)

    sys.exit()

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


    # newtable = Table([ra, dec], names=('ra', 'dec'))
    # outfile = 'tmp.vot'
    # newtable.write(outfile, table_id='alma_data', format='votable')
