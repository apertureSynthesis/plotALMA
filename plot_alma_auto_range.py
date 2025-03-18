import os,sys
sys.path.append('/Users/nxroth/scripts')
from astroquery.jplhorizons import Horizons
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pymodules.imtools import sigClip
from spectral_cube import SpectralCube
from regions import CirclePixelRegion, PixCoord
import astropy.units as u
from datetime import datetime, timedelta

def plot_alma(contFile,objectName,figname,figtitle,plotCont=True,lineFile=None,vlow=None,vup=None,projectedDistance=True):

    #Set some plotting options
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    #plt.style.use('dark_background')
    

    #Read in continuum file
    hcont = fits.open(contFile)
    cont = hcont[0]
    hdr = hcont[0].header

    #Read in line file if present
    if lineFile != None:
        hline = fits.open(lineFile)
        line = hline[0]
        hdr = hline[0].header

    #Get an ephemeris to retrieve orbital parameters at the time of observations
    dateObs, timeObs = hdr['DATE-OBS'].split('T')[0], hdr['DATE-OBS'].split('T')[1]
    #Find the time one minute ahead
    dObs = datetime.strptime(dateObs+' '+timeObs, '%Y-%m-%d %H:%M:%S.%f')
    deltaObs = dObs + timedelta(minutes=1)
    startTime = dObs.strftime('%Y-%m-%d %H:%M')
    endTime = deltaObs.strftime('%Y-%m-%d %H:%M')


    #Set up the astroquery
    obj = Horizons(id=objectName, id_type='designation', location='ALMA, center',
                   epochs = {'start': startTime, 'stop': endTime, 'step': '1m'})
    eph = obj.ephemerides(quantities='10,20,27', no_fragments=True, closest_apparition=True)

    df_eph = eph.to_pandas()

    delta = df_eph['delta'][0].item()*u.au
    ifrac = df_eph['illumination'][0].item()/100.
    psAng = df_eph['sunTargetPA'][0].item()*u.deg
    psAMV = df_eph['velocityPA'][0].item()*u.deg

    #Get spatial axis
    #pixel scale
    pscl = float(hdr['CDELT2'])*u.deg

    #Compute velocity axis
    if lineFile != None:
        dv = float(hdr['CDELT3'])
        v0 = float(hdr['CRVAL3'])
        p0 = int(hdr['CRPIX3'])

        v = np.arange(0,line.data.shape[0], dtype=float)*dv + v0 - (p0-1.)*dv
        v *= 1./1000.
        dv *= 1./1000.

        #Find velocity integration range
        vinds = np.where((v>=vlow) & (v<=vup))

    #Extract continuum signal in mJy
    cdata = cont.data[:,:]*1000.
    c1max = np.where(cdata == np.nanmax(cdata))

    #Find photocenter of the image
    pcen = [c1max[1][0],c1max[0][0]]
    #Clip the image to 30% of its width centered on the photocenter
    nside = int(0.15*cdata.shape[0])

    #Make sure we have an odd number of pixels (so photcenter is truly centered)
    if (2*nside %2 == 0):
        npix = 2*nside + 1
    else:
        npix = 2*nside
    contSignal = cdata[pcen[1]-nside:pcen[1]+nside,pcen[0]-nside:pcen[0]+nside]

    #Calculate the RMS of the emission-free portion of the image    
    rms1 = np.nanstd(cdata)
    masked_cont = sigClip(cdata,2)
    contrms = np.nanstd(masked_cont)
    cmax_val = cdata[c1max]
    SNR_cont = cmax_val/contrms

    #Take moment-0 in Jy km/s if lineFile is present
    #Convert to Jy/pix
    if lineFile != None:
        mom0 = line.data[vinds[0],:,:] 
        #Convert to mJy km/s
        lineSignal = np.sum(mom0,axis=0)*abs(dv)*1000
        #Find photocenter and calculate RMS of the emission-free region
        lmax = np.where(lineSignal == np.nanmax(lineSignal))
        rms1 = np.nanstd(lineSignal)
        masked_line = sigClip(lineSignal,2)
        linerms = np.nanstd(masked_line)

        lmax_val = lineSignal[lmax]
        lineSignal = lineSignal[pcen[1]-nside:pcen[1]+nside,pcen[0]-nside:pcen[0]+nside]
        SNR_line = lmax_val/linerms

    
    #Work out our image extent depending on whether we are plotting
    #in angular or projected distance units
    if projectedDistance:
        xscl = (delta.to(u.km)).value * np.tan((pscl.to(u.rad)).value)
        #Get beam properties
        bmaj = (delta.to(u.km)).value*np.tan((hdr['BMAJ']*u.deg).to(u.rad).value)
        bmin = (delta.to(u.km)).value*np.tan((hdr['BMIN']*u.deg).to(u.rad).value)
        bpa = hdr['BPA']
    else:
        xscl = (pscl.to(u.arcsec)).value
        bmaj = (hdr['BMAJ']*u.deg).to(u.arcsec).value
        bmin = (hdr['BMIN']*u.deg).to(u.arcsec).value
        
    #Beam position angle
    bpa = hdr['BPA']

    #Extent of image in desired units
    xcnt = (np.arange(0, npix, dtype=float) - nside) * xscl

    if lineFile != None:
        #Find distance from photocenter of the continuum vs. spectral line emission in pixels
        dx = (lmax[1][0] - c1max[1][0])
        dy = (lmax[0][0] - c1max[0][0])
        ndx = dx*xscl
        ndy = dy*xscl

        #Work out the pixel radius for a 10" diameter circle
        pix_as = pscl.to(u.arcsec)
        rpix = 5./pix_as.value

        #Work out pixels per beam for conversion to Jy
        barea = np.pi * hdr['BMAJ'] * hdr['BMIN'] / (4*np.log(2)) #Area of Gaussian beam
        npix_beam = barea / hdr['CDELT2']

    if plotCont:
        rms = contrms
        Signal = contSignal
        barlabel = 'Continuum Flux (mJy beam$^{-1}$)'
        slabel = '$\sigma$ = %.2f mJy beam$^{-1}$'%(rms)
    else:
        rms = linerms
        Signal = lineSignal
        barlabel = 'Integrated Flux (mJy beam$^{-1}$ km s$^{-1}$)'
        slabel = '$\sigma$ = %.2f mJy beam$^{-1}$ km s$^{-1}$'%(rms)
        dcont = np.sqrt(ndx**2+ndy**2)
        dcont_err = bmaj / min(SNR_line,SNR_cont)[0]
        delta_label = '$\delta_{cont}$ = (%d $\pm$ %d) km'%(dcont,dcont_err)

 
    #Work out which level contours we want based on the S/N of the data at the peak pixel
    #1-sigma increments up to 5-sigma
    if np.nanmax(Signal) / rms <= 5:
        clevels = np.arange(-5,6)
        
    if (np.nanmax(Signal)/rms >= 5) & (np.nanmax(Signal)/rms <= 10):
        clevels = np.array([-10, -5, -3, 0, 3, 5, 10])

    if (np.nanmax(Signal)/rms >= 10) & (np.nanmax(Signal)/rms <= 30):
        clevels = np.arange(-30,35,5) 

    if (np.nanmax(Signal)/rms >= 30) & (np.nanmax(Signal)/rms <= 100):
        clevels = np.arange(-100,110,10)

    if (np.nanmax(Signal)/rms >= 100):
        clevels = np.arange(-1000,1100,10)

    zindex = np.where(clevels == 0)
    contLevels = np.delete(clevels,zindex)
    contourLevels = [x*rms for x in contLevels]
    strs = [str(i)+'$\sigma$' for i in contLevels]

    #Create figure
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    im = ax.imshow(Signal,origin='lower',extent=[xcnt[0],-1*xcnt[0],xcnt[0],-1*xcnt[0]],interpolation='none',cmap=cm.CMRmap,vmin=-1*rms,vmax = np.nanmax(Signal))
    cont = ax.contour(Signal,levels=contourLevels,colors='white',alpha=0.85,origin='lower',extent=[xcnt[0],-1*xcnt[0],xcnt[0],-1*xcnt[0]],linewidths=1.5)
    fmt = {}
    for l,s in zip(cont.levels,strs):
        fmt[l] = s
    ax.clabel(cont,cont.levels,inline=True,fmt=fmt,fontsize=20)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, location='right')
    cbar.set_label(barlabel,fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    #Add synthesized beam
    ax.add_patch(Ellipse((xcnt[0]-0.30*xcnt[0],xcnt[0]-0.30*xcnt[0]),width=bmaj,height=bmin,angle=bpa+90.,edgecolor='white',facecolor='none',hatch='///',linewidth=2))
    
    #Axis label preferences
    ax.tick_params(axis='x',direction='in',color='white',length=7,labelsize=20)
    ax.tick_params(axis='y',direction='in',color='white',length=7,labelsize=20)
    ax.tick_params(bottom=True,top=True,left=True,right=True)
    ax.tick_params(labelleft=True,labelbottom=True,labeltop=False,labelright=False)
    if projectedDistance:
        ax.set_xlabel('Distance West (km)',fontsize=22,color='black')
        ax.set_ylabel('Distance North (km)',fontsize=22,color='black')
    else:
        ax.set_ylabel(r'$\Delta\delta$ (arcsec)',fontsize=22,color='black')
        ax.set_xlabel(r'$\Delta\alpha$ (arcsec)',fontsize=22,color='black')
    ax.set_title(figtitle,fontsize=24,fontweight='bold')


    #Add illumination geometry

    #Define geometric location of illumination geometry plots
    pwidth=xcnt[0]*0.20
    iwidth = pwidth*(2*ifrac-1)
    ac = [0.70*np.abs(xcnt[0]),-0.70*np.abs(xcnt[0])]
    r = 0.20*np.sqrt(ac[0]**2 + ac[1]**2)
    ax.add_patch(Arc((ac[0],ac[1]),height=pwidth,width=pwidth,theta1=270,theta2=90,color='yellow'))
    ax.add_patch(Arc((ac[0],ac[1]),height=pwidth,width=iwidth,theta1=90,theta2=270,color='yellow'))
    ax.add_patch(Arc((ac[0],ac[1]),height=pwidth,width=pwidth,theta1=90,theta2=270,color='yellow',linestyle='--'))

    #Solar vector
    #Convert psa and ps_amv to proper units and orientation (counterclockwise from North)
    psAng = (psAng.to(u.rad)).value - np.pi/2.
    psAMV = (psAMV.to(u.rad)).value + np.pi/2.

    ax.annotate("",xytext=(ac[0],ac[1]),xy=(ac[0]+r*np.cos(psAng),ac[1]+r*np.sin(psAng)),xycoords='data',textcoords='data',arrowprops=dict(color='white',headwidth=10,width=0.1))
    ax.text(1.01*(ac[0]+r*np.cos(psAng)),1.01*(ac[1]+r*np.sin(psAng)),'S',fontsize=25,color='white')

    #Negative heliocentric velocity vector
    ax.annotate("",xytext=(ac[0],ac[1]),xy=(ac[0]+r*np.cos(psAMV),ac[1]+r*np.sin(psAMV)),xycoords='data',textcoords='data',arrowprops=dict(color='white',headwidth=10,width=0.1))
    ax.text(1.01*(ac[0]+r*np.cos(psAMV)),1.01*(ac[1]+r*np.sin(psAMV)),'T',fontsize=25,color='white')

    #Add sigma label
    ax.text(-0.85*np.abs(xcnt[0]),0.9*np.abs(xcnt[0]),slabel,fontsize=16,color='white',fontweight='bold')

    #Add a spectral plot
    if not plotCont:
        ax.plot(0,0,marker='+',color='black',markersize=18,markeredgewidth=3)
        circPix = CirclePixelRegion(center=PixCoord(x=pcen[1],y=pcen[0]),radius=rpix)
        cube = SpectralCube.read(lineFile)
        subcube = cube.subcube_from_regions([circPix])
        sum2 = subcube.sum(axis=(1,2)) / npix_beam
        inset = fig.add_axes([0.60,0.67,0.13,0.13])
        plt.setp(inset.spines.values(),color='white')
        plt.setp([inset.get_xticklines(), inset.get_yticklines(), inset.get_xticklabels()],color='white')
        inset.set_facecolor('black')
        inset.plot(v,sum2,drawstyle='steps-mid',color='white',linewidth=1)
        inset.tick_params(axis='x',direction='in',labelsize=18)
        inset.tick_params(labelleft=False,labelbottom=True,left=False)
        inset.set_xlabel('v (km s$^{-1}$)',color='white',fontsize=18)
        inset.set_xlim(-2,2)

        #Add delta label
        ax.text(-0.85*np.abs(xcnt[0]),0.80*np.abs(xcnt[0]),delta_label,fontsize=16,color='white',fontweight='bold')

    plt.tight_layout()
    plt.savefig(figname)
    plt.show()

def main():
    plot_alma(contFile='ER61.cont.clean1.image.fits',objectName='C/2015 ER61',figname='ER61.Apr11.cont.jpg',
              figtitle='C/2015 ER61 345 GHz Continuum\n UT 2017 April 11',projectedDistance=True)
    plot_alma(contFile='ER61.cont.clean1.image.fits',objectName='C/2015 ER61',figname='ER61.Apr11.HCN.jpg',
              figtitle='C/2015 ER61 HCN (J=4-3)\nUT 2017 April 11',plotCont=False,
              lineFile='ER61.HCN.4-3.clean1.image.fits',vlow=-0.8,vup=0.8,projectedDistance=True)
    
main()
