import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import astropy.units as u
from astropy.io import fits
from astroquery.jplhorizons import Horizons
from datetime import datetime, timedelta
from matplotlib.patches import Ellipse, Arc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from spectral_cube import SpectralCube
from regions import CirclePixelRegion, PixCoord

class plotALMA(object):

    def __init__(self,contFile,objectName,figName,figTitle,plotCont=True,lineFile=None,vlow=None,vup=None,projectedDistance=True):

        self.contFile = contFile
        self.objectName = objectName
        self.figName = figName
        self.figTitle = figTitle
        self.plotCont = plotCont
        self.lineFile = lineFile
        if vlow != None:
            self.vlow = vlow.to(u.km/u.s).value
        else:
            self.vlow = None
        if vup != None:
            self.vup = vup.to(u.km/u.s).value
        else:
            self.vup = None
        self.projectedDistance = projectedDistance

    def _read_files(self):
        
        #Read in continuum file
        hcont = fits.open(self.contFile)
        cont = hcont[0]
        self.hdr = hcont[0].header
        self.contData = cont.data
        self.lineData = None

        #Read in line file if present
        if self.lineFile != None:
            hline = fits.open(self.lineFile)
            line = hline[0]
            self.hdr = hline[0].header
            self.lineData = line.data


    def _get_eph(self):

        #Find the time of the observations from the header
        dateObs, timeObs = self.hdr['DATE-OBS'].split('T')[0], self.hdr['DATE-OBS'].split('T')[1]

        #Find the time one minute ahead so we can generate a JPL/HORIZONS ephemeris

        dObs = datetime.strptime(dateObs+' '+timeObs, '%Y-%m-%d %H:%M:%S.%f')
        deltaObs = dObs + timedelta(minutes=1)
        startTime = dObs.strftime('%Y-%m-%d %H:%M')
        endTime = deltaObs.strftime('%Y-%m-%d %H:%M')

        #Run the query to JPL/HORIZONS and store the results in a dataframe
        obj = Horizons(id=self.objectName, id_type='designation', location='ALMA, center',
                    epochs = {'start': startTime, 'stop': endTime, 'step': '1m'})
        eph = obj.ephemerides(quantities='10,20,27', no_fragments=True, closest_apparition=True)

        self.df_eph = eph.to_pandas()
    
    # Return sigma-clipped (masked) numpy array (iterate until new_RMS <= 1.01*previous_RMS). 
    # posneg = 'positive': Vlaues > nSigma * RMS are masked
    # posneg = 'negative': Vlaues < (-nSigma) * RMS are masked
    # posneg = 'both' means both
    def sigClip(self,img,nSigma,DELTARMS=1.01,posneg='both'):
        sigma = np.nanstd(img)
        sigma0 = sigma*2.
        imgmasked = img
        while sigma0/sigma > DELTARMS:
            if posneg == 'positive':
                imgmasked = np.ma.masked_where((img-np.nanmean(imgmasked))>nSigma*sigma,img)
            if posneg == 'negative':
                imgmasked = np.ma.masked_where((img-np.nanmean(imgmasked))<(-nSigma)*sigma,img)
            if posneg == 'both':
                imgmasked = np.ma.masked_where(np.abs(img-np.nanmean(imgmasked))>nSigma*sigma,img)       
            sigma0 = sigma
            sigma=np.nanstd(imgmasked)
            if imgmasked.mask.all(): print("All data points have been rejected, cannot continue")
        return imgmasked
    
    def _generate_maps(self):

        #Get beam properties and ephemeris information
        #Geocentric distance, illumination fration, position angle of the Sun-comet vector,
        #position angle of the heliocentric velocity vector
        self.delta = self.df_eph['delta'][0].item()*u.au
        self.ifrac = self.df_eph['illumination'][0].item()/100.
        self.psAng = self.df_eph['sunTargetPA'][0].item()*u.deg
        self.psAMV = self.df_eph['velocityPA'][0].item()*u.deg

        #Spatial pixel scale
        self.pscl = float(self.hdr['CDELT2'])*u.deg

        #Compute velocity axis (km/s) if lineFile is present
        if self.lineFile != None:
            dv = float(self.hdr['CDELT3'])
            v0 = float(self.hdr['CRVAL3'])
            p0 = int(self.hdr['CRPIX3'])

            self.v = np.arange(0,self.lineData.shape[0], dtype=float)*dv + v0 - (p0-1.)*dv
            self.v *= 1./1000.
            dv *= 1./1000.

            #Find velocity integration range
            vinds = np.where((self.v>=self.vlow) & (self.v<=self.vup))

        #Extract continuum signal in mJy, find photocenter, and calculate the image noise
        cdata = self.contData*1000.
        cmax = np.where(cdata == np.nanmax(cdata))
        self.pcen = [cmax[1][0],cmax[0][0]]

        #Calculate the rms and SNR
        masked_cont = self.sigClip(cdata,2)
        contRMS = np.nanstd(masked_cont)
        SNR_cont = cdata[cmax] / contRMS

        #Clip the image to 25% of its width compared to the photocenter
        nside = int(0.25*self.contData.shape[0])
        #Make sure there is an odd number of pixels so the photocenter is truly centered
        if (2*nside %2 == 0):
            npix = 2*nside + 1
        else:
            npix = 2*nside
        contSignal = cdata[self.pcen[1]-nside:self.pcen[1]+nside,self.pcen[0]-nside:self.pcen[0]+nside]

        #Take moment-0 in mJy km/s if lineFile is present
        #Convert to Jy/pix
        if self.lineFile != None:
            mom0 = self.lineData[vinds[0],:,:] 
            #Convert to mJy km/s
            lineSignal = np.sum(mom0,axis=0)*abs(dv)*1000
            #Find photocenter and calculate RMS of the emission-free region
            lmax = np.where(lineSignal == np.nanmax(lineSignal))
            masked_line = self.sigClip(lineSignal,2)
            lineRMS = np.nanstd(masked_line)
            #SNR of the image
            SNR_line = lineSignal[lmax] / lineRMS

            #Find the distance from the photocenter of the continuum to that of the line in pixels
            dx = (lmax[1][0] - cmax[1][0])
            dy = (lmax[0][0] - cmax[0][0])

            #Crop lineSignal
            lineSignal = lineSignal[self.pcen[1]-nside:self.pcen[1]+nside,self.pcen[0]-nside:self.pcen[0]+nside]

        #Work out our image extent depending on whether we are plotting
        #in angular or projected distance units
        if self.projectedDistance:
            xscl = (self.delta.to(u.km)).value * np.tan((self.pscl.to(u.rad)).value)
            #Size and orientation of the synthesized beam
            self.bmaj = (self.delta.to(u.km)).value*np.tan((self.hdr['BMAJ']*u.deg).to(u.rad).value)
            self.bmin = (self.delta.to(u.km)).value*np.tan((self.hdr['BMIN']*u.deg).to(u.rad).value)
            self.bpa = self.hdr['BPA']
        else:
            xscl = (self.pscl.to(u.arcsec)).value
            #Size and orientation of the synthesized beam
            self.bmaj = (self.hdr['BMAJ']*u.deg).to(u.arcsec).value
            self.bmin = (self.hdr['BMIN']*u.deg).to(u.arcsec).value
            self.bpa = self.hdr['BPA']

        #Spatial axis of the image in desired units along with extent of future image
        self.xcnt = (np.arange(0, npix, dtype=float) - nside) * xscl

        #Set final details for plotting
        if self.plotCont:
            self.rms = contRMS
            self.Signal = contSignal
            #Colorbar label and units
            self.barlabel = 'Continuum Flux (mJy beam$^{-1}$)'
            self.slabel = '$\sigma$ = %.2f mJy beam$^{-1}$'%(self.rms)
            #Other labels
            self.delta_label = None
        else:
            self.rms = lineRMS
            self.Signal = lineSignal
            self.barlabel = 'Integrated Flux (mJy beam$^{-1}$ km s$^{-1}$)'
            self.slabel = '$\sigma$ = %.2f mJy beam$^{-1}$ km s$^{-1}$'%(self.rms)
            #Offset of continuum vs line in these units
            ndx = dx*xscl
            ndy = dy*xscl 
            dcont = np.sqrt(ndx**2+ndy**2)
            dcont_err = self.bmaj / min(SNR_line,SNR_cont)[0]
            if self.projectedDistance:
                self.delta_label = '$\delta_{{cont}}$ = ({:d} $\pm$ {:d}) km'.format(dcont,dcont_err)
            else:
                self.delta_label = '$\delta_{{cont}}$ = ({:.2f} $\pm$ {:.2f}) arcsec'.format(dcont,dcont_err)
    
    def _make_plots(self):

        #Set some plotting options
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['mathtext.default'] = 'regular'
        matplotlib.rcParams['font.weight'] = 'bold'
        matplotlib.rcParams['axes.labelweight'] = 'bold'

        #Work out which level contours we want based on the S/N of the data at the peak pixel
        #1-sigma increments up to 5-sigma
        if np.nanmax(self.Signal) / self.rms <= 5:
            clevels = np.arange(-5,6)
            
        if (np.nanmax(self.Signal)/self.rms >= 5) & (np.nanmax(self.Signal)/self.rms <= 10):
            clevels = [-10, -5, -3, 0, 3, 5, 10]

        if (np.nanmax(self.Signal)/self.rms >= 10) & (np.nanmax(self.Signal)/self.rms <= 30):
            clevels = np.arange(-30,35,5) 

        if (np.nanmax(self.Signal)/self.rms >= 30) & (np.nanmax(self.Signal)/self.rms <= 100):
            clevels = np.arange(-100,110,10)

        if (np.nanmax(self.Signal)/self.rms >= 100):
            clevels = np.arange(-1000,1100,10)

        zindex = np.where(np.array(clevels) == 0)
        contLevels = np.delete(clevels,zindex)
        contourLevels = [x*self.rms for x in contLevels]
        strs = [str(i)+'$\sigma$' for i in contLevels]   

        #Create figure
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        fig.subplots_adjust(hspace=0.20,wspace=0.20)
        #Make pixels outside the field black
        current_map = matplotlib.colormaps.get_cmap(cm.CMRmap)
        current_map.set_bad(color='black')
        #Extent of image
        extent = [self.xcnt[0],-1*self.xcnt[0],self.xcnt[0],-1*self.xcnt[0]]
        im = ax.imshow(self.Signal,origin='lower',extent=extent,interpolation='none',cmap=cm.CMRmap,vmin=-1*self.rms,vmax = np.nanmax(self.Signal))

        #Add contours and labels
        cont = ax.contour(self.Signal,levels=contourLevels,colors='white',alpha=0.85,origin='lower',extent=extent,linewidths=1.5)
        fmt = {}
        for l,s in zip(cont.levels,strs):
            fmt[l] = s
        ax.clabel(cont,cont.levels,inline=True,fmt=fmt,fontsize=20)

        #Add colorbar and labels
        cbar = plt.colorbar(im)
        cbar.set_label(self.barlabel,fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        #Add synthesized beam
        ax.add_patch(Ellipse((0.70*self.xcnt[0],0.70*self.xcnt[0]),width=self.bmaj,height=self.bmin,angle=self.bpa+90.,edgecolor='white',facecolor='none',hatch='///',linewidth=2))

        #Set axis labels and title
        ax.tick_params(axis='x',direction='in',color='white',length=7,labelsize=20)
        ax.tick_params(axis='y',direction='in',color='white',length=7,labelsize=20)
        ax.tick_params(bottom=True,top=True,left=True,right=True)
        ax.tick_params(labelleft=True,labelbottom=True,labeltop=False,labelright=False)
        if self.projectedDistance:
            ax.set_xlabel('Distance West (km)',fontsize=22,color='black')
            ax.set_ylabel('Distance North (km)',fontsize=22)
        else:
            ax.set_xlabel(r'$\Delta\delta$ (arcsec)')
            ax.set_ylabel(r'$\Delta\alpha$ (arcsec)')
        ax.set_title(self.figTitle,fontsize=24,fontweight='bold')

        #Add illumination geometry
        #Quantities for positioning the orientation of the illumination geometry and other labels
        pwidth=self.xcnt[0]*0.20
        iwidth = pwidth*(2*self.ifrac-1)
        ac = [0.70*np.abs(self.xcnt[0]),-0.70*np.abs(self.xcnt[0])]
        r = 0.20*np.sqrt(ac[0]**2 + ac[1]**2)
        ax.add_patch(Arc((ac[0],ac[1]),height=pwidth,width=pwidth,theta1=270,theta2=90,color='yellow'))
        ax.add_patch(Arc((ac[0],ac[1]),height=pwidth,width=iwidth,theta1=90,theta2=270,color='yellow'))
        ax.add_patch(Arc((ac[0],ac[1]),height=pwidth,width=pwidth,theta1=90,theta2=270,color='yellow',linestyle='--'))

        #Solar vector
        #Convert psa and ps_amv to proper units and orientation (counterclockwise from North)
        self.psAng = (self.psAng.to(u.rad)).value - np.pi/2.
        self.psAMV = (self.psAMV.to(u.rad)).value + np.pi/2.

        ax.annotate("",xytext=(ac[0],ac[1]),xy=(ac[0]+r*np.cos(self.psAng),ac[1]+r*np.sin(self.psAng)),xycoords='data',textcoords='data',arrowprops=dict(color='white',headwidth=10,width=0.1))
        solar_vector = ax.text(1.01*(ac[0]+r*np.cos(self.psAng)),1.01*(ac[1]+r*np.sin(self.psAng)),'S',fontsize=25,color='white')
        solar_vector.set_path_effects([pe.withStroke(linewidth=4,foreground="black")])

        #Tail vector
        ax.annotate("",xytext=(ac[0],ac[1]),xy=(ac[0]+r*np.cos(self.psAMV),ac[1]+r*np.sin(self.psAMV)),xycoords='data',textcoords='data',arrowprops=dict(color='white',headwidth=10,width=0.1))
        tail_vector = ax.text(1.01*(ac[0]+r*np.cos(self.psAMV)),1.01*(ac[1]+r*np.sin(self.psAMV)),'T',fontsize=25,color='white')
        tail_vector.set_path_effects([pe.withStroke(linewidth=4,foreground="black")])

        #Add sigma label
        sigma_text = ax.text(-0.85*np.abs(self.xcnt[0]),0.9*np.abs(self.xcnt[0]),self.slabel,fontsize=16,color='white',fontweight='bold')
        sigma_text.set_path_effects([pe.withStroke(linewidth=4,foreground="black")])

        #Add a spectral plot
        if not self.plotCont:
            #Work out number of pixels for a 10" diameter circle for spectral extract
            #Convert pixel scale to arcsec
            pix_as = self.pscl.to(u.arcsec)
            rpix = 5. / pix_as.value

            #Work out angular size of beam
            #Area of Gaussian beam
            omega = np.pi*self.hdr['BMAJ']*self.hdr['BMIN'] / (4*np.log(2))
            #Number of pixels in beam
            npix_beam = omega / (self.hdr['CDELT2']**2)

            #Add spectral plot inset
            ax.plot(0,0,marker='+',color='black',markersize=18,markeredgewidth=3)
            circPix = CirclePixelRegion(center=PixCoord(x=self.pcen[0],y=self.pcen[1]),radius=rpix)
            cube = SpectralCube.read(self.lineFile)
            subcube = cube.subcube_from_regions([circPix])
            spec = subcube.sum(axis=(1,2)) / npix_beam
            inset = fig.add_axes([0.60,0.67,0.13,0.13])
            plt.setp(inset.spines.values(),color='white')
            plt.setp([inset.get_xticklines(), inset.get_yticklines(), inset.get_xticklabels()],color='white')
            inset.set_facecolor('black')
            inset.plot(self.v,spec,drawstyle='steps-mid',color='white',linewidth=1)
            inset.tick_params(axis='x',direction='in',labelsize=18)
            inset.tick_params(labelleft=False,labelbottom=True,left=False)
            inset.set_xlabel('v (km s$^{-1}$)',color='white',fontsize=18)
            inset.set_xlim(-5,5)

            #Add delta label for continuum vs. line
            delta_text = ax.text(-0.85*np.abs(self.xcnt[0]),0.80*np.abs(self.xcnt[0]),self.delta_label,fontsize=16,color='white',fontweight='bold')
            delta_text.set_path_effects([pe.withStroke(linewidth=4,foreground="black")])

        plt.savefig(self.figName)
        plt.show()

    def __call__(self):
        self._read_files()
        self._get_eph()
        self._generate_maps()
        self._make_plots()


        



