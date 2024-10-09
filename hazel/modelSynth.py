from hazel.chromosphere import Hazel_atmosphere
from hazel.photosphere import SIR_atmosphere
from hazel.parametric import Parametric_atmosphere
from hazel.stray import Straylight_atmosphere
from hazel.configuration import Configuration
from hazel.io import Generic_output_file
from collections import OrderedDict
from hazel.codes import hazel_code, sir_code
from hazel.spectrum import Spectrum
from hazel.transforms import transformed_to_physical, physical_to_transformed, jacobian_transformation
from hazel.util import i0_allen
import hazel.util
import numpy as np
import copy
import os
from pathlib import Path
import scipy.stats
import scipy.special
import scipy.signal
#import scipy.linalg
import scipy.optimize
import warnings
import logging
import sys
import matplotlib.pyplot as plt #EDGAR: Im placing plotting routines here, but is a bit ugly


labdic = {'z1':r'$\mathrm{z \, [Mm]}$',
        'tt':r'$\mathrm{T\,[kK]}$','tit':r'$\mathrm{Temperature}$',
        'vdop':r'$\mathrm{V^{dop}_z \,[D.u.]}$',
        'b':r'$\mathrm{|B| \,[G]}$','tb':r'$\mathrm{\theta_B \,[Degrees]}$',
        'cb':r'$\mathrm{\chi_B \,[Degrees]}$',
        'frokq':r'$\{\rho}^K_Q/\rho^0_0$',
        'xx0':r'$\mathrm{\lambda-\lambda_0[{\AA}]}$',  
        'xx':r'$\mathrm{\lambda[{\AA}]}$',  
        'iic':r'$\mathrm{I/I_c}$','qi':r'$\mathrm{Q/I}$','ui':r'$\mathrm{U/I}$', 
        'vi':r'$\mathrm{V/I}$','qic':r'$\mathrm{Q/I_c}$','uic':r'$\mathrm{U/I_c}$', 'vic':r'$\mathrm{V/I_c}$', 
        'epsi':r'$\mathrm{\epsilon_I}$','epsq':r'$\mathrm{\epsilon_Q}$','epsu':r'$\mathrm{\epsilon_U}$',
        'epsv':r'$\mathrm{\epsilon_V}$','etai':r'$\mathrm{\eta_I}$','etaq':r'$\mathrm{\eta_Q}$',
        'etau':r'$\mathrm{\eta_U}$','etav':r'$\mathrm{\eta_V}$','rhoq':r'$\mathrm{\rho_Q}$',
        'rhou':r'$\mathrm{\rho_U}$','rhov':r'$\mathrm{\rho_V}$'
        }


def mylab(lab):
    return labdic.get(lab,lab) #return the input keyword string if lab is not in labdic

'''
def latex1(str):
    return r'$\mathrm{z \,{0} [Mm]}$'.format(str) 
'''
def exact_parabols(xax,P_ini,P_end,aval=0.2):
    #P_ini-->[p1x,p1y]
    #P_end-->[p2x,p2y]
    #define the function, the a parameter allows different parabols
    fn= lambda x,a,pa,pb : (a*(x-pb[0])+(pb[1]-pa[1])/(pb[0]-pa[0]))*(x-pa[0])+pa[1]
    return fn(xax,aval,P_ini,P_end)

def exp_3points(xax,p0,p1,p2):#needs 3 points to be fit 
    fn = lambda x,a,b,c : a + b*np.exp(c * x)
    return fn(xax,p0,p1,p2)


def exp_2points(xax, xl, yl):#fit with 2 points
    c=np.log(yl[0]/yl[1])/(xl[0]-xl[1])
    b=yl[1]*np.exp(-c*xl[1])
    return b*np.exp(c * xax)


__all__ = ['ModelRT']

class ModelRT(object):
    def __init__(self, config=None, mode='synthesis', atomfile='helium.atom',apmosekc='1110', dcol=None,
        extrapars=None, verbose=0, debug=False,rank=0, randomization=None, plotit=False, root=''):
        '''
        edic={'Atompol':1,'MO effects':1,'Stim. emission':1, 'Kill coherences':0,'dcol':[0.,0.,0.]}
        dcol=[0.,0.,0.],extrapars=edic,'helium.atom' and verbose =0 are default

        ap-mo-se-nc --> atompol, magopt, stimem, nocoh = 1, 1, 1, 0  
        #nocoh =0 includes all coherences, set to level number to deactive cohs in it

        dcol=[0.0,0.0,0.0]  #D^(K=1 and 2)=delta_collision, D^(K=1),D^(K=2)

        synMode = 5 #Opt.thin (0), DELOPAR (3),  EvolOp (5)
        '''

        #check/init mutable keywords at starting to avoid having memory of them among function/class calls
        if extrapars is None:extrapars={}
        if dcol is None:dcol=[0.,0.,0.]

        np.random.seed(123)
        if (rank != 0):
            return

        #EDGAR: dictionary of dictionaries with possible atoms and lines with their indexes for HAZEL atmospheres and spectra        
        #A variation of this dict to add more atoms and lines required to do similar changes in 
        #multiplets dict in general_atmosphere object (atmosphere.py). 
        #Now everything is encapsulated here in atomsdic.
        #From here we could also extract the number of transitions as len(self.multipletsdic[atom]).
        #However that would be a redundancy with respect to the data in the atom files,
        #and furthermore ntrans must be read at the very beginning before initializing j10, 
        #so the choice is to read ntrans from atom file: 
        #io_py.f90 ->hazel_py.f90 -> hazel_code.pyx -> init routine here in model.py

        self.atomfile=atomfile #memory for mutations

        self.atomsdic={'helium':{'10830': 1, '3888': 2, '7065': 3,'5876': 4},
            'sodium':{'5895': 1, '5889': 2}} 
        
        self.multipletsdic={'helium':{'10830': 10829.0911, '3888': 3888.6046, '7065': 7065.7085, '5876': 5875.9663},
                            'sodium':{'5895': 5895.924, '5889': 5889.95}}

        self.atwsdic={'helium':4.,'sodium':22.9897,'calcium':40.08} 


        self.apmosekc=apmosekc #memory for mutations, must be before get_apmosekcl
        self.dcol=dcol #memory for mutations, must be before get_apmosekcl

        #apmosekcL is List whose last element is other list with dcol
        self.apmosekcl=self.get_apmosekcl(apmosekc,dcol,extrapars) 


        #Dictio of minimum, default, and maximum values for all possible pars in Hazel atmosphere 
        #this could be conflicting with the use of "ranges", but such ranges seem to be applied only
        #in inversion and I dont see their actual set up to numeric meaningful values anywhere.
        #although 3 last pars are lists of ntr elements themselves, the limit is common to all of elements
        #use dmm in set_B_vals or make it global:  
        self.limB=4000.
        self.dmm={'Bx': [0.,100.,self.limB], 'By': [0.,100.,self.limB], 'Bz': [0.,100.,self.limB], \
            'B': [0.,100.,self.limB], 'thB': [0.,0.,180.], 'phB': [-360.,0.,360.], \
                'tau':[0.,1.,20.],'v':[-50.,0., 50.],'deltav':[0.5,2.,15.], \
                'beta':[0.,1.,10.],'a':[0.,0.1,10.],'ff':[0.,1.,1.], \
                'j10':[0.,0.,1.],'j20f':[0.,1.,1000.],'nbar':[0.,1.,1.]}
        self.dlims=None
        self.pars2D=None        
        self.B2D, self.hz = None, None 


        self.plotit=plotit
        self.plotscale=3
        #Set up figures and axes: stokes,optical coeffs,mutations,atmosphere 
        self.labelf1,self.labelf2,self.labelf3,self.labelf4='1','2','3','4'
        self.f1,self.f2,self.f3,self.f4=None,None,None,None
        self.ax1,self.ax2,self.ax3,self.ax4= None,None,None,None
        
        self.lock_fractional=None

        #synthesis methods to be implemented
        self.methods_dicT={0:'Emissivity',1:'Delo1',2:'Delo2',3:'Hermite',4:'Bezier',5:'EvolOp',6:'Guau'} 
        self.methods_dicS={'Emissivity':0,'Delo1':1,'Delo2':2,'Hermite':3,'Bezier':4,'EvolOp':5,'Guau':6} 
        self.methods_list=[ss for ss,tt in self.methods_dicS.items()] #list with only the names
        
        self.synmethod=5 #5 is default and can be changed by add_spectrum and /or by synthesize.
        #------------------------------------
        
        self.chromospheres = []
        self.chromospheres_order = []
        self.atmospheres = {}
        self.order_atmospheres = []        
        self.configuration = None
        self.n_cycles = 1
        self.spectrum = {}  #EDGAR:self.spectrum was initialized as [] and now as {}
        self.topologies = {}#[] EDGAR: before it was a list, now it is a dictionary
        self.atms_in_spectrum={} #EDGAR: of the kind -> {'sp1': string_with_atmosphere_order_for_sp1}
        
        #default mu where Allen continuum shall be taken for normalizing Stokes output
        #the actual value is set when calling synthesize_spectrum
        self.muAllen=1.0 

        self.working_mode = 'synthesis' #=mode    hardcoded to synthesis
        self.pixel = 0
        self.debug = debug
        self.use_analytical_RF_if_possible = False
        self.nlte_available = False
        self.use_nlte = False
        self.root = root

        self.epsilon = 1e-2
        self.svd_tolerance = 1e-8
        self.step_limiter_inversion = 1.0
        self.backtracking = 'brent'
        
        self.verbose = verbose
        
        self.logger = logging.getLogger("model")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        ch = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set randomization
        if (randomization is None):
            self.n_randomization = 1        
        else:
            self.n_randomization = randomization

        if (self.verbose >= 1):
            self.logger.info('Hazel2 Experimental')
        
        if ('torch' in sys.modules and 'torch_geometric' in sys.modules):
            if (self.verbose >= 1):
                self.logger.info('PyTorch and PyTorch Geometric found. NLTE for Ca II is available')
            self.nlte_available = True

        #We initialize pyhazel (and set self.ntrans) before calling add_chromosphere in use_configuration 
        #in order to setup self.ntrans before defining j10 length.
        #before these changes, the Hazel init was done after the next if..else.  
        self.ntrans=hazel_code._init(atomfile,verbose)   #EDGAR
        
        if (config is not None):
            if (self.verbose >= 1):
                self.logger.info('Using configuration from file : {0}'.format(config))
            self.configuration = Configuration(config)

            #EDGAR:n_chromospheres is set here.
            self.use_configuration(self.configuration.config_dict) 


    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    def __str__(self):
        tmp = ''
        for l, par in self.__dict__.items():
            if (l != 'LINES'):
                tmp += '{0}: {1}\n'.format(l, par)
        return tmp

    def get_apmosekcl(self,apmosekc,dcol,extrapars):
        '''
        Parameters in extrapars overwrite those in apmosekc.
        apmosekc and extrapars are mostly redudant on purpose. 
        apmosekc is much more compact but extrapars is there for who requires more readibility
        extrapars={'Atompol':1,'MO effects':1,'Stim. emission':1, 'Kill coherences':0,'dcol':[0.,0.,0.]}
        '''
        apmosekcl = [int(x) for x in apmosekc] #now is a list of atompol,magopt,stimem,nocoh
        for elem in apmosekcl[0:3]:
            if (elem != 0) and (elem!=1):raise Exception("ERROR: apmosekc first values can only be zeros or ones")
        #EDGAR: we should also check that the nocoh value does not go beyond number of levels(pending)

        for kk,keyw in enumerate(['Atompol','MO effects','Stim. emission','Kill coherences']):
            if (keyw in extrapars) and (extrapars[keyw]!=apmosekcl[kk]):
                apmosekcl[kk]=extrapars[keyw]        
                if (keyw!='Kill coherences') and (extrapars[keyw] != 0) and (extrapars[keyw]!=1):
                    raise Exception("ERROR: firsts parameters in extrapars can only be zeros or ones")    
        
        if ('dcol' in extrapars):apmosekcl.append(extrapars['dcol'])
        else:apmosekcl.append(dcol)

        return apmosekcl

    '''
    #EDGAR: this routine will not work if the backend is not compatible. 
    #I use MacOs backend, whose canvas does not have the window attribute.

    def move_figure(self,f, x, y):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        print(backend)
        if backend == 'TkAgg':
            f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            f.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            f.canvas.manager.window.move(x, y)
    '''

    def setup_set_figure(self,fignum,scale=3,tf=4,xy=[]):
        '''Centralize control of text and figure scaling, number of plots, labelsm ad positioning.
        This subroutine is not used when a plotting routine manages its own plotting parameters.
        Returns pscale for those plotting routines that need it'''
        
        #Centralize control of text and figure scaling
        #font = {'family' : 'normal','weight' : 'bold', 'size'   : 22}
        #matplotlib.rc('font', **font)
        pscale=self.plotscale
        if scale!=pscale:pscale=scale
        plt.rcParams.update({'font.size': pscale*tf})

        if fignum==self.labelf1:
            #plt.close(self.labelf1)  
            self.f1, self.ax1 = plt.subplots(2, 2,figsize=(pscale*tf,pscale*tf),label=self.labelf1)  
            self.ax1 = self.ax1.flatten()
        if fignum==self.labelf2:tbd=1
        if fignum==self.labelf3:tbd=1
        if fignum==self.labelf4:#ready to be usd but not yet in use
            self.f4, self.ax4 = plt.subplots(nrows=xy[0], ncols=xy[1],figsize=(pscale*2,pscale*2.5),label=self.labelf4)  
            self.ax4 = self.ax4.flatten()

        #positioning of the figures pending:
        #start_x, start_y, dx, dy = self.f1.figbbox.extents
        #self.move_figure(self.f1,505,500)

        return pscale

    def reshow_this(self,fig):
        fig.canvas.manager.destroy()#first close the canvas just in case is still open
        newfig = plt.figure(label=fig.get_label())
        new_manager = newfig.canvas.manager
        new_manager.canvas.figure = fig #associate self.fX (i.e., fig) to the new canvas
        fig.set_canvas(new_manager.canvas)
        fig.set_figwidth(fig.get_figwidth())
        fig.set_figheight(fig.get_figheight())
        fig.set_visible(True)


    def reshow(self,fig):
        '''reshow a closed figure creating a newfig and using its manager to display "fig"
        Example: m1.reshow(m1.f1) or m1.reshow('1'), m1.reshow('all'). '''
        #we dont use manager.close() but manager.destroy() because destroy allows reshowing
        #but close dont. To definitely close the figure use remove_fig() below.
        
        ptrs=[self.f1,self.f2,self.f3,self.f4]
        
        if fig=='all':
            for fig in ptrs:
                if fig is not None:self.reshow_this(fig)   
        else:
            for elem in ['1','2','3','4']: 
                if fig==elem:fig=ptrs[int(elem)-1]

            self.reshow_this(fig)

    def toggle_visible(self, fig,visible=True):
        #if invisible, make it visible, or viceversa
        fig.set_visible(not fig.get_visible()) 
        plt.draw()
        
                       
    def remove_fig(self,obj,fig):
        '''
        Any necessary operations to remove figures.
        Close and set to None all (ghost) figures of the new model object used to create mutations
        We use getattr and setattr to update the original object, not a dummy copy
        
        plt.close(X) X can be:
        *None*: the current figure
        .Figure: the given .Figure instance
        int: a figure number
        str: a figure name
        'all': all figures.
        '''
        if getattr(obj,fig) is not None:
            plt.close(getattr(obj,fig))#plt.close(fig)
            setattr(obj,fig, None) #equivalent to obj.fig = None 


    def fractional_polarization(self,sp,scale=3,tf=4,ax=None,lab=['iic','qi','ui','vi']): 
        '''
        Compares fractional and not fractional polarization.
        This routine is valid when fractional polarization is not implemented in synthesize_spectrum
        '''
        if type(sp) is not hazel.spectrum.Spectrum:sp=self.spectrum[sp]

        self.setup_set_figure('Free Fig 1',scale=scale,tf=tf)
        
        if ax is None:
            f, ax = plt.subplots(nrows=2, ncols=2,figsize=(pscale*2,pscale*2),label='Fractional Polarization')  
            ax = ax.flatten()

        labs=[r'$P/I_c$',r'$P/I$']
        
        for i in range(4):
            if i==0:
                ax[i].plot(sp.wavelength_axis, sp.stokes[i,:],color='r')
            else:
                l1,=ax[i].plot(sp.wavelength_axis, sp.stokes[i,:],color='r')
                l2,=ax[i].plot(sp.wavelength_axis, sp.stokes[i,:]/sp.stokes[0,:],color='k')
                
            ax[i].set_title(mylab(lab[i]))
            if i>1:ax[i].set_xlabel(mylab('xx'))
        
        #f.legend(tuple(lines), tuple(labs), loc=(0.1,0.1), bbox_to_anchor=(0.1, 0.3))
        plt.gcf().legend(tuple([l1,l2]), tuple(labs))#loc=(0.1,0.1), bbox_to_anchor=(0.1, 0.3))

        plt.tight_layout()
        plt.show()
        
        return 


    def plot_stokes(self,sp,scale=3,tf=2,fractional=False,lab=None): 
        '''
        Routine called by synthesize to plot stokes profiles either fractional 
        or normalized to continuum 
        '''
        if fractional:lab=['iic','qi','ui','vi']
        else:lab=['iic','qic','uic','vic']
        
        if type(sp) is not hazel.spectrum.Spectrum:sp=self.spectrum[sp]
        
        if self.ax1 is None:
            self.setup_set_figure(self.labelf1,scale=scale,tf=tf)
            '''
            pscale=self.plotscale
            if scale!=pscale:pscale=scale
            plt.rcParams.update({'font.size': pscale*tf}) 
            self.f1, self.ax1 = plt.subplots(2, 2,figsize=(pscale*tf,pscale*tf),label=self.labelf1)  
            self.ax1 = self.ax1.flatten()
            '''
        else:#if the window was created but was closed reshow it 
            if self.f1.canvas.manager.get_window_title() is None:self.reshow(self.f1)

        for i in range(4): 
            if i ==0:
                self.ax1[i].plot(sp.wavelength_axis, sp.stokes[i,:])
            else:
                if fractional:self.ax1[i].plot(sp.wavelength_axis, sp.stokes[i,:]/sp.stokes[0,:])
                else:self.ax1[i].plot(sp.wavelength_axis, sp.stokes[i,:])
            
            self.ax1[i].set_title(mylab(lab[i]))#,size=8 + 0.7*pscale)
            if i>1:self.ax1[i].set_xlabel(mylab('xx'))#,size=8 +0.7*pscale)#,labelpad=lp)
        
        plt.tight_layout()
        plt.show()
        
        return 


    def plot_stokes_direct(self,sp,scale=3,tf=4,lab=None): 
        '''
        This plots stokes profiles normalized to continuum directly from spectrum object
        without considering fractional polarization 
        '''
        if lab is None:lab=['iic','qic','uic','vic']

        if type(sp) is not hazel.spectrum.Spectrum:sp=self.spectrum[sp]

        if self.ax1 is None:
            self.setup_set_figure(self.labelf1,scale=scale,tf=tf)
            '''
            pscale=self.plotscale
            if scale!=pscale:pscale=scale
            plt.rcParams.update({'font.size': pscale*tf})
            self.f1, self.ax1 = plt.subplots(2, 2,figsize=(pscale*tf,pscale*tf),label=self.labelf1)  
            self.ax1 = self.ax1.flatten()
            '''
        for i in range(4): 
            self.ax1[i].plot(sp.wavelength_axis, sp.stokes[i,:])
            self.ax1[i].set_title(mylab(lab[i]))#,size=8 + 0.7*pscale)
            if i>1:self.ax1[i].set_xlabel(mylab('xx'))#,size=8 +0.7*pscale)#,labelpad=lp)
        
        plt.tight_layout()
        plt.show()
        
        return 

    def build_coeffs(self,sp,ats=None):
        '''
        eta_i=eta^A_i - eta^S_i and idem for rho (rho_i=rho^A_i - rho^S_i)
        From fortran vars in hazel_py.f90:
        !eta_i(1:4)=eta(1:4) - stim(1:4)  
        !rho_i(1:3)=eta(5:7) - stim(5:7)  
        From here: etas=sp.eta[aa,s,:]-sp.stim(aa,s,:) con s =0,1,2,3
                   rhos=idem con s =4,5,6

        Hazel build the total opt coeffs internally for the different 
        synthesis methods but it does not store them in runtime. We do it now, at the end.
        '''
        #from string to hazel.spectrum.Spectrum 
        if type(sp) is not hazel.spectrum.Spectrum:sp=self.spectrum[sp]

        sp.etas=sp.eta[:,0:4,:]-sp.stim[:,0:4,:]
        sp.rhos= np.zeros_like(sp.etas)
        sp.rhos[:,1:4,:]=sp.eta[:,4:7,:]-sp.stim[:,4:7,:]

        codic1={'epsi':sp.eps[:,0,:],'epsq':sp.eps[:,1,:],'epsu':sp.eps[:,2,:],'epsv':sp.stim[:,3,:],
            'etai':sp.etas[:,0,:],'etaq':sp.etas[:,1,:],'etau':sp.etas[:,2,:],'etav':sp.etas[:,3,:],
            'rhoq':sp.rhos[:,0,:],'rhou':sp.rhos[:,1,:],'rhov':sp.rhos[:,2,:]}

        codic2={'eps':sp.eps,'etas':sp.etas,'rhos':sp.rhos}

        return codic1,codic2 

    def plot_coeffs(self,sp,coefs=None,par=None,ats=None,scale=2,figsize=None,tf=4):
        if type(sp) is not hazel.spectrum.Spectrum:sp=self.spectrum[sp]
        
        #----------------------------------
        #Consider only the atmospheres in sp.
        #self.atms_in_spectrum[sp.name] --->. [['c0'], ['c1','c2']]
        if ats is None:ats=self.atms_in_spectrum[sp.name]#ats=self.atmospheres

        labs=[]
        #get name of atmospheres in sp
        for n, order in enumerate(self.atms_in_spectrum[sp.name] ): #n run layers along the ray
            for k, atm_name in enumerate(order):  #k runs subpixels of topologies c1+c2                                                  
                #at=self.atmospheres[atm]
                labs.append(atm_name)
        lines=[]
        #----------------------------------
        
        cd,cd2=self.build_coeffs(sp) #set sp.etas and sp.rhos
        #cds={**cd, **cd2} #merge the two dictionaries

        pscale=self.setup_set_figure('dummy',scale=scale,tf=tf)

        if coefs is None:#default    
            if figsize is not None:fs=figsize
            else:fs=(pscale*4,pscale*3)
            
            lab=['epsi','epsq','epsu','epsv','etai','etaq','etau','etav','','rhoq','rhou','rhov']

            alp=[1.,1.,1.] #make plots of MO terms transparent when not used in the calculation 
            if self.apmosekcl[1]==0:alp[2]=0.3

            f, ax = plt.subplots(nrows=3, ncols=4,figsize=fs,label=self.labelf2)
            for cc,coef in enumerate(['eps','etas','rhos']):
                for k,at in enumerate(ats):
                    for sto in range(4):
                        lx, =ax[cc,sto].plot(sp.wavelength_axis,cd2[coef][k,sto,:],alpha=alp[cc]) 
                        ax[cc,sto].set_title(mylab(lab[4*cc+sto]))
                        ax[cc,sto].set_xlabel(mylab('xx'))
        
                        if (sto==0) and (cc ==2):lines.append(lx)
            f.legend(tuple(lines), tuple(labs), loc=(0.1,0.1), bbox_to_anchor=(0.1, 0.3))
        else:
            f, ax = plt.subplots(nrows=1, ncols=len(coefs),figsize=(pscale*len(coefs),pscale),label=self.labelf2)
            for cc,coef in enumerate(coefs):
                alp=1.#make plots of MO terms transparent when not used in the calculation 
                if (self.apmosekcl[1]=='0' and coef[0:3]=='rho'):alp=0.3
                if coef in cd:
                    for k,at in enumerate(ats):
                        ax[cc].plot(sp.wavelength_axis,cd[coef][k,:],alpha=alp)
                        ax[cc].set_title(mylab(coefs[cc]))
                        ax[cc].set_xlabel(mylab('xx'))

                else:
                    raise Exception("Names can only be epsi,epsq,epsu,epsv,etai,etaq,etau,etav,rhoq,rhou,or rhov.")

        plt.tight_layout()
        plt.show()

        return f,ax


    def plot_funcatmos(self,dlims,hz,atmat=None,axs=None,scale=4,tf=2,**pkws): 
        '''
        Plot the variations of the atmospheric parameters after inserting a parametric multicell atmopshere
        **pkws: remaining keyword arguments for plot_PolyFx
        pkws={var':'mono','method':2}
    
        '''
        rct=[3,3,9]
        if pkws['plotit']!=9:rct=[4,3,10]

        pscale=self.setup_set_figure('dummy',scale=scale,tf=tf)

        if self.ax4 is None:#create the figure
            #following line of code is ready to be used (remove adjacent lines)
            #but we leave it until the end of development 
            #self.setup_set_figure(self.labelf4,scale=scale,tf=tf,xy=rct[0:2])
            self.f4, self.ax4 = plt.subplots(nrows=rct[0], ncols=rct[1],figsize=(pscale*2,pscale*2.5),label=self.labelf4)  
            self.ax4 = self.ax4.flatten()
            axs=self.ax4
        else:#reuse the axis of the figure already created
            axs=self.ax4
            for ax in axs:ax.cla()#delete curves in axes in future calls

        #selected labs in set_funcatmos :
        #selected=['B1','B2','B3','tau','v','deltav','beta','a','j10','j20f']# ATMAT ORDER
        #idem but with 'beta' at the end 
        labs=['B1','B2','B3','tau','v','deltav','a','j10','j20f','beta'] #PLOT ORDER
        #But after calling self.set_funcatmos from main, dlims is modified to bunch of labels:
        allabs=labs+['ff', 'nbar']#list(dlims.keys())

        for i,ax in enumerate(axs):
            if allabs[i]=='deltav':
                hzi,yi=self.fun_minT([hz[0],hz[-1]],dlims['deltav'])
                ax.plot(hzi, yi, '-')
            else:    
                var=pkws['var']
                if allabs[i]=='tau':var='exp'
                self.plot_PolyFx(ax,hz,dlims[allabs[i]],nps=pkws['nps'],var=var,method=pkws['method'])
                
            ax.set_title(mylab(allabs[i]))
            if i >8:ax.set_xlabel(mylab('hz'))

        #just plot the selected parameters that do change and were generated from a polynomial
        #As beta was stored in atmat after deltav, we must move it to plot it at the end as the labels
        if atmat is not None:
            j=[0,1,2,3,4,5,7,8,9,6]#beta at the end and only 10 values because atmat contains 10
            for i in range(rct[2]):#9 or 10
                axs[i].plot(hz, atmat[j[i],:], 'bo',ms=3)

        #plt.gcf().legend(tuple([l1,l2]), tuple(labs))#loc=(0.1,0.1), bbox_to_anchor=(0.1, 0.3))
        plt.tight_layout()
        plt.show()
        
        return axs


    def get_Tpars(self,lam0,vth=None,temp=None,dlamd=None,atw=4.,vmic=0): 
        '''
        Return all parameters related to Doppler broadening: deltav(i.e. vthermal),T, and dnuD or dlamD
        lam0 in angstroms. Velocities in km/s but we tranform them to cm/s for using cgs constants below
        Microturbulent velocity vmic is optional to transform to/from Temperature
        The equations are:
        vth=np.sqrt(2.0*kb*tt/(mp*atw)+vmic*vmic)  
        T=(vth*vth-vmic*vmic)*mp*atw/(2.0*kb)
        dlamd = (lam0/c)*vth 
        nu0=cc/(lam0);        dnud = (nu0/c)*vth       

        Typical call:
        tem,dlamd,vth=get_Tpars(lam0,atw=self.atwsdic['atom'], vth=vth_array )
        '''
        kb=1.380662E-16   ;mp=1.67265E-24  ;cc=3E10   #CGS
        #kb,cc,mp =1.3807E-23, 3E8, 1.6726E-27 #(J/K=kgm2s-2, m/s, kg)       
        if vth is not None:
            #vth,vmic =  vth*1E5,vmic*1E5  #in cm/s
            temp=1E10*(vth*vth-vmic*vmic)*mp*atw/(2.0*kb) #in Kelvin
            dlamd=1E5*vth*(lam0/cc)  #in Angstroms
        else:
            if temp is not None:
                vth=np.sqrt(2.0*kb*temp/(mp*atw)+vmic*vmic*1E10)# in cm/s  
                dlamd=vth*lam0/cc  #in Angstroms (if lam0 in Angstroms)
                vth=vth*1e-5 #output in km/s
            else: #dlamd must enter in same units as lam0(Angstroms)
                vth=1E-5*dlamd*cc/lam0 #in km/s.  
                temp=1E10*(vth*vth-vmic*vmic)*mp*atw/(2.0*kb) #in Kelvin

        #nu0=cc/(lam0*1E-10)
        #dnud=(nu0/cc)*np.sqrt(2*kb*temp/(matom*mp)+ vmic*vmic*1E10)  

        return vth,temp,dlamd

    def fun_minT(self,hzl,dlims,f2=None,d1=2.,z1=500.,hz=None):
        '''Create non-monotonic function in vth mimicking a chromospehric minimum of T
        at (z1,d1)=(500,2) with exact lower boundary value, minimum value, and an upper boundary value 
        determined by d2=d1*f2. Typically f2 >1.0 for a chromospheric rise 
        and the minimum value at minimum point is vth=2.0, i.e. d1=2.

        Typical call:
        hz=np.linspace(hzlims[0],hzlims[-1],Ncells)
        yfun=fun_minT(hz,4.,f2=1.5)

        '''
        #set data. z1 is hardcoded to 500 km
        if hz is None:hz = np.linspace(hzl[0], hzl[-1], num=30)
        z2=hz[-1]

        d0,d2=dlims[0],dlims[-1]
        if f2 is None:f2=d2/d0
        d2=d0*f2*f2 #f2 squared does the trick to fit d2 approx

        #define function
        BminT= lambda x,a,b,c,gam: (a+b*x+c*np.exp(-gam*x))

        gam=0.02#first guess 0.00125-0.003-0.01  
        #constraints at d1
        bb=(d2-d0*np.exp(-gam*z2))/(z2-(1./gam+z1)*(1.-np.exp(-gam*z2)))
        #consraints at d2
        aa=-bb*(1./gam+z1)
        #constraint at d0
        cc= d0-aa
        #correct gamma assuring constraint at d2
        ff=20.#15-100
        gam1=-(1./z2)*(np.log(d2)-np.log(ff*(d0-aa)))

        argum=-(aa+bb*z1)/(d0-aa)
        if argum>0:gam2=-(1./z1)*np.log(argum)
        gam=np.min([gam2,gam1])

        #recalculate
        bb=(d2-d0*np.exp(-gam*z2))/(z2-(1./gam+z1)*(1.-np.exp(-gam*z2)))
        aa=-bb*(1./gam+z1)
        cc= d0-aa

        #calculate seed function with right shape and relative proportions
        yy=BminT(hz,aa,bb,cc,gam)
        #fit it exactly
        yy=(yy-np.min(yy))
        yyprime=yy*(d0-d1)/yy[0] + d1
        

        return hz,yyprime

    def check_method(self,method):
        if (self.verbose >= 1):self.logger.info('Synthesis method : {0}'.format(method))
        if method not in self.methods_list:raise Exception("I do not know that synthesis method. Stopping.")


    def mutates(self,spec,parsdic=None,\
        B1=None,B2=None,B3=None,tau=None,v=None,deltav=None, beta=None,a=None,ff=None,\
        j10=None,j20f=None,nbar=None, \
        apmosekc=None,dcol=None, \
        compare=True,frac=False,ax=None,pkws=None,bylayer=False):
        '''
        Create mutations in the synthesis Model, allowing repetition of an experiment changing
        a few parameters from a previous synthesis. 
        The versatility of reading pars both from parsdic and from keywords, and the checking of 
        the pars as done originally when setting up the model, make this routine cumbersome.
        
        Future aspects to mutate : ref_frame, hz topology,los,boundary...  
        Due to the Python behavior, newmo=self is just a reference assignment where both variable names 
        point to the same object, it would only create newmo as a pointer to the object self, 
        without truly performing an indepedent copy. For doing a copy of arrays one has np.copy(). 
        For objects/dictionaries we have deepcopy. 
     
        Parsdic (and any other mutable type,lists or dictionaries) defined as keyword parameters
        will no reset their values between function calls, so having memory of previous  mutates()
        calls from main. If we do not want to use a DTO class or avoid parsdic, then we need to define its
        default to None and checking and reserting to {} at the beggining of this function. 
        
        We use dlims (a new property of the models) and pwks para rellamar a funcatmos desde mutates.
        Here, pkws are the same keywords that were introduced for set_funcatmos. 

        The subroutine accepts mutations of the atmospheric parameters in two ways: 
        1)When "bylayer=True", a given keyword is set as Keyword=[L,val] with L the specific layer, identified
        by an integer between 1 and Nlayers, where the keyword is going to mutate.eg B1=[3,100.0]
        2) If bylayer is False(default) , the values for each keyword contain a two-element list with
         specifying a range of variation to be applied to set_funcatmos for that physical quantity.
         e.g., B1=[50.,200.]
        '''

        dloc=dict(locals())#all keywords of the subroutine
        extrapars={}
        mutating_keys=[]

        #----next vars could be extracted out as global to avoid repetition--------
        message="Unknown mutable parameter. The options are:  \n Atompol, MO effects,Stim. emission, Kill coherences, B1, B2, B3 ...{0}"

        '''Identify Model object parameters to be changed. 
        First, general parameters of the Model: apmosekc, dcol and their product apmosekcl'''
        extrapars_list=['Atompol','MO effects','Stim. emission','Kill coherences']

        #parameters of chromosphere objects
        atmpars=['B1','B2','B3','tau','v','deltav','beta','a','ff','j10','j20f','nbar']

        #checkdictio lists parameters that can be mutated until now using parsdic.
        checkdictio={'apmosekc':None,'dcol':None,'Atompol':None,'MO effects':None,'Stim. emission':None,'Kill coherences':None,
            'B1': None, 'B2': None, 'B3': None,'tau':None,'v':None,'deltav':None,'beta':None,
            'a':None,'ff':None,'j10':None,'j20f':None,'nbar':None}
        kwskeys=['apmosekc','dcol']+atmpars    
        #Atompol, MOeffects, S.emiss, and Kill-coherences can be introduced via
        #apmosekc or via parsdic but not as specific long keywords. However those 
        #long keys are set there in the object because we have "decompress" them from apmosekc before
        #As they can also be specifically introduced via parsdic dictionary, their four long keywords 
        #contained in extrapars_list are not considered in the next block to include them 
        #in parsdic, because they are already there and never as individual subr keywords.
        #------------------------------------------------------------------------------------------------
        if parsdic is None:parsdic={}
        #copying keywords to parsdic
        for k in kwskeys:
            if dloc[k] is not None:parsdic[k]=dloc[k]    

        if (bylayer is False) and (pkws is None):
            warnings.warn("A plotting dictionary 'pkws' is needed to mutate all layers at once with set_funcatm().")
            warnings.warn("The following default dictionary is assumed:")
            warnings.warn("{'plotit':9,'nps':3,'var':'mono','method':1}")
            pkws={'plotit':9,'nps':3,'var':'mono','method':1}


        #now that parsdic is complete, its fields are those that will be mutated 
        #and we collect them those that are in atmpars below in mutating_keys list

        newmo= copy.deepcopy(self) #here we are already copying the spectrum objects inside spectrum
        
        '''kill ghost figures f1,... appearing when replicating with deep copy the model object.
        otherwise, ugly replicants of the figures in self.fx pops up when calling again a plt.show''' 
        for fig in ['f1','f2','f3','f4']:self.remove_fig(newmo,fig)        
        
        # add the use of reshow for recovering closed windows,delete
        #things in syntesize, remember the use of compare mutation, add f2 to the list,
        if (len(parsdic)!=0): #if parsdic is not empty (default is {}) we shall ignore mutation keywords!
            #write a list of all possible parameters and check whether the inputs are compatible/valid
            for elem in parsdic:
                if (elem not in checkdictio):raise Exception(message.format(elem))

            #build extrapars dictionary (if its keywords are in parsdic) to call get_apmosekcl()
            for elem in extrapars_list:
                if (elem in parsdic):extrapars[elem]=parsdic[elem]

            apmosekc,dcol=self.apmosekc,self.dcol
            if ('apmosekc' in parsdic):apmosekc=parsdic['apmosekc']
            if ('dcol' in parsdic):dcol=parsdic['dcol']
            if (len(extrapars)!=0) or ('apmosekc' in parsdic) or ('dcol' in parsdic):#update self.apmosekcl
                newmo.apmosekcl=newmo.get_apmosekcl(apmosekc,dcol,extrapars)

            for key in atmpars:#key=[atm_number,value]#key value. list mutating keys for atmospheres
                if (key in parsdic):mutating_keys.append(key) #ONLY FOR ATM PARS!

        else:#read mutation pars only from keywords because only extrapars are being mutated
            if (apmosekc is not None) or (dcol is not None): #if one of them must be updated
                if (apmosekc is None):apmosekc=self.apmosekc #let the other as in original model
                if (dcol is None):dcol=self.dcol              #let the other as in original model
                newmo.apmosekcl=newmo.get_apmosekcl(apmosekc,dcol,{}) #and update + check

        #--------------------------------------------------------------------------------
        #Close and reopen Hazel seems necessary to detect updated values of j10,j20f,and nbar 
        newmo.exit_hazel()   
        newmo.ntrans=hazel_code._init(self.atomfile,0) #We initialize pyhazel (and setup self.ntrans) with verbose=0 

        if type(spec) is not hazel.spectrum.Spectrum:spec=self.spectrum[spec]
        '''        
        original spectrum was already duplicated when duplicated de Model object. 
        Here we just get a pointer to it maintaining the name of spectrum 
        '''
        newspec=newmo.spectrum[spec.name] #just for shortening sintaxis
        '''
        Here we decide to leave same name, but we make here explicit how to proceed otherwise.
        If we wish to change the name of the new spectrum in the new object
        we have to change it everywhere (in self.topologies and in atms_in_spectrum):
        '''
        newspecname=spec.name #same name
        #substitute the key spec.name by newspecname in all dictionaries:
        #newmo.spectrum[newspecname]= newmo.spectrum.pop(spec.name) 
        #newmo.atms_in_spectrum[newspecname]= newmo.atms_in_spectrum.pop(spec.name)
        #newmo.topologies[newspecname]= newmo.topologies.pop(spec.name) #{'sp1':'ch1->ch2'}
       
        '''Now we reset spectrum without calling model.add_spectrum again (inefficient).
        We only need to reset the optical coeffs and stokes, and change few pars in 
        spectrum object: LOS, BOUNDARY.'''        

        #spectrum.add_spectrum (NOT model.add_spectrum) to reset spectrum
        wvl=newspec.wavelength_axis
        wvl_lr=newspec.wavelength_axis_lr 
        wvl_range = [np.min(wvl), np.max(wvl)]#used below
        newspec.add_spectrum(newmo.nch, wvl, wvl_lr)#reset stokes, eps, eta, stim, etas, rhos
        '''
        We could directly modify hazelpars in atmopsheres(with this line in
        chromosphere.py: self.atompol,self.magopt,self.stimem,self.nocoh,self.dcol = hazelpars) 
        without updating apmosekcl in model, but we take advantage of update in Model to make input checks
        '''
        if (self.verbose >= 1):self.logger.info('Mutating atmospheric pars...')
        
         
        if bylayer is True:#change the atmosphere only at the specified layers
            #run over the existing atmospheres of this spectrum topology and set pars
            for n, order in enumerate(newmo.atms_in_spectrum[newspecname] ): #n run layers along the ray
                for k, atm in enumerate(order):  #k runs subpixels of topologies c1+c2                                                  
                    at=newmo.atmospheres[atm]
                    """
                    Activate this spectrum with add_active_line for all existing atmospheres.
                    In normal setup, activate_lines is called after adding all atmospheres in topology.
                    """         
                    at.add_active_line(spectrum=newspec, wvl_range=np.array(wvl_range))

                    #SET HAZELPARS: self.atompol,self.magopt,self.stimem,self.nocoh,self.dcol = hazelpars                
                    at.atompol,at.magopt,at.stimem,at.nocoh,at.dcol = newmo.apmosekcl 
                    
                    #key=[atm_number,value]#key value. Produce mutation updating atm.dna[key]            
                    for key in mutating_keys:#print(key,parsdic[key][0],parsdic[key][1])
                        #if in selected layer mutate one layer at a time for every parameter,
                        #but layer can be different among parameters
                        if (parsdic[key][0]==n+k):at.dna[key]=parsdic[key][1] #mutates dna.  #print(key,n+k)
                    pars,kwds=at.get_dna() #get updated dna pars of this single atmosphere                     
                    at.set_pars(pars,**kwds)#ff=at.atm_dna[8],j10=at.atm_dna[9],j20f=at.atm_dna[10],nbar=at.atm_dna[11]) 
        else:#change the whole atmosphere 
            for k in mutating_keys:newmo.dlims[k]=parsdic[k]
            pkws['plotit']=0
            hz=newmo.set_funcatm(newmo.dlims,selected=mutating_keys,hztype='parab',\
                orders=4,**pkws)

        #--------------------------------------------------------------------------------
        #Synthesize the new spectrum in original model FROM the new model object:
        newmo.synthesize(method=self.methods_dicT[newmo.synmethod],muAllen=newmo.muAllen,obj=self)
        if (self.verbose >= 1):self.logger.info('Spectrum {0} has mutated.'.format(spec.name))
        
        if (compare is True):self.compare_mutation(spec,newspec,fractional=frac) 
        
        return newmo,newspec


    def compare_mutation(self,sp,newsp,scale=3,tf=2.4,ls='dashed',fractional=False,
        lab=['iic','qic','uic','vic']):

        #all plots in compare_mutation will always be done with the fractional keyword set 
        #in the very first call to mutates and compare_mutation in main to avoid ill comparisons.
        if self.lock_fractional is None:self.lock_fractional=fractional
        else:fractional=self.lock_fractional

        if fractional:lab=['iic','qi','ui','vi']
        else:lab=['iic','qic','uic','vic']            

        pscale=self.setup_set_figure('dummy',scale=scale,tf=tf)
        
        if self.ax3 is None:#create and arrange axes, plot old spectra
            self.f3, ax = plt.subplots(2,2,figsize=(pscale*2,pscale*2),label=self.labelf3)#'Mutations (Model.ax2)')  
            self.ax3 = ax.flatten()    

            #to make it alwayss visible , the original first one is plot last one with dashed black line 
            self.ax3[0].plot(sp.wavelength_axis,sp.stokes[0,:],linestyle=ls,color='k')
        
            for i in range(1,4):
                if fractional:self.ax3[i].plot(sp.wavelength_axis,sp.stokes[i,:]/sp.stokes[0,:],linestyle=ls,color='k')
                else:self.ax3[i].plot(sp.wavelength_axis,sp.stokes[i,:],linestyle=ls,color='k')
            
            for i in range(4):
                self.ax3[i].set_title(mylab(lab[i]))#,size=8 + 0.7*pscale)
                if i>1:self.ax3[i].set_xlabel(mylab('xx'))#,size=8 +0.7*pscale)#,labelpad=lp)
        else:#if the window was created but was closed reshow it 
            if self.f3.canvas.manager.get_window_title() is None:self.reshow(self.f3)

        #plot mutated spectra
        self.ax3[0].plot(newsp.wavelength_axis,newsp.stokes[0,:])
        for i in range(1,4):
            if fractional:self.ax3[i].plot(newsp.wavelength_axis,newsp.stokes[i,:]/newsp.stokes[0,:])
            else:self.ax3[i].plot(newsp.wavelength_axis,newsp.stokes[i,:])

        plt.tight_layout()
        plt.show()

    def compare_experiments(self,objB,specname,bylayers=True,pre='8.3f'):
        '''Variables to compare are: 
        1) general atmospheric keywords:ref_frame,coordB,...
        2) 5 global atmopsheric physical switches:
        objB.apmosekcl (at.atompol,at.magopt,at.stimem,at.nocoh,at.dcol )
        at.atompol,at.magopt,at.stimem,at.nocoh,at.dcol = newmo.apmosekcl 
        self.atompol,self.magopt,self.stimem,self.nocoh,self.dcol = hazelpars
        3)The 12 DNA pars ('B1','B2','B3','tau','v','deltav','beta','a','ff','j10','j20f','nbar')
        -->pars,kwds=at.get_dna() #are divided in 8 pars and 4 keywords that were passed to ancient subroutines        
        '''

        self.logger.info('Comparing atmospheric models for spectrum {0}...'.format(specname))
        

        #check that spec is both in self object and objB
        if (specname not in objB.spectrum) or (specname not in self.spectrum):
            raise Exception("No spectrum {0} found. Are you comparing the right model objects?".format(specname))            

        keylabs=['Atompol','MO effects','Stim. emission','Kill coherences','Depol. colls.']
        for kk,elem in enumerate(objB.apmosekcl):
            if elem!=self.apmosekcl[kk]:print("{0} values DIFFER".format(keylabs[kk]))
        
        #objects to compare are assumed to have same cell atm names because mutations do not affect that
        anyatmname=list(self.atmospheres.keys())[0]
        if self.atmospheres[anyatmname].coordinates_B!=objB.atmospheres[anyatmname].coordinates_B:
            print("Magnetic coordinates DIFFER")
        if self.atmospheres[anyatmname].reference_frame!=objB.atmospheres[anyatmname].reference_frame:
            print("LOS ref frames DIFFER")

        #check the 12 dna parameters 
        #run over the existing atmospheres of this spectrum topology in both objects
        for n, order in enumerate(objB.atms_in_spectrum[specname]): #n run layers along the ray
            differ=0
            for k, atm in enumerate(order):  #k runs subpixels of topologies c1+c2                                                  
                if bylayers or (n in [0,self.n_chromospheres-1]):
                    for key in self.atmospheres[atm].dna.keys():
                        aa=self.atmospheres[atm].dna[key]
                        bb=objB.atmospheres[atm].dna[key]
                        if np.all(np.isclose(aa,bb)) != True:
                            print("LAYER {0}".format(n),end=': ')
                            if type(aa) is np.ndarray:
                                print("{0} : {1} vs. {2}".format(key,[float('%7.3f'%x) for x in aa],[float('%7.3f'%x) for x in bb]))
                            else:
                                print("{0} : {1:{pp}} vs. {2:{pp}}".format(key,aa,bb,pp=pre))
                            differ=1
            if bylayers and (differ==0):print("LAYER {0} MATCH".format(n))
        #check only limiting values of pars in the atmosphere
                
        return     

    def setup(self):
        """
        Setup the model for synthesis/inversion. This setup includes adding the topologies, removing unused
        atmospheres, reading the number of cycles for the inversion and some sanity checks.
        Setup requires values in self.topologies that were added in add_spectral
        we want create global optical coeffs in add_spectral together with stokes var
        to create global coeffs we require n_chromospheres because is set in setup.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
              
        # Adding topologies
        if (self.verbose >= 1):
            self.logger.info("Adding topologies") 
        #for value in self.topologies:
        #    self.add_topology(value)
        for specname, value in self.topologies.items():
            self.add_topology(value,specname)

        # Remove unused atmospheres defined in the configuration file and not in the topology
        if (self.verbose >= 1):
            self.logger.info("Removing unused atmospheres")
        self.remove_unused_atmosphere()        

        # Calculate indices for atmospheres, n_chromospheres is set here
        index_chromosphere = 1
        self.n_photospheres = 0
        self.n_chromospheres = 0
        for k, v in self.atmospheres.items():
            if (v.type == 'chromosphere'):
                v.index = index_chromosphere 
                index_chromosphere += 1  
                self.n_chromospheres += 1
        #EDGAR:index_chromosphere could be avoided using v.index as counting variable.
        #Also, as the variation of v.index happens all inside the above loop, that parameter endsup being equal
        #to n_chromospheres, hence most of this loop looks unnecessary.

        if (self.verbose >= 1):#EDGAR: print number of Hazel chromospheres/slabs
            self.logger.info('N_chromospheres at setup',self.n_chromospheres)

        self.use_analytical_RF = False


        # Check that number of pixels is the same for all atmospheric files in synthesis mode
        n_pixels = [v.n_pixel for k, v in self.atmospheres.items()]
        all_equal = all(x == n_pixels[0] for x in n_pixels)
        if (not all_equal):
            for k, v in self.atmospheres.items():
                self.logger.info('{0} -> {1}'.format(k, v.n_pixel))
            raise Exception("Files with model atmospheres do not contain the same number of pixels")
        else:
            if (self.verbose >= 1):
                self.logger.info('Number of pixels to read : {0}'.format(n_pixels[0]))
            self.n_pixels = n_pixels[0]
        

        filename = os.path.join(os.path.dirname(__file__),'data/LINEAS')
        ff = open(filename, 'r')
        self.LINES = ff.readlines()
        ff.close()

        #if (self.verbose >= 1):#print number of Hazel chromospheres/slabs
        #    self.logger.info('N_chromospheres',self.n_chromospheres)

        #if self.plotit:
            #self.setup_set_figure(self.labelf1) #self.fig and self.ax are created here    
            #return f,ax
        

    def open_output(self):
        self.output_handler = Generic_output_file(self.output_file)        
        self.output_handler.open(self)

    def close_output(self):        
        self.output_handler.close()

    def write_output(self, randomization=0):
        self.flatten_parameters_to_reference(cycle=0)        
        self.output_handler.write(self, pixel=0, randomization=randomization)


    def add_spectrum(self, name, config=None, wavelength=None, topology=None, los=None, 
        i0fraction=1.0,boundary=None, atom=None, synmethod=None,
        linehazel=None, atmos_window=None, instrumental_profile=None,
        wavelength_file=None):
        """
        Similar to add_spectral but with keywords and more compact.
        Programmatically add a spectral region
        """
        value = dict(locals()) #retrieve keywords as dictionary
        value['name'] =name #we add the positional argument "name" to the dictionary


        if (self.verbose >= 1):            
            self.logger.info('Adding spectral region {0}'.format(name))        
    
        # Wavelength file is not present
        if (wavelength_file is None): 
            # If wavelength is defined            
            if (wavelength is not None): #if ('wavelength' in value):
                axis = wavelength
                wvl = np.linspace(float(axis[0]), float(axis[1]), int(axis[2]))                
                wvl_lr = None
                if (self.verbose >= 1):
                    self.logger.info('  - Using wavelength axis from {0} to {1} with {2} steps'.format(float(axis[0]), float(axis[1]), int(axis[2])))
            else:
                raise Exception('Wavelength range is not defined. Please, use "Wavelength" or "Wavelength file"')
        else:
            # If both observed and synthetic wavelength points are given
            if (wavelength is not None):
                axis = wavelength
                if (len(axis) != 3):
                    raise Exception("Wavelength range is not given in the format: lower, upper, steps")
                wvl = np.linspace(float(axis[0]), float(axis[1]), int(axis[2]))
                if (self.verbose >= 1):
                    self.logger.info('  - Using wavelength axis from {0} to {1} with {2} steps'.format(float(axis[0]), float(axis[1]), int(axis[2])))
                    self.logger.info('  - Reading wavelength axis from {0}'.format(wavelength_file))
                wvl_lr = np.loadtxt(self.root + wavelength_file)
            else:
                if (self.verbose >= 1):
                    self.logger.info('  - Reading wavelength axis from {0}'.format(wavelength_file))
                wvl = np.loadtxt(self.root + wavelength_file)
                wvl_lr = None
   

        if (los is None):
            raise Exception("You need to provide the LOS for spectral region {0}".format(name))
        else:
            if (self.verbose >= 1):
                self.logger.info('  - Using LOS {0}'.format(los))
            los = np.array(los).astype('float64')
        #---------------------------------------------
        if (self.verbose >= 1):
            self.logger.info('  - Using I0fraction = {0} for normalization in spectral region {1}'.format(i0fraction,name))
        '''
        EDGAR: the value of boundary that enters in the call to spectrum below  
        must be Ibackground(physical units)/I0Allen (all input and output Stokes shall always be
        relative to the Allen I0 continuum). Then, the value given to the boundary keyword from outside 
        must be such that is normalized to the Allen continuum. Otherwise, we have to multiply by 
        i0fraction to assure that quantity.
        In general this i0fraction keyword shall be ignored and assumed always to be 1.0, thus concentrating
        all the definition of the boundary condition in the boundary keyword. 

        But if the boundary keyword value was defined in main program 
        as Ibackground(physical units)/Icont_wing (i.e. with the first wing value of I equal to 1.0), 
        then we have to multiply by i0fraction keyword, which should then be different than 1.0 to describe
        a background continuum that is different from Allen. 
        (By the definition of i0fraction we have that Icont_wing is I0fraction*I0Allen).
        
        Example: for boundary=1, the boundary intensity given to Hazel synthesize routine should be
        Iboundary = 1 * I0fraction*I0Allen(lambda). 
        Then, in this section we first multiply boundary * I0fraction, 
        and before calling hazel we shall add units multiplying by I0Allen.
        '''
        if (boundary is None):
            if (self.verbose >= 1):self.logger.info('  - Using default boundary conditions [1,0,0,0] in spectral region {0} or read from file. Check carefully!'.format(name))
            boundary = i0fraction*np.array([1.0,0.0,0.0,0.0])  
            self.normalization = 'on-disk'
        else:
            if (self.verbose >= 1):
                if (np.ndim(boundary[0])==0):#boundary elements are scalars [1.0,0.0,0.0,0.0]
                    self.logger.info('  - Using constant boundary conditions {0}'.format(boundary))
                    if (boundary[0] == 0.0):self.logger.info('  - Using off-limb normalization (peak intensity)')          
                else:#the user already introduced float 64 arrays with spectral dependences for I.
                    self.logger.info('  - Using spectral profiles in boundary conditions')
                    if (boundary[0,0] == 0.0):self.logger.info('  - Using off-limb normalization (peak intensity)')          
            boundary = i0fraction*np.array(boundary).astype('float64')#gives array([1.0,0.0,0.0,0.0]) or array of (4,Nwavelength) 
        
        #---------------------------------------------
        
        #EDGAR: atom, line_to_index and line keywords moved to add_spectral
        if (atom is not None) and (atom in self.atomsdic):
            self.atom=atom#self.atom can be deleted because is not used anywhere else
            self.line_to_index=self.atomsdic[atom]
        else:
            raise Exception('Atom is not specified or not in the database. Please, define a valid atom.')
    
        if (self.verbose >= 1):self.logger.info('Atom added.')
        
        '''
        EDGAR:The default initialization of self.synmethod was done in the init above. 
        Below is the general set up of the synthesis method to use for synthesizing this spectrum.
        In principle here we assume one single method per spectrum, but one could later 
        change the method during runtime with the keyword method in synthesize(), 
        such that a single spectrum could be synthesize with different methods in different 
        layers during the transfer. Thats is why we do not associate a synmethod to the spectrum object,
        although if necessary we could do it below generating a list of methods in spectrum.
        We work with string names for the user, but internally hazel work with the numbers and the spectrum
        list is made with numbers to shorten because we can have many layers in Hazel2.
        '''
        if (synmethod is not None) and (synmethod != self.methods_dicT[self.synmethod] ):
            self.check_method(synmethod) #synmethod is a string with name, self.synmethod is the number
            self.synmethod=self.methods_dicS[synmethod] #update self.synmethod with the number


        #EDGAR: line for Hazel chromospheres and for SIR photosphere
        #it seems lines for SIR read in add_photosphere were wrong because they were introduced programatically
        #with the field atm['spectral lines'] in add_photosphere, but there was no such a field defined anywhere 
        lineH= ''
        if (linehazel is not None) and (linehazel in self.atomsdic[atom]):
            lineH=linehazel #e.g. '10830'.  Lines for activating in Hazel atmos
            if (self.verbose >= 1):self.logger.info("    * Adding HAZEL line : {0}".format(lineH))
        else:
            raise Exception('Line is not specified or not in the database. Please, define a valid line.')

        #Count chromospheres for defining optical coeffs containers, now that all atmospheres have been added
        self.nch=0  #n_chromospheres=0    
        #careful:is this the number of chromospheres added or the number associated to spectrum??
        for k, atm in self.atmospheres.items():            
            if (atm.type == 'chromosphere'):self.nch += 1 #should be equal to self.n_chromospheres.

        if (self.verbose >= 1):self.logger.info('N_chromospheres before setup',self.nch)


        #initialize here the optical coefficient containers with self.nch dimension:
        self.spectrum[name] = Spectrum(wvl=wvl, 
            name=name, los=los, boundary=boundary, 
            instrumental_profile=instrumental_profile, 
            root=self.root, wvl_lr=wvl_lr,lti=self.line_to_index,lineHazel=lineH,
            n_chromo=self.nch, synmethod=self.synmethod)

        #EDGAR: update spectrum object with the multiplets for later accesing it from synthesize at chromosphere.py
        self.spectrum[name].multiplets = self.multipletsdic[atom] 
        #ntrans needed to define length of nbar,omega, and j10.
        self.spectrum[name].ntrans = self.ntrans #len(self.multipletsdic[atom])

        #--EDGAR---------------------------------------------------------------
        #we are here defining the wavelength window for all atmospheres associated to this spectral region
        if (atmos_window is not None):#EDGAR: if not in dictionary, then take the one of current spectral region
            wvl_range = [float(k) for k in atmos_window]
        else:
            wvl_range = [np.min(self.spectrum[name].wavelength_axis), np.max(self.spectrum[name].wavelength_axis)]

        #self.topologies.append(topology)#'ph1->ch1+ch2'
        self.topologies[name]=topology# anade una entrada del tipo {'sp1':'ch1->ch2'}
        
        """
        Activate this spectrum with add_active_line for all existing atmospheres.
        Part of this routine was previously inside every add_atmosphere routine.
        Now all spectral and atmospheric actions and routines are disentangled. 
        Activate_lines is now called after adding all atmospheres in topology.
        """
        if (self.verbose >= 1):self.logger.info('Activating lines in atmospheres',self.nch)
        for k, atm in self.atmospheres.items():            
            atm.add_active_line(spectrum=self.spectrum[name], wvl_range=np.array(wvl_range))
                        

        return self.spectrum[name]


    def check_key(dictio,keyword,default):
        if (keyword not in dictio):
            dictio[keyword] = default
        elif (dictio[keyword] == 'None'):
            dictio[keyword] = default
        return dictio



    def add_chromosphere(self, atmosphere):
        """
        Programmatically add a chromosphere

        Parameters
        ----------
        atmosphere : dict
            Dictionary containing the following data
            'Name', 'Spectral region', 'Height', 'Line', 'Wavelength',
            ...
            magnetic field reference frame
            coordinates for magnetic field parameters
            ...
            'Reference atmospheric model',
            'Ranges', 'Nodes'

        Returns
        -------
        self.atmospheres[atm['name']]         #None
        """

        # Make sure that all keys of the input dictionary are in lower case
        # This is irrelevant if a configuration file is used because this has been already done
        atm = hazel.util.lower_dict_keys(atmosphere)

        self.atmospheres[atm['name']] = Hazel_atmosphere(working_mode=self.working_mode, \
        name=atm['name'],ntrans=self.ntrans,hazelpars=self.apmosekcl)#EDGAR:,atom=atm['atom'])

        #EDGAR: more efficient and flexible reference frame set up
        refkey=''
        self.atmospheres[atm['name']].reference_frame = 'vertical'#default
        if ('reference frame' in atm):refkey='reference frame'
        if ('ref frame' in atm):refkey='ref frame' #short keyword alias
        if (refkey != ''):#desired reference frame has been specified
            if (atm[refkey] == 'line-of-sight' or atm[refkey] == 'LOS'):
                self.atmospheres[atm['name']].reference_frame = 'line-of-sight'
            elif (atm[refkey] != 'vertical'):
                raise Exception('Error: wrong specification of reference frame.')

        if (self.verbose >= 1):
            self.logger.info("    * Adding line : {0}".format(atm['line']))
            self.logger.info("    * Magnetic field reference frame : {0}".format(self.atmospheres[atm['name']].reference_frame))

        if ('ranges' in atm):
            for k, v in atm['ranges'].items():
                for k2, v2 in self.atmospheres[atm['name']].parameters.items():
                    if (k.lower() == k2.lower()):
                        if (v == 'None'):
                            self.atmospheres[atm['name']].ranges[k2] = None
                        else:
                            self.atmospheres[atm['name']].ranges[k2] = hazel.util.tofloat(v)

        for k2, v2 in self.atmospheres[atm['name']].parameters.items():
            self.atmospheres[atm['name']].regularization[k2] = None

        if ('regularization' in atm):
            for k, v in atm['regularization'].items():                
                for k2, v2 in self.atmospheres[atm['name']].parameters.items():                    
                    if (k.lower() == k2.lower()):                        
                        if (v == 'None'):
                            self.atmospheres[atm['name']].regularization[k2] = None
                        else:
                            self.atmospheres[atm['name']].regularization[k2] = v
        #EDGAR : now 'coordinates for magnetic field vector' is 'coordB' 
        #but remember to write it here in lower case (see above)
        if ('coordb' in atm):
            if (atm['coordb'] == 'cartesian'):
                self.atmospheres[atm['name']].coordinates_B = 'cartesian'
            if (atm['coordb'] == 'spherical'):
                self.atmospheres[atm['name']].coordinates_B = 'spherical'
        else:
            self.atmospheres[atm['name']].coordinates_B = 'spherical' #default

        self.atmospheres[atm['name']].select_coordinate_system()

        if (self.verbose >= 1):            
            self.logger.info("    * Magnetic field coordinates system : {0}".format(self.atmospheres[atm['name']].coordinates_B))            


        if ('reference atmospheric model' in atm):
            my_file = Path(self.root + atm['reference atmospheric model'])
            if (not my_file.exists()):
                raise FileExistsError("Input file {0} for atmosphere {1} does not exist.".format(my_file, atm['name']))

            self.atmospheres[atm['name']].load_reference_model(self.root + atm['reference atmospheric model'], self.verbose)

            if (self.atmospheres[atm['name']].model_type == '3d'):
                self.atmospheres[atm['name']].n_pixel = self.atmospheres[atm['name']].model_handler.get_npixel()
        
        # Set values of parameters
        self.atmospheres[atm['name']].height = float(atm['height'])

        if ('nodes' in atm):
            for k, v in atm['nodes'].items():
                for k2, v2 in self.atmospheres[atm['name']].parameters.items():
                    if (k.lower() == k2.lower()):                            
                        self.atmospheres[atm['name']].cycles[k2] = hazel.util.toint(v)
        #EDGAR: return directly the dict entry with the name of the chromo that has been just added
        return self.atmospheres[atm['name']] 

    def add_chrom(self, atmosphere):#EDGAR: alias to add_chromosphere
        return self.add_chromosphere(atmosphere)

    def add_Nchroms(self,tags,ckey,hz=None):
        '''
        Add N chromospheres with a list of tags/names(e.g.['c1','c2']), 
        a dictionary of common paraemters (ckeys), and 
        a list of specific heigths as optional keyword.
        '''
        if (hz is None):hz=[0.0]*len(tags)
        #chout=[]
        for kk,ch in enumerate(tags):
            #chout.append(self.add_chromosphere({'name': ch,'height': hz[kk],**ckey}))
            self.chromospheres.append(self.add_chromosphere({'name': ch,'height': hz[kk],**ckey}))

        #return chout, tags
        return self.chromospheres, tags

    def set_hz(self,hzlims=[0.,1500.],hztype='lin',Ncells=None):
        if Ncells is None:Ncells=self.n_chromospheres        

        hz = np.linspace(hzlims[0], hzlims[-1], Ncells)
        xaux=np.linspace(0,Ncells,Ncells)#yaux is hz : yaux=np.linspace(hzlims[0],hzlims[1],Ncells)
        if hztype=='lin':funX=xaux
        if hztype=='parab':funX=xaux*xaux
        xnew=np.max(xaux)*funX/np.max(funX)#np.exp(-x)
        hz = np.interp(xnew, xaux, hz)
        
        return hz 

    def add_funcatmos(self,Ncells,ckey,hzlims=[0.,1500.],hztype='lin',topo=''):
        '''
        Creates and add a full chromosphere made of N elemental pieces/slabs/cells
        and making certain parameters to vary according to given P-order polinomials.
        There is a dictionary of common parameters (ckeys), and 
        a list of specific heigths as optional keyword. For the moment, this function
        shall just be a small improvement to add_Nchroms.

        Explanation: Hazel2 allows for a versatile serial or parallel combination 
        of slabs that can be associated to different spectral lines, heights, filling 
        factors and topologies for perform radiative trasnfer on them. On the other side,
        other RT codes work directly to stratified fully discretized atmospheres with many 
        points. The function add_funcatmos here is in between these two approaches, 
        pretending to mimick a full atmosphere with several points from serially concatenating
        Hazel atmospheres but yet mantaining a control on the functional variations
        of its physical parameters. The goal is to simplify its creation process because
        when many chromospheres are added as in add_Nchroms, the qualitative properties
        of each of them (like its labels, or reference frame) become irrelevant or equal
        for the whole piece. The next step would be to directly add a realistic atmosphere
        or convert one to a toy full chromosphere as those here built for Hazel. But for that
        it we shall need to work  directly with temperature, density, etc.

        '''
        #create list of tags/names for each cell (['c1','c2',...])
        #actual value of hz only matters in relation with tau. If tau is left defined
        #for each cell as incoming parameter we are also setting the height scale
        #indirectly with dtau=eta_I*dz. Hence we will need to wait for eta_I before returning 
        #a meaningful value of hz scales
        if 'hzlims' in ckey:hzlims=ckey['hzlims']
        if 'hztype' in ckey:hztype=ckey['hztype']
        self.n_chromospheres=Ncells
        self.hzlims=hzlims

        self.hz=self.set_hz(hzlims=hzlims,hztype=hztype)

        #we must return topology string used later for add spectrum in the main program
        tags=[]
        
        if topo == '':#we generate ordered tags with names from c_0 to c_{N-1}
            for kk in range(Ncells):
                tags.append('c'+str(kk)) #['c0','c1',...,'c_{N-1}']
                topo=topo+tags[kk]+'->'
            topo=topo[:-2]
        else:#from topo extract tags
            tags=topo.rsplit(sep='->') #['c0','c1',...,'c_{N-1}']
        '''
        if topo == '':#we generate ordered tags with names from c_1 to c_N
            for kk in range(Ncells):#rango de 0 a n-1
                tags.append('c'+str(kk+1)) #['c1','c2',...,'c_{N}']#rango de 1 a N
                topo=topo+tags[kk]+'->' #posiciones de 0 a n-1
            topo=topo[:-2]
        else:#from topo extract tags
            tags=topo.rsplit(sep='->') #['c1','c2',...,'c_{N-1}']
        '''
        #chout=[]
        for kk in range(Ncells):        
            #chout.append(self.add_chromosphere({'name': tags[kk],'height': self.hz[kk],**ckey}))
            self.chromospheres.append(self.add_chromosphere({'name': tags[kk],'height': self.hz[kk],**ckey}))

        #return chout, topo
        return topo

    def fix_point_polyfit_fx(self,n, x, y, xf, yf) :
        '''Solves a system of equations that allow o determine the parameters of a polynomial 
        that fit points (x,y) approximatelly passing exactly through points (xf,yf).
        At the end return the resulting  polynomial'''
        mat = np.empty((n + 1 + len(xf),) * 2)
        vec = np.empty((n + 1 + len(xf),))
        x_n = x**np.arange(2 * n + 1)[:, None]
        yx_n = np.sum(x_n[:n + 1] * y, axis=1)
        x_n = np.sum(x_n, axis=1)
        idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
        mat[:n + 1, :n + 1] = np.take(x_n, idx)
        xf_n = xf**np.arange(n + 1)[:, None]
        mat[:n + 1, n + 1:] = xf_n / 2
        mat[n + 1:, :n + 1] = xf_n.T
        mat[n + 1:, n + 1:] = 0
        vec[:n + 1] = yx_n
        vec[n + 1:] = yf
        params = np.linalg.solve(mat, vec)
        
        sel_pars=params[:n + 1] 

        return np.polynomial.Polynomial(sel_pars)


    def get_yps(self,xps,ypl,sortit=1,method=2,nps=2):
        '''creates few random nps points in y contained between limits ypl 
        creates random polynomial and few points mapping it''' 
        if (np.abs(sortit) != 1): sortit=1#sortit can only be 1 or -1
        if method==1:
            order=4 #for nps=2, large over bumps if order is not 3 or 4
            #polyfx = np.polynomial.Polynomial(np.random.rand(order + 1)) #simpler option
            polyfx = np.polynomial.Polynomial(np.random.uniform(-2, 8, size=(order + 1,)  ))
            y=polyfx(xps)
            yps=ypl[0]+(ypl[1]-ypl[0])*y/np.max(y)
            if ypl[0]>ypl[-1]:yps=np.sort(yps)[::-1]
        
        if method==2:
            #take some random nps-2 points inside the interval
            ypsr = np.random.uniform(low=ypl[0], high=ypl[1], size=nps-2)
            if ypl[0]>ypl[-1]:ypsr=np.sort(ypsr)[::-1]#sort from smaller to larger and reverse
            yps=[ypl[0]]+list(ypsr)+[ypl[1]] #monotonic series always

        return yps

    def get_exp3points(self,xx,xpl,ypl,nps=3):
        from scipy.optimize import curve_fit
        xps = np.linspace(xpl[0],xpl[1],nps)#few nps points contained between limits xpl 
        yps=self.get_yps(xps,ypl,nps=nps)#,sortit=1,method=2)
        par,cov = curve_fit(exp_3points, xps, yps, p0=np.array([0, -1, 1]), absolute_sigma=True)
        return xps,yps,exp_3points(xx,par[0],par[1],par[2])

    def plot_PolyFx(self,ax,xx,ypl,nps=2,var='mono',hztype='lin',method=2): 
        '''This function just plots some reference polynomials and functions to 
        illustrate possible variations to be assigned to the physical variables
        or to visualize them with respect to the actual variations set.
        We can directly work with this function from main program
        nps=2 is below hardcoded for monotonic method
        '''
        #fig = plt.figure()
        #ax = fig.gca()
        npoints=30
        xpl=[xx[0],xx[-1]]
        if hztype=='lin':xx = np.linspace(xpl[0], xpl[-1], num=npoints)

        #array of fixed limiting points, AT LEAST including limiting interval points
        xf, yf = np.array(xpl), np.array(ypl)

        if var == 'non-mono':
            #play with the location of the control points to get different variations
            #creates few nps points contained between limits xpl 
            xps = np.linspace(xpl[0],xpl[1],nps)
            yps=self.get_yps(xps,ypl,method=method,nps=nps)#method 2 is preferred by default

            #creates a polynomial function fitting previous points and passing
            # through given fixed points
            for order in [1,2,3,4]:
                myfx=self.fix_point_polyfit_fx(order, xps , yps, xf, yf)
                ax.plot(xx, myfx(xx), '-')
        if var == 'mono':
            #this method works best with nps=2 to deliver monotonic order-N polyn. functions 
            xps = np.linspace(xpl[0],xpl[1],2)  #here only 2 points to achieve monotonicity
            yps=self.get_yps(xps,ypl,method=2)#method 2 preferred by default
            with warnings.catch_warnings():#avoid printing polyfit  warnings
                warnings.simplefilter("ignore")
                for order in [1,2,3,4]:
                    myfx = np.poly1d(np.polyfit(xps, yps, order))
                    ax.plot(xx, myfx(xx), '-')
        if var == 'mint':#bump mimicking minimum of T
            xps = np.linspace(xpl[0],xpl[1],nps)
            yps=self.get_yps(xps,ypl,method=method,nps=nps)#method 2 is preferred by default
            for order in [1,2,3,4]:
                myfx=self.fix_point_polyfit_fx(order, xps , yps, xf, yf)         
                ax.plot(xx, myfx(xx), '-')

        if var == 'exp':   
            xps,yps=[],[]
            myfx=exp_2points(xx,xpl,ypl)#exponential for tau       
            ax.plot(xx, myfx, '-')
            #xps,yps,myfx=self.get_exp3points(xx,xpl,ypl,nps=3)#ypl is dlims['tau']
            #ax.plot(xx, myfx, '-')

        ax.plot(xps, yps, 'bo')
        ax.plot(xf, yf, 'ro')
        plt.show()

        return #ax


    def PolyFx(self,xx,ypl,nps=2,order=4,npoints=10,var='mono',method=2):
        '''Get order-N polynomial function connecting points with coords xpl ypl.
        Method=monotonic uses a hardcoded nps=2. Test variations
        with plot_PolyFx function.
        Coeffs given by polyfit are in descending order (x**o to x**0).
        xpl,ypl = [p1[0],p2[0]],[p1[1],p2[1]] --> for two points
        ''' 
        #xx = np.linspace(xpl[0], xpl[-1], num=npoints)
        #now xpl is obtained from xx (i.e. hz) directly. Hence, this routine can be simplified
        xpl=[xx[0],xx[-1]]

        #array of fixed limiting points, AT LEAST including limiting interval points
        xf, yf = np.array(xpl), np.array(ypl)
        
        if var == 'mono':
            xps = np.linspace(xpl[0],xpl[1],2)#few nps points between limits xpl 
            yps=self.get_yps(xps,ypl,method=2)#method 2 is preferred here

            #needs this to avoid printing polyfit  warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                myfx_eval = np.poly1d(np.polyfit(xps, yps, order))
                myfx=myfx_eval(xx)
        else:
            if var=='tau':myfx=exp_2points(xx,xpl,ypl)#exponential for tau  
            #exp3 not working properly:
            if var=='exp3':xps,yps,myfx=self.get_exp3points(xx,xpl,ypl)#ypl is dlims['tau']
            if var=='deltav':aux,myfx=self.fun_minT(0,ypl,hz=xx)#get minimum of temp function
            if var=='non-mono':#crazy non-monotonic
                xps = np.linspace(xpl[0],xpl[1],nps)#few nps points contained between limits xpl 
                yps=self.get_yps(xps,ypl,method=method,nps=nps)#method 2 is preferred by default
                #creates a polynom func fitting previous points and passing
                # through the given fixed points
                myfx_eval=self.fix_point_polyfit_fx(order, xps , yps, xf, yf)
                myfx=myfx_eval(xx)

        return myfx

    def check_limits(self,dlims,checkthis):
        #check the max and min values in input dict parameters.
        #this routine is more general than check_Bvals (now deleted)
        corresp={'B1':['Bx','B'],'B2':['By','thB'],'B3':['Bz','phB']}
        system=self.atmospheres[list(self.atmospheres.keys())[0]].coordinates_B
        for key in checkthis:
            kk=key#kk is to be inserted in dlims, key goes in dmm
            if key in ['B1','B2','B3']:
                if ( system == 'cartesian'):key=corresp[key][0] 
                else:key=corresp[key][1]    
            if (self.dmm[key][0] <= dlims[kk][0] <= self.dmm[key][2]) and (self.dmm[key][0] <= dlims[kk][1] <= self.dmm[key][2]):
                ok=1
            else:
                raise Exception("Atmosphere values out of range. Revise limits of parameter {0} in set_funcatm.".format(key))    

    def set_funcatm(self,dlims,hztype='lin',hzlims=None,orders=3,selected=[],**pkws):
        '''EDGAR:Set the parameters of every atmospheric cell assuming a functional polynomial
        variation between ini and end values given as input parameters in dlims. 
        zhlims was stored in model self.hzlims but here we can overwrite with args '''

        #tags in the order expected for building the matrix pars2D in the order expected for set_paramaters
        alls=['B1','B2','B3','tau','v','deltav','beta','a','j10','j20f']
        Ncells=self.n_chromospheres        

        if not selected:#empty list marks first call to set_funcatm
            if dlims.keys():#dict not empty
                for key in ['ff','nbar','v','beta']:#if pars omited in call,here we set them to default constant values
                    if dlims.get(key)==None:dlims[key]=[self.dmm[key][1]]*2  #dlims is now complete always
            else: # dict are empty
                raise Exception("Atmosphere cannot be set. I need a dict of parameters in set_funcatm.")
            
            self.check_limits(dlims,alls+['nbar']) #nbar should be allowed to be in alls
            selected=alls
            self.pars2D=np.zeros((len(selected),Ncells))     #initialize pars2D        
            ksel=np.arange(len(selected))#build ksel, array of indices associated to the selected pars in pars2D
        else:#calling set_funcatmos from mutates()
            self.check_limits(dlims,selected)#check only mutated keys
            ksel=np.zeros(len(selected),dtype=int)
            for kk,elem in enumerate(selected):ksel[kk]=alls.index(elem)


        if hzlims is not None:
            self.hzlims=hzlims #mutates can also reset self.hzlims
            self.hz=self.set_hz(hzlims=hzlims,hztype=hztype)    
        
        hz=self.hz#pointer for brevity
        p2D=self.pars2D#pointer for brevity
        self.dlims=dlims#remember dlims in case mutates() is called later
        
                    
        if type(orders) is int:orders=[orders]*len(alls) 

        if (Ncells > 2):    #build functions and values for the parameters in pars2D 
            for kk,key in enumerate(selected):#create polynomial variations
                if (dlims[key][0]==dlims[key][1]):#constant case
                    p2D[ksel[kk],:]=dlims[key][0]
                    if (orders[ksel[kk]]>0)&(self.verbose >= 1):warnings.warn("The quantity {0} is being forced to keep constant values.".format(key))
                else:
                    if key=='tau' or key=='deltav':#create exponential or minT functions
                        p2D[ksel[kk],:]=self.PolyFx(hz,[dlims[key][0],dlims[key][1]],order=orders[ksel[kk]],npoints=Ncells,var=key)
                    else:
                        p2D[ksel[kk],:]=self.PolyFx(hz,[dlims[key][0],dlims[key][1]],order=orders[ksel[kk]],npoints=Ncells)
        else:  #only 2-cell case (limiting cells)
            for kk,key in enumerate(selected):
                p2D[ksel[kk],:]=np.array( [dlims[key][0],dlims[key][1] ] )


        if len(selected)==len(alls):#set chromospheric cells from pars2D for all parameters
            for ii in range(Ncells):
                self.chromospheres[ii].set_pars(p2D[0:8,ii],dlims['ff'][1],j10=p2D[8,ii],j20f=p2D[9,ii],nbar=dlims['nbar'][1])
            #creates and fill the matrix of Hazel magnetic parameters for all points
            self.B2D=np.array((3,Ncells))
            if (self.chromospheres[0].coordinates_B == 'spherical'):self.B2D=p2D[0:3,:]
            else:self.B2D=self.chromospheres[0].just_B_Hazel(*p2D[0:3,:])

        else:#set chromospheric cells from pars2D all at once for given parameters            
            #reconstruct ALL B pars for ALL atm cells from mutated B pars
            Bhaz=None
            if ('B1' in selected) or ('B2' in selected) or ('B3' in selected):
                if (self.chromospheres[0].coordinates_B == 'spherical'):self.B2D=p2D[0:3,:]
                else:self.B2D=self.chromospheres[0].just_B_Hazel(*p2D[0:3,:])
                Bhaz=self.B2D
                for elem in ['B1','B2','B3']:
                    if elem in selected:selected.remove(elem)#remove magnetic keys from selected
            for i in range(Ncells):
                # nbar & ff do not change with height but nbar can mutate ...dlims['ff'][1]
                nonmag=list(p2D[3:10,i])+[dlims['nbar'][1]]#ff not included 
                self.chromospheres[i].reset_pars(selected,ksel,nonmag,p2D[0:3,i],mag=Bhaz[:,i])           


        if pkws['plotit']!=0:self.plot_funcatmos(dlims,hz,atmat=p2D,**pkws)
        #...,var=pkws['var'],method=pkws['method'])

        return



    def remove_unused_atmosphere(self):
        """
        Remove unused atmospheres
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
    
        """
        
        to_remove = []        
        for k, v in self.atmospheres.items():
            if (not v.active):
                to_remove.append(k)
                if (self.verbose >= 1):
                    self.logger.info('  - Atmosphere {0} deleted.'.format(k))
                
        for k in to_remove:
            self.atmospheres.pop(k)
                    

    def exit_hazel(self):
        for k, v in self.atmospheres.items():            
            if (v.type == 'chromosphere'):
                hazel_code._exit(v.index) 

    def add_topology(self, atmosphere_order,specname):
        """
        Add a new topology.
        EDGAR: A topology is always associated to a spectrum. Hence these two aspects should be
        related in such a way that we know exactly the atmospheres by the name of the spectrum 
        (i.e. by the name of the spectral region). If this is done, then in synthesize_spectral_region
        we dont need to run over all atmospheres, but only through those linked to the spectrum name.
        For carrying out the radiative transfer, we need then a routine get_transfer_path 
        that returns the list of atmosphere objects (order) in the self.order_atmospheres list
        associated to the spectrum name.
        

        Parameters
        ----------
        topology : str
            Topology
        
        Returns
        -------
        None

        """

        # Transform the order to a list of lists
        if (self.verbose >= 1):
            self.logger.info('  - {0}'.format(atmosphere_order))

        vertical_order = atmosphere_order.split('->')        
        order = []
        for k in vertical_order:
            name = k.strip().replace('(','').replace(')','').split('+')
            name = [k.strip() for k in name]
            
            tmp = []
            for n in name:
                if (n in self.atmospheres):#EDGAR:check that atmosphres in topology were add before
                    tmp.append(n)
                    self.atmospheres[n].active = True
                else:
                    raise Exception("Atmosphere {0} has not been add. Revise the slab names.".format(name))

            order.append(tmp)
        
        order_flat = [item for sublist in order for item in sublist]

        
        self.order_atmospheres.append(order)
        
        self.atms_in_spectrum[specname]=order  #new for making synthesize_spectral_region easier

        

    def normalize_ff(self):
        """
        Normalize all filling factors so that they add to one to avoid later problems.
        We use a softmax function to make sure they all add to one and can be unconstrained

        ff_i = exp(x_i) / sum(exp(x_i))

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        for atmospheres in self.order_atmospheres:
            for order in atmospheres:
                total_ff = 0.0
                for atm in order:            
                    ff = transformed_to_physical(self.atmospheres[atm].parameters['ff'], -0.00001, 1.00001)
                    total_ff += ff

                for atm in order:                            
                    ff = transformed_to_physical(self.atmospheres[atm].parameters['ff'], -0.00001, 1.00001)
                    self.atmospheres[atm].parameters['ff'] = ff / total_ff
                    self.atmospheres[atm].parameters['ff'] = physical_to_transformed(self.atmospheres[atm].parameters['ff'], -0.00001, 1.00001)

    def check_filling_factors(self, spectral_region):
        for n, order in enumerate(self.atms_in_spectrum[spectral_region] ): #n run layers along the ray
            if (len(order) > 1):#k runs subpixels of topologies c1+c2                                                  
                count=0
                for k, atm in enumerate(order):count+=self.atmospheres[atm].parameters['ff']
                if (count!=1.0):
                    print("WARNING: Filling factors of layer {0} do not add up to one. Assuming iso-contribution.".format(n))            
                    for k, atm in enumerate(order):self.atmospheres[atm].parameters['ff']=1.0/len(order)

    def synthesize_spectrum(self, spectral_region, method, stokes=None,stokes_out = None,fractional=False):
        """
        Synthesize chromospheres of spectral region and normalize to continuum of quiet Sun at disk center
        Stokes and stokes_out are local variables initialized in header (not intended to be inputs!).
        atms_in_spectrum makes unnecessary to check the asp spectral region inside the double loop below
        -----------Parameters:----------
        spectral_region : str.    Spectral region to synthesize
        method: synthesis method for solving the RTE
        fractional: to calculate emergent Stokes profiles as normalized to continuum or as divided by I(lambda)
        --------------------------------
        """        

        if method==5:dn=1
        else:dn=1
        #nsteps:integer number of blocks of "dn" cells
        #kind: index qunatifying the remaining cells. Can be 0,1,,..,dn-1
        nsteps,kind=np.divmod(self.n_chromospheres,dn)
        if nsteps==0:raise Exception("WARNING: Multistep RT methods require more points in height.")            
        if kind!=0:nsteps+=1#add the last step for the remaining cells
        #Multistep with step dn with Ncells-1 as last point:
        #step: ini:end
        #0: 0:dn  (for dn=3 : 0,1,2)
        #1: dn:2*dn
        #nk-1(last): (nk-1)*dn:(nk-1)*dn+kind

        #for n in range(nsteps): #n run layers along the ray
        for n, order in enumerate(self.atms_in_spectrum[spectral_region] ): #n run layers along the ray
            #update line_to_index in atm/hazel synthesize with that in add_spectral. 
            self.chromospheres[n].line_to_index=self.line_to_index
            for k, atm in enumerate(order):  #k runs subpixels of topologies c1+c2                              
                if (k != 0):raise Exception("WARNING: Subpixel components are not yet allowed in this Model version.")            

        #same for all Hazel chromospheres in a same ray, so can be outside the loop
        xbot, xtop = self.chromospheres[0].wvl_range
        
        for n in range(nsteps): #n run layers along the ray
            ini=n*dn
            end=ini+dn#(n+1)*dn
            if n==nsteps:
                end=ini+kind
                print(n,dn,kind,ini,end)
                sys.exit()

            sp=self.spectrum[spectral_region] #pointer for local compact notation
            sp.synmethods.append(method)#here method is already a number
            stokes,sp.eps[ini:end,:,:],sp.eta[ini:end,:,:],sp.stim[ini:end,:,:],error = \
            self.synth_piece(ini,end,method,stokes=stokes_out, nlte=self.use_nlte)#For single chromospheres
            #if (n > 0 ):Update boundary cond. for layers above bottom one      
            stokes_out = stokes[:,xbot:xtop] 
        #-------------------------------------------------------------------
        i0=hazel.util.i0_allen(np.mean(sp.wavelength_axis[xbot:xtop]), self.muAllen)  #at mean wavelength
        #i0=hazel.util.i0_allen(sp.wavelength_axis[xbot:xtop], self.muAllen)[None,:] #at each wavelength

        if fractional:i0=stokes[0,:] #when fractional, P(lambda)/I(lambda) will be stored in spectrum object
        sp.stokes[:,xbot:xtop] = stokes/ i0

    def set_nlte(self, option):
        """
        Set calculation of Ca II 8542 A to NLTE

        Parameters
        ----------
        option : bool
            Set to True to use NLTE, False to use LTE
        """
        self.use_nlte = option
        if (self.verbose >= 1):
            self.logger.info('Setting NLTE for Ca II 8542 A to {0}'.format(self.use_nlte))

    def synthesize(self, method=None,muAllen=1.0,frac=None,fractional=False,obj=None,plot=None,ax=None):
        """
        Synthesize atmospheres

        
        Returns
        -------
        None

        """
        if frac is True:fractional=frac #abreviated keyword to fractional

        self.muAllen=muAllen #mu where Allen continuum shall be taken for normalizing Stokes output 

        if (method is not None) and (method != self.methods_dicT[self.synmethod]):
            #print(method,self.methods_dicT[self.synmethod])
            print('Changing synthesis method to {0}.'.format(method))
            self.check_method(method)
            self.synmethod=self.methods_dicS[method]#pass from string label to number label and update self

        #EDGAR: WARNING,I think normalize_ff will not work for the synthesis. 
        #if (self.working_mode == 'inversion'):
        #    self.normalize_ff()
        #    fractional=False #always work with Stokes/Icont in inversions.

        for k, v in self.spectrum.items():#k is name of the spectrum or spectral region
            #EDGAR: checking correct filling factors in composed layers
            #TBD: this kind of check should be done during setup, not in calculations time 
            self.check_filling_factors(k)
                     
            self.synthesize_spectrum(k, self.synmethod) 
            #we never call synthesize with fractional=True to avoid storing the fractional
            #polarization in spectrum and thus avoid possible mistakes
            #the fractional polarization shall only be shown in plotting

            if (v.normalization == 'off-limb'):
                v.stokes /= np.max(v.stokes[0,:])

            if (v.psf_spectral is not None):                
                for i in range(4):
                    v.stokes[i,:] = scipy.signal.convolve(v.stokes[i,:], v.psf_spectral, mode='same', method='auto')
            
            if (v.interpolate_to_lr):
                for i in range(4):                
                    v.stokes_lr[i,:] = np.interp(v.wavelength_axis_lr, v.wavelength_axis, v.stokes[i,:])                    

            if (plot is not None):#plot called inside loops
                if (k == plot):self.plot_stokes(plot,fractional=fractional)
            else:#plot is None because synthesize routine was called without intention of plotting or from mutation
                if obj is None:TBD=1                
            
        #return ax

    def synth_piece(self,ini,end,method,stokes=None, nlte=None,epsout=None, etaout=None, stimout=None):
        """
        Carry out the synthesis and returns the Stokes parameters directly from python user main program.
        ----------Parameters----------
        method = synthesis method for Hazel
        stokes : float
        An array of size [4 x nLambda] with the input Stokes parameter.                
        -------Returns-------
        stokes : float
        Stokes parameters, with the first index containing the wavelength displacement and the remaining
                                    containing I, Q, U and V. Size (4,nLambda)        
        self.pars2D -> ['B1','B2','B3','tau','v','deltav','beta','a','j10','j20f']
        """
        dn=end-ini
        
        dn=3 #DElete
        end=ini+dn #delete

        BIn = np.asfortranarray(self.B2D[:,ini:end])# 3 x dn 
        
        #REMEMBER THAT ALL THE SELFS HERE WERE REFERRING TO A CELL ATMOSPHERE
        #BECAUSE THIS ROUTINE WAS IN CHROMOSPHERE.PY
        aself=self.chromospheres[ini]
        
        hIn = self.hz[ini:end]#aself.height#self.hz[ini:end]  #aself.height --> was single height for a given atmosphere cell 
        tau1In = self.pars2D[3,ini:end] #aself.parameters['tau']
        
        anglesIn = aself.spectrum.los
        transIn = aself.line_to_index[aself.active_line]  #This was defined in add_spectral/um
        mltp=aself.spectrum.multiplets #only for making code shorter below
        lambdaAxisIn = aself.wvl_axis - mltp[aself.active_line]        
        nLambdaIn = len(lambdaAxisIn)
        
        print(ini,dn,BIn[:,0],hIn, tau1In)

        '''
        # Renormalize nbar so that its CLV is the same as that of Allen, but with a decreased I0
        # If I don't do that, fitting profiles in the umbra is not possible. The lines become in
        # emission because the value of the source function, a consequence of the pumping radiation,
        # is too large. In this case, one needs to use beta to reduce the value of the source function.
        ratio = boundaryIn[0,0] / i0_allen(mltp[self.active_line], self.spectrum.mu)

        '''
        '''-----------EDGAR: THOUGHTS CONCERNING BOUNDARY CONDITIONS------------------------------
        
        Concerning the above comment and the following code...
        1) In the first layer of the transfer (where boundary cond. are applied), the value of ratio 
        ends up being the value of I set in boundary=[1,0,0,0], because one first multiplies the intensity
        boundary component by i0Allen, and later divides it again in ratio, hence it has no effect other
        than setting ratio to boundary[0]=1. This 1 is a normalization corresponding to the value 
        that the user wants to assume for the continuum intensity(e.g. 1 for Allen continuum at a given mu, 
        or smaller values for the lower continuum intensities such as that of an umbra). 
        Then, this value should be inferred or even inverted for fitting observations, 
        but in synthesis we provide it.

        2) The next block of code also sets the value of BoundaryIn for Hazel. The value of BoundaryIn 
        entering Hazel in the very first boundary layer was here the i0Allen with spectral dependence 
        because boundary=[1,0,0,0] is multiplied by i0Allen and broadcasted to wavelength in the spectrum
        methods subroutines. In upper layers the stokes in boundaryIn is the result of previous transfer 
        but the intensity value at the wings (boundaryIn[0,0,]) used to define 'ratio' is still the continuum intensity
        set to 1 by the user in the boundary keyword because transfer does not affect the far wings in absence
        of continuum opacity.

        3) Then, one could think of avoiding to multiply and divide by i0Allen by just setting ratio 
        directly to the spectrum.boundary for intensity set by the user(which is the fraction, normally 1, 
        of the i0Allen at mu; in a dark background, of course this number should be lower, but this is set 
        ad hoc by the user or ideally by the inversion code.)
        However, we have to multiply and divide by I0Allen in every step in order to use the updated boundaryIn of the 
        layer. Improving this is not important given the approximations associated to the anisotropy (see below).

        4)We want the possibility of defining the boundaryIn for Hazel as the spectrum of i0Allen 
        or, more generally, as spectral profiles introduced by the user for every Stokes parameter 
        (for instance coming from observations or from ad hoc profiles for experiments). 

        So we reuse the keyword boundary to allow the possibility 
        of containing a provided spectral dependence or just 4 numbers (that then would be broadcasted
        in wavelength as constant profiles as before).
        Complementing the boundary keyword, we define the new keyword i0fraction,
        which is just one number representing the fraction of i0Allen continuum for intensity 
        and for normalization of Stokes vector to Icontinuum, but this keyword .

        It seems correct to modify nbar as of the number of photons of continuum relative to I0Allen,
        but this is just an approximation limited by how Hazel includes the anisotropy. 
        The nbar does not belong to a layer, but to the rest of the 
        atmopshere that illuminates the layer because nbar is the number of photons used to estimate radiation field tensors in
        the SEE. One then can argue that nbar for only one layer is related to the background continuum
        but, for multiple layers, the lower layers can modify the nbar of subsequent upper layers, which 
        is inconsistent with using the same Allen anisotropy for all layers. 
        Hence we would like to estimate this variation, realizing also that the nbar is not the same 
        for line center than for the wings. As this theory is limited to zero order we just 
        use the continuum because the spectral variation is not considered (flat spectrum aproximation).
        The above inconsistency is an intrinsic limitation of Hazel that is directly associated to the way 
        we introduce the radiation field tensor (the anisotropy) in the calculations. In more general cases 
        we calculate the anisotropy layer by layer using the intensity(and polarization) that arrives to the layer after performing the 
        radiative transfer along surrounding shells, and thus the nbar comes implicitly set by the sourroundings 
        and modulated in the trasnfer.  
           
        Here, ratio only affects nbar (hence anisotropies), but not the transfer of stokes. 
        For the first layer, ratio=boundaryIn[0,0]/i0_allen gives a fraction without units as required for 
        ratio and nbar.boundaryIn[0,0] is later updated from previous output and the code does that fresh 
        division again for every subsequent layer. As boundaryIn[0,0] is the continuum intensity, it does not change 
        appreciably with the transfer in absence of continuum opacity (Hazel still does not have it anyways).
        But if the continuum opacity is introduced or the opacity of the previous layer
        was large in the wings, boundaryIn[0,0] decreases along the LOS, which decreases ratio too as the transfer advances.
        The only problem with this is that the reduction of ratio (hence of nbar) is the same for all 
        rays in the radiation field sphere, so anisotropic transfer is not considered,as explained above.
        In any case, as boundaryIn always has physical units in every step of the transfer, it is reasonable 
        to divide by I0Allen to get the reamaining number of photons per mode (the fraction nbar) at every layer. 
        
        '''
        #-------------------------------------------------
        #we multiply by i0Allen to get units right. When introducing ad hoc the boundary 
        #with spectral dependence, we shall do it normalized to I0Allen(instead of with physical units), 
        #so still multiplication by I0Allen is necessary here.
        #self.spectrum.boundary arrives already multiplied by i0fraction if necessary. 
        if (stokes is None):
            boundaryIn  = np.asfortranarray(np.zeros((4,nLambdaIn)))
            boundaryIn[0,:] = i0_allen(mltp[aself.active_line], aself.spectrum.mu) #hsra_continuum(mltp[self.active_line]) 
            boundaryIn *= aself.spectrum.boundary[:,aself.wvl_range[0]:aself.wvl_range[1]]
        else:            
            boundaryIn = np.asfortranarray(stokes)

        '''
        EDGAR: the value of boundary that enters here 
        must be Ibackground(physical units)/I0Allen (all input and output Stokes shall always be
        relative to the Allen I0 continuum). If the first value of this quantity is 1.0 then we have 
        an Allen background. Otherwise, that first value represents the true background 
        continuum realtive to I0Allen which can be be a fraction  of I0Allen (i.e., is the i0fraction
        introduced in model.py as tentative keyword). 

        For spectrum.boundary=1, the boundary intensity given to Hazel synthesize routine should be
        Iboundary = 1 *I0Allen(lambda), such that it has physical units (is not relative).

        spectrum.boundary is I0(physical)/I0Allen
        boundaryIn is then spectrum.boundary*I0Allen = I0(physical)
        then ratio=boundaryIn/I0Allen=I0(physical)/I0Allen , 
        which is a fraction of 1.0 (relative to the I0llen), as desired for nbarIn.
        '''
        ratio = boundaryIn[0,0]/ i0_allen(mltp[aself.active_line], aself.spectrum.mu)

        #nbarIn are reduction factors of nbar Allen for every transition! This means that it has 4 elements
        #for Helium and 2 for Sodium for instance, but this number was hardcoded to 4.
        #In addition omegaIn was wrong because it was put to zero, meaning no anisotropy,
        #while ones would mean that we use Allen for these pars.
        #when different than 0.0 and 1.0 they are used as modulatory factors in  hazel
        nbarIn = aself.nbar.vals * ratio #np.ones(self.ntr) * ratio
        omegaIn = aself.j20f.vals #np.ones(self.ntr)    #Not anymore np.zeros(4) 
        j10In = aself.j10.vals   #remind j10 and j20f are vectors (one val per transition).

        betaIn = aself.parameters['beta']      

        #-------------------------------------------------

        dopplerWidthIn = aself.parameters['deltav']
        dampingIn = aself.parameters['a']
        dopplerVelocityIn = aself.parameters['v']

        #Check where self.index is updated. It is index of current chromosphere,from 1 to n_chromospheres. 
        args = (aself.index, dn, method, BIn, hIn, tau1In, boundaryIn, transIn, anglesIn, nLambdaIn,
            lambdaAxisIn, dopplerWidthIn, dampingIn, j10In, dopplerVelocityIn,
            betaIn, nbarIn, omegaIn, aself.atompol,aself.magopt,aself.stimem,aself.nocoh,np.asarray(aself.dcol) )
        
        #2D opt coeffs yet (not height dependent), for current slab self.index
        l,stokes,epsout,etaout,stimout,error = hazel_code._synth(*args)

        if (error == 1):raise NumericalErrorHazel()

        ff = aself.parameters['ff'] #include it in the return below
        
        return ff * stokes, epsout,etaout,stimout,error #/ hsra_continuum(mltp[self.active_line])

