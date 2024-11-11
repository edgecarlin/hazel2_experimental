import numpy as np
import h5py
from asciitree import LeftAligned
from collections import OrderedDict
from asciitree.drawing import BoxStyle, BOX_DOUBLE, BOX_BLANK
import pickle, os
from timeit import default_timer as timer 

__all__ = ['aft','i0_allen', 'myfmt','cmap_center_adjust',
'cmap_powerlaw_adjust','cmpadjust','beauty2','savemodel','readmodel',
'save_RTcoeffs','read_RTcoeffs','_extract_parameter_cycles', 
'isint', 'fvoigt', 'lower_dict_keys', 'show_tree']

def aft(x):
    return np.asfortranarray(x)

def i0_allen(wavelength, muAngle):
    """
    Return the solar intensity at a specific wavelength and heliocentric angle
    wavelength: wavelength in angstrom
    muAngle: cosine of the heliocentric angle
    """
    C = 2.99792458e10
    H = 6.62606876e-27

    if (muAngle == 0):
        return 0.0

    lambdaIC = 1e4 * np.asarray([0.20,0.22,0.245,0.265,0.28,0.30,0.32,0.35,0.37,0.38,0.40,0.45,0.50,0.55,0.60,0.80,1.0,1.5,2.0,3.0,5.0,10.0])
    uData = np.asarray([0.12,-1.3,-0.1,-0.1,0.38,0.74,0.88,0.98,1.03,0.92,0.91,0.99,0.97,0.93,0.88,0.73,0.64,0.57,0.48,0.35,0.22,0.15])
    vData = np.asarray([0.33,1.6,0.85,0.90,0.57, 0.20, 0.03,-0.1,-0.16,-0.05,-0.05,-0.17,-0.22,-0.23,-0.23,-0.22,-0.20,-0.21,-0.18,-0.12,-0.07,-0.07])

    lambdaI0 = 1e4 * np.asarray([0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.48,0.50,0.55,0.60,0.65,0.70,0.75,\
        0.80,0.90,1.00,1.10,1.20,1.40,1.60,1.80,2.00,2.50,3.00,4.00,5.00,6.00,8.00,10.0,12.0])
    I0 = np.asarray([0.06,0.21,0.29,0.60,1.30,2.45,3.25,3.77,4.13,4.23,4.63,4.95,5.15,5.26,5.28,5.24,5.19,5.10,5.00,4.79,4.55,4.02,3.52,3.06,2.69,2.28,2.03,\
        1.57,1.26,1.01,0.81,0.53,0.36,0.238,0.160,0.078,0.041,0.0142,0.0062,0.0032,0.00095,0.00035,0.00018])
    I0 *= 1e14 * (lambdaI0 * 1e-8)**2 / C

    u = np.interp(wavelength, lambdaIC, uData)
    v = np.interp(wavelength, lambdaIC, vData)
    i0 = np.interp(wavelength, lambdaI0, I0)
    
    return (1.0 - u - v + u * muAngle + v * muAngle**2)* i0

'''-------------SOME ROUTINES HELPING PLOTS--------------------------------------'''
def myfmt(x, pos,numdec=0):  
    '''EDGAR:this routine makes '0.000..0' to look like 0 and 
    controls the format as a function of the most signficant decimal'''
    if x==0:return '0'
    if np.abs(x)<0.1 and x!=0:
        fmtstring='{:.'+str(numdec)+'e}'
        a, b = fmtstring.format(x).split('e')
        b = int(b)
        #return r'${} \cdot 10^{{{}}}$'.format(a, b)    
        return r'${} e^{{{}}}$'.format(a, b)#scientic notation  
    else:#if the second decimal is close to zero by two units or less this rounds to only 1 decimal prec
        if np.abs(100*x-10*int(10*x))<2:return '{:1.1f}'.format(x)
        else:return '{:.2f}'.format(x)

def cmap_powerlaw_adjust(cmap, a):
    '''
    returns a new colormap based on the one given
    but adjusted via power-law:
    newcmap = oldcmap**a
    '''
    import copy
    from matplotlib import colors
    if a < 0.:
        return cmap
    cdict = copy.copy(cmap._segmentdata)
    fn = lambda x : (x[0]**a, x[1], x[2])
    for key in ('red','green','blue'):
        ll=[]
        for elem in cdict[key]:ll.append(fn(elem)) #apply the function to each tuple value
        cdict[key]=sorted(ll)  #order the tuples and store again in the color table
        #cdict[key] = map(fn, cdict[key])
        #cdict[key].sort()
        #assert cdict[key][0]<0 or cdict[key][-1]>1, "Resulting indices extend out of the [0, 1] segment."
    return colors.LinearSegmentedColormap('colormap',cdict,1024)

def cmap_center_adjust(cmap, center_ratio):
    '''
    returns a new colormap based on the one given
    but adjusted so that the old center point higher
    (>0.5) or lower (<0.5)
    '''
    import math
    if not (0. < center_ratio) & (center_ratio < 1.):
        return cmap
    a = math.log(center_ratio) / math.log(0.5)
    return cmap_powerlaw_adjust(cmap, a)

def cmpadjust(cmap, range, center):
    '''
    cmap_center_point_adjust:
    converts center to a ratio between 0 and 1 of the
    range given and calls cmap_center_adjust(). Returns
    a new adjusted colormap accordingly.
    '''
    if not ((range[0] < center) and (center < range[1])):
        return cmap
    #print (abs(center - range[0]) / abs(range[1] - range[0]) )
    new= cmap_center_adjust(cmap,
        abs(center - range[0]) / abs(range[1] - range[0]))
    return new

def beauty2(ax,lims,sxy,lpxy,xlab,ylab,tit,xl=0,yl=0,xdiv=0,sym='n',norm='n',
    xticks='',yticks='',nbx=4,prx='lower',nby=0,pry='lower',xtitex='n'):
    #font = {'family' : 'normal','weight' : 'bold','size': 22}
    #if font != {'':}:matplotlib.rc('font', **font)
    from matplotlib.ticker import MaxNLocator

    if yl==1:ax.set_yscale('log')       
    if xl==1:ax.set_xscale('log')       
    if lims!=[]:
        if lims[0]!=lims[1]:ax.set_xlim(lims[0],lims[1])    
        if lims[2]!=lims[3]:ax.set_ylim(lims[2],lims[3])
        if xdiv!=0:ax.xaxis.set_ticks(np.arange(lims[0],lims[1],(lims[1]-lims[0])/xdiv))    
    if sym=='y':
        allines=np.array([0,0])
        for line in ax.lines:allines=np.concatenate((allines,line.get_ydata()))
        mx=np.max(np.abs(allines))
        if norm=='y':
            for line in ax.lines:line.set_ydata(line.get_ydata()/mx)
            mx=1.0
        ax.set_ylim(-mx,mx)

    ax.tick_params(axis='y',pad=lpxy[1])
    if ylab!='':
        ax.set_ylabel(ylab,size=sxy[0],labelpad=lpxy[1])
        
    ax.tick_params(axis='x',pad=lpxy[0])
    if xlab!='':
        ax.set_xlabel(xlab,size=sxy[0],labelpad=lpxy[0])
        
    if tit!='':
        ax.set_title(tit,size=sxy[2])
        ttt=ax.title
        ttt.set_position([.5, 0.99])
    if sxy[1]!=0:
        for elem in [ax.xaxis,ax.yaxis]:elem.set_tick_params(which='major',labelsize=sxy[1])

    if nbx!=0:ax.xaxis.set_major_locator( MaxNLocator(nbins = nbx, prune = prx) )
    if nby!=0:ax.yaxis.set_major_locator( MaxNLocator(nbins = nby, prune = pry) )

    if xtitex=='y':  #ticks are literally text#
        xticktex=xticks
        xticks=[float(val) for val in xticks]

    if xticks!='' and xticks!='no':ax.set_xticks(xticks)# ,fontsize=9)
    if yticks!='' and yticks!='no':ax.set_yticks(yticks)# ,fontsize=9)
    
    if xtitex=='y':ax.set_xticklabels(xticktex)
    if xticks=='no':
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticklabels(['']*len(labels))
        ax.set_xticks([])
    
    if yticks=='no':
        labels = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_yticklabels(['']*len(labels))
        ax.set_yticks([])
    return ax

'''..................................................................'''

''' Place here saving/restoring routines to access them without need of loading a model'''
mydir='saved_data/'

def savemodel(vars,fname,dir=mydir,description='Read this description'):
    start=timer()
    vars.append(description)
    with open(dir+fname, 'wb') as fi:
        pickle.dump(vars, fi, protocol=pickle.HIGHEST_PROTOCOL)
    #print('File size: {0} kbytes'.format(os.path.getsize(dir+fname)/1024.0))
    #print('Saved in {0} seconds'.format(timer()-start))
    print("Saved to file: {0:{pp}} kb in: {1:{pp}} s\n".format(os.path.getsize(dir+fname)/1024.0,
        timer()-start,pp='11.3f'))

def readmodel(fname,dir=mydir):
    start=timer()
    with open(dir+fname, 'rb') as fi:
        model_atmdic_desc_list = pickle.load(fi)
    end=timer()
    print("Read in {0:{pp}} s.\n".format(timer()-start,pp='11.4f'))

    return model_atmdic_desc_list

def save_RTcoeffs(mm,sp,dlims,fname,description='Add a description',dir=mydir):
    packed=[mm.spectrum[sp].rteps,mm.spectrum[sp].rteta,mm.spectrum[sp].rtrho,dlims]
    savemodel(packed,fname,dir=dir,description=description)

def read_RTcoeffs(fname,dir=mydir):
    #eps,eta,rho,dlims,des=readmodel(fname,dir=dir)
    return readmodel(fname,dir=dir)

''' ----------------------------------------------------------------------------------'''


def _extract_parameter_cycles(s):
    tmp = s[0].split('->')
    value = float(tmp[0])
    cycle1 = tmp[1].strip()
    cycles = [cycle1] + s[1:]

    return value, cycles

def isint(str):
    try:
        int(str)
        return True
    except ValueError:
        return False

def isfloat(str):
    if (str is None):
        return False
    try:
        float(str)
        return True
    except ValueError:
        return False

def toint(l):
    return [int(x) if isint(x) else x for x in l]

def tofloat(l):
    return [float(x) if isfloat(x) else None for x in l]

def tobool(l):
    return True if l == 'True' else False

def onlyint(l):
    return [i for i in l if isinstance(i, int)]


def fvoigt(damp,v):
    
    """
    Fast implementation of the Voigt-Faraday function

    Parameters
    ----------
        damp : float
            damping parameter

        v : float
            normalized wavelength (lambda-lambda0) / sigma
        
    Returns
    -------
        voigt, faraday : float
            Value of the Voigt and Faraday functions


    Notes
    ----- 
        A rational approximation to the complex error function is used
        after Hui, Armstrong, and Wray(1978, JQSRT 19, 509). H and F are 
        the real and imaginary parts of such function, respectively.
        The procedure is inspired on that in SIR (Ruiz Cobo & del Toro 
        Iniesta 1992, ApJ 398, 385). On its turn, that routine was taken
        from modifications by A. Wittmann (1986) to modifications by S.K.
        Solanki (1985) to an original FORTRAN routine written by J.W. Harvey
        and A. Nordlund.
    """
    
    A = [122.607931777104326, 214.382388694706425, 181.928533092181549,\
        93.155580458138441, 30.180142196210589, 5.912626209773153,\
        0.564189583562615]

    B = [122.60793177387535, 352.730625110963558, 457.334478783897737,\
        348.703917719495792, 170.354001821091472, 53.992906912940207,\
        10.479857114260399,1.]

    z = np.array(damp*np.ones(len(v)) + -abs(v)*1j)

    Z = ((((((A[6]*z+A[5])*z+A[4])*z+A[3])*z+A[2])*z+A[1])*z+A[0])/\
    (((((((z+B[6])*z+B[5])*z+B[4])*z+B[3])*z+B[2])*z+B[1])*z+B[0])

    h = np.real(Z)
    f = np.sign(v)*np.imag(Z)*0.5

    return h, f


def lower_dict_keys(d):
    out = {}
    for k, v in d.items():
        out[k.lower()] = v
    return out

def show_tree(hdf5_file):
    tree = {hdf5_file: OrderedDict()}

    f = h5py.File(hdf5_file, 'r')
    for k, v in f.items():
        tree[hdf5_file][k] = OrderedDict()
        for k2, v2 in v.items():
            tree[hdf5_file][k][f'{k2} -> {v2.shape}  {v2.dtype}'] = OrderedDict()            

    chrs = dict(
            UP_AND_RIGHT=u"\u2514",
            HORIZONTAL=u"\u2500",
            VERTICAL=u"\u2502",
            VERTICAL_AND_RIGHT=u"\u251C"
        )

    tr = LeftAligned(draw=BoxStyle(gfx = chrs, horiz_len=1))
    print(tr(tree))