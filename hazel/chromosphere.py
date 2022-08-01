from collections import OrderedDict
import numpy as np
import os
from hazel.atmosphere import General_atmosphere
from hazel.util import i0_allen
from hazel.codes import hazel_code
from hazel.hsra import hsra_continuum
from hazel.io import Generic_hazel_file
from hazel.exceptions import NumericalErrorHazel
import copy


__all__ = ['Hazel_atmosphere']

class Hazel_atmosphere(General_atmosphere):
    def __init__(self, working_mode, name=''):#, atom='helium'):
    
        super().__init__('chromosphere', name=name)#, atom=atom)

        self.height = 3.0
        self.working_mode = working_mode
        #self.atom = atom

        self.parameters['Bx'] = 0.0
        self.parameters['By'] = 0.0
        self.parameters['Bz'] = 0.0
        self.parameters['B'] = 0.0
        self.parameters['thB'] = 0.0
        self.parameters['phiB'] = 0.0
        self.parameters['tau'] = 1.0
        self.parameters['v'] = 0.0
        self.parameters['deltav'] = 8.0
        self.parameters['beta'] = 1.0
        self.parameters['a'] = 0.0
        #EDGAR. j10 to be defined as independent elements, not vector:
        self.parameters['j10'] = np.zeros(4)  
        self.parameters['ff'] = np.log(1.0)
        

        self.units['Bx'] = 'G'
        self.units['By'] = 'G'
        self.units['Bz'] = 'G'
        self.units['B'] = 'G'
        self.units['thB'] = 'deg'
        self.units['phiB'] = 'deg'
        self.units['tau'] = ''
        self.units['v'] = 'km/s'
        self.units['deltav'] = 'km/s'
        self.units['beta'] = ''
        self.units['a'] = ''
        self.units['j10'] = '%' #EDGAR
        self.units['ff'] = ''
        

        self.nodes['Bx'] = 0.0
        self.nodes['By'] = 0.0
        self.nodes['Bz'] = 0.0        
        self.nodes['B'] = 0.0
        self.nodes['thB'] = 0.0
        self.nodes['phiB'] = 0.0
        self.nodes['tau'] = 0.0
        self.nodes['v'] = 0.0
        self.nodes['deltav'] = 0.0
        self.nodes['beta'] = 0.0
        self.nodes['a'] = 0.0
        self.nodes['j10'] = 0.0  #EDGAR
        self.nodes['ff'] = 0.0
        

        self.nodes_location['Bx'] = None
        self.nodes_location['By'] = None
        self.nodes_location['Bz'] = None       
        self.nodes_location['B'] = None
        self.nodes_location['thB'] = None
        self.nodes_location['phiB'] = None
        self.nodes_location['tau'] = None
        self.nodes_location['v'] = None
        self.nodes_location['deltav'] = None
        self.nodes_location['beta'] = None
        self.nodes_location['a'] = None
        self.nodes_location['j10'] = None  #EDGAR
        self.nodes_location['ff'] = None
        

        self.error['Bx'] = 0.0
        self.error['By'] = 0.0
        self.error['Bz'] = 0.0
        self.error['B'] = 0.0
        self.error['thB'] = 0.0
        self.error['phiB'] = 0.0
        self.error['tau'] = 0.0
        self.error['v'] = 0.0
        self.error['deltav'] = 0.0
        self.error['beta'] = 0.0
        self.error['a'] = 0.0
        self.error['j10'] = np.zeros(4) #EDGAR
        self.error['ff'] = 0.0 
        

        self.n_nodes['Bx'] = 0
        self.n_nodes['By'] = 0
        self.n_nodes['Bz'] = 0        
        self.n_nodes['B'] = 0
        self.n_nodes['thB'] = 0
        self.n_nodes['phiB'] = 0
        self.n_nodes['tau'] = 0
        self.n_nodes['v'] = 0
        self.n_nodes['deltav'] = 0
        self.n_nodes['beta'] = 0
        self.n_nodes['a'] = 0
        self.n_nodes['j10'] = np.zeros(4) #EDGAR
        self.n_nodes['ff'] = 0
        
        
        self.ranges['Bx'] = None
        self.ranges['By'] = None
        self.ranges['Bz'] = None
        self.ranges['B'] = None
        self.ranges['thB'] = None
        self.ranges['phiB'] = None
        self.ranges['tau'] = None
        self.ranges['v'] = None
        self.ranges['deltav'] = None
        self.ranges['beta'] = None
        self.ranges['a'] = None
        self.ranges['j10'] = None #EDGAR
        self.ranges['ff'] = None
        

        self.cycles['Bx'] = None
        self.cycles['By'] = None
        self.cycles['Bz'] = None  
        self.cycles['B'] = None
        self.cycles['thB'] = None
        self.cycles['phiB'] = None
        self.cycles['tau'] = None
        self.cycles['v'] = None
        self.cycles['deltav'] = None
        self.cycles['beta'] = None
        self.cycles['a'] = None
        self.cycles['j10'] = None  #EDGAR
        self.cycles['ff'] = None
        

        self.epsilon['Bx'] = 0.01
        self.epsilon['By'] = 0.01
        self.epsilon['Bz'] = 0.01
        self.epsilon['B'] = 0.01
        self.epsilon['thB'] = 0.01
        self.epsilon['phiB'] = 0.01
        self.epsilon['tau'] = 0.01
        self.epsilon['v'] = 0.01
        self.epsilon['deltav'] = 0.01
        self.epsilon['beta'] = 0.01
        self.epsilon['a'] = 0.01
        self.epsilon['j10'] = 0.01 #EDGAR
        self.epsilon['ff'] = 0.01
        

        self.jacobian['Bx'] = 1.0
        self.jacobian['By'] = 1.0
        self.jacobian['Bz'] = 1.0
        self.jacobian['B'] = 1.0
        self.jacobian['thB'] = 1.0
        self.jacobian['phiB'] = 1.0
        self.jacobian['tau'] = 1.0
        self.jacobian['v'] = 1.0
        self.jacobian['deltav'] = 1.0
        self.jacobian['beta'] = 1.0
        self.jacobian['a'] = 1.0
        self.jacobian['j10'] = 1.0  #EDGAR
        self.jacobian['ff'] = 1.0
        

        self.regularization['Bx'] = None
        self.regularization['By'] = None
        self.regularization['Bz'] = None
        self.regularization['B'] = None
        self.regularization['thB'] = None
        self.regularization['phiB'] = None
        self.regularization['tau'] = None
        self.regularization['v'] = None
        self.regularization['deltav'] = None
        self.regularization['beta'] = None
        self.regularization['a'] = None
        self.regularization['j10'] = None  #EDGAR
        self.regularization['ff'] = None
        

    def select_coordinate_system(self):
        #EDGAR: if you choose one coordB, this pops the other one out
        if (self.coordinates_B == 'cartesian'):
            labels = ['B', 'thB', 'phiB']
        if (self.coordinates_B == 'spherical'):
            labels = ['Bx', 'By', 'Bz']
        
        for label in labels:            
            self.parameters.pop(label)
            # self.units.pop(label)
            # self.nodes.pop(label)
            # self.error.pop(label)
            # self.n_nodes.pop(label)
            # self.ranges.pop(label)
            # self.cycles.pop(label)
            # self.jacobian.pop(label)
            # self.regularization.pop(label)
    

    def add_active_line(self, spectrum, wvl_range):#EDGAR def add_active_line(self, line, spectrum, wvl_range):

        """
        Add an active lines in this atmosphere
        
        Parameters
        ----------        
        lines : str
            Line to activate: ['10830','5876']
        spectrum : Spectrum
            Spectrum object
        wvl_range : float
            Vector containing wavelength range over which to synthesize this line
        
        Returns
        -------
        None
    
        EDGAR: number of transitions is already in atom%ntran
        """      
        #EDGAR:removed this block from here once line_to_index is defined in add_spectral
        #if (self.atom == 'helium')  :
        #    self.line_to_index = {'10830': 1, '3888': 2, '7065': 3,'5876': 4}
        #if (self.atom == 'sodium')  :
        #    self.line_to_index = {'5895': 1, '5889': 2}
            

        self.active_line =spectrum.lineHazel # line
        self.wavelength_range = wvl_range
        ind_low = (np.abs(spectrum.wavelength_axis - wvl_range[0])).argmin()
        ind_top = (np.abs(spectrum.wavelength_axis - wvl_range[1])).argmin()

        self.spectrum = spectrum
        self.wvl_axis = spectrum.wavelength_axis[ind_low:ind_top+1]
        self.wvl_range = [ind_low, ind_top+1]


    def cartesian_to_spherical(self,Bx,By,Bz):
        # Transform to spherical components in the vertical reference frame which are those used in Hazel
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        if (B == 0):
            thetaB = 0.0
        else:
            thetaB = 180.0 / np.pi * np.arccos(Bz / B)
        phiB = 180.0 / np.pi * np.arctan2(By, Bx)
    
        return B, thetaB,phiB

    def los_to_vertical(self,Bx_los,By_los,Bz_los):
        Bx_vert = Bx_los * self.spectrum.mu + Bz_los * np.sqrt(1.0 - self.spectrum.mu**2)
        By_vert = By_los
        Bz_vert = -Bx_los * np.sqrt(1.0 - self.spectrum.mu**2) + Bz_los * self.spectrum.mu

        return Bx_vert,By_vert,Bz_vert


    def spherical_to_cartesian(self,B,thetaB,phiB):
        Bx = B * np.sin(thetaB * np.pi / 180.0) * np.cos(phiB * np.pi / 180.0)
        By = B * np.sin(thetaB * np.pi / 180.0) * np.sin(phiB * np.pi / 180.0)
        Bz = B * np.cos(thetaB * np.pi / 180.0)

        return Bx,By,Bz        


    def get_B_Hazel(self,B1,B2,B3):#pars[0],pars[1],pars[2]
        '''
        Transform magnetic field parameters to vertical reference frame 
        in spherical coordinates, which is the Hazel working system.
        Output provides both cartesian and spherical coordinates
        in case both are needed (I would leave just spherical).
        '''
        if (self.coordinates_B == 'cartesian'):
            # Cartesian - LOS : just carry out the rotation in cartesian geometry and transform to spherical
            if (self.reference_frame == 'line-of-sight'):
                Bx,By,Bz=self.los_to_vertical(B1,B2,B3) #here Bi would be cartesian Bx,By,Bz in LOS
            else:# Cartesian - vertical : do nothing
                Bx,By,Bz=B1,B2,B3

            # Transform to spherical components in the vertical reference frame which are those used in Hazel
            B,thB,phiB=self.cartesian_to_spherical(Bx,By,Bz) #from cartesian vertical to spherical vertical

        if (self.coordinates_B == 'spherical'):
            # Spherical - vertical by default: do nothing but get cartesians .
            # we assume vertical but result is overwritten in next block if it is not like that
            Bx,By,Bz=self.spherical_to_cartesian(B1,B2,B3) #delete this line if Bx,By,Bz not needed in the code 
            B,thB,phiB= B1,B2,B3

            # Spherical - LOS : transform to cartesian, do rotation to vertical and come back to spherical
            if (self.reference_frame == 'line-of-sight'):
                Bx_los,By_los,Bz_los=self.spherical_to_cartesian(B,thB,phiB)
                Bx,By,Bz=self.los_to_vertical(Bx_los,By_los,Bz_los)
                # To spherical comps in vertical frame as required in Hazel
                B,thB,phiB=self.cartesian_to_spherical(Bx,By,Bz)

        return B,thB,phiB,Bx,By,Bz #Bx,By,Bz can be deleted if not needed in the code 

    def set_parameters(self, pars, ff=1.0,j10=np.zeros(4),m=None):
        """
        Set the parameters of this model chromosphere

        Parameters 
        ----------
        pars : list of float
            This list contains the following parameters in order: Bx, By, Bz, tau, v, delta, beta, a

        ff : float optional keyword
            Filling factor

        #EDGAR: 
        j10: array of doubles optional keyword, in percentage units. It is a vector with Ntransitions length

        Returns
        -------
        None
        """
        
        self.parameters['B'],self.parameters['thB'],self.parameters['phiB'], \
        self.parameters['Bx'],self.parameters['By'],self.parameters['Bz']= \
        self.get_B_Hazel(pars[0],pars[1],pars[2]) 
        

        #---DELETE THIS BLOCK AFTER VALIDATION----------------------------------------------------
        '''
        #EDGAR CHECK: this block seems redundant to the block in synthesize!
        #if cartesian input read it, compute spherical and update spherical parameters
        if (self.coordinates_B == 'cartesian'): 
            self.parameters['Bx'] = pars[0]
            self.parameters['By'] = pars[1]
            self.parameters['Bz'] = pars[2]

            # Now compute the spherical components
            B = np.sqrt(self.parameters['Bx']**2 + self.parameters['By']**2 + self.parameters['Bz']**2)
            if (B == 0):
                thetaB = 0.0
            else:
                thetaB = 180.0 / np.pi * np.arccos(self.parameters['Bz'] / B)
            phiB = 180.0 / np.pi * np.arctan2(self.parameters['By'], self.parameters['Bx'])

            self.parameters['B'] = B
            self.parameters['thB'] = thetaB
            self.parameters['phiB'] = phiB
        
        #if spherical input, read it, compute cartesian and update cartesian parameters
        if (self.coordinates_B == 'spherical'):
            self.parameters['B'] = pars[0]
            self.parameters['thB'] = pars[1]
            self.parameters['phiB'] = pars[2]

            # Now compute the cartesian components
            Bx = self.parameters['B'] * np.sin(self.parameters['thB'] * np.pi / 180.0) * np.cos(self.parameters['phiB'] * np.pi / 180)
            By = self.parameters['B'] * np.sin(self.parameters['thB'] * np.pi / 180.0) * np.sin(self.parameters['phiB'] * np.pi / 180)
            Bz = self.parameters['B'] * np.cos(self.parameters['thB'] * np.pi / 180.0)

            self.parameters['Bx'] = Bx
            self.parameters['By'] = By
            self.parameters['Bz'] = Bz
        
        '''
        self.parameters['tau'] = pars[3]
        self.parameters['v'] = pars[4]
        self.parameters['deltav'] = pars[5]
        self.parameters['beta'] = pars[6]
        self.parameters['a'] = pars[7]
        #EDGAR: j10 is not just a number, but a vector with Ntransitions length.
        self.parameters['j10'] = np.array(j10)#from list to array of doubles   
        self.parameters['ff'] = ff
        

        #EDGAR: I think this is not being used because ranges are defined to none above.
        # Check that parameters are inside borders by clipping inside the interval with a border of 1e-8
        if (self.working_mode == 'inversion'):
            for k, v in self.parameters.items():
                if (self.ranges[k] is not None):
                    self.parameters[k] = np.clip(v, self.ranges[k][0] + 1e-8, self.ranges[k][1] - 1e-8)
        
        #EDGAR: this avoids calling it in main user script everytime after
        #doing set_parameters but it is inefficient 
        #if (m is not None):m.synthesize()  

    #EDGAR: alias to set_parameters.:
    def set_pars(self, pars,ff=1.0,j10=np.zeros(4),m=None):
        return self.set_parameters(pars,ff,j10=j10,m=m)

    def load_reference_model(self, model_file, verbose):
        """
        Load a reference model or a model for every pixel for synthesis/inversion

        Parameters
        ----------
        
        model_file : str
            String with the name of the file. Extensions can currently be "1d" or "h5"
        verbose : bool
            verbosity flag

        Returns
        -------
            None
        """
        extension = os.path.splitext(model_file)[1][1:]
        
        if (extension == '1d'):
            if (verbose >= 1):
                self.logger.info('    * Reading 1D model {0} as reference'.format(model_file))
            self.model_type = '1d'
            self.model_filename = model_file          
        
        if (extension == 'h5'):
            if (verbose >= 1):
                self.logger.info('    * Reading 3D model {0} as reference'.format(model_file))
            self.model_type = '3d'
            

        self.model_handler = Generic_hazel_file(model_file)
        self.model_handler.open()
        out, ff = self.model_handler.read(pixel=0)
        self.model_handler.close()

        self.set_parameters(out, ff)

        self.init_reference(check_borders=True)

    def nodes_to_model(self):
        """
        Transform from nodes to model
        
        Parameters
        ----------
        None
                                
        Returns
        -------
        None
        """         
        for k, v in self.nodes.items():            
            # if (self.n_nodes[k] > 0):                
            self.parameters[k] = self.reference[k] + np.squeeze(self.nodes[k])
            # else:                
                # self.parameters[k] = self.reference[k]            
                            
    def print_parameters(self, first=False, error=False):
        if (self.coordinates_B == 'cartesian'):
            self.logger.info("     {0}        {1}        {2}        {3}       {4}       {5}      {6}      {7}      {8}".format('Bx', 'By', 'Bz', 'tau', 'v', 'deltav', 'beta', 'a', 'j10'))
            self.logger.info("{0:8.3f}  {1:8.3f}  {2:8.3f}  {3:8.3f}  {4:8.3f}  {5:8.3f}  {6:8.3f}  {7:8.3f}  {7:8.3f}".format(self.parameters['Bx'], self.parameters['By'], self.parameters['Bz'], self.parameters['tau'], 
                self.parameters['v'], self.parameters['deltav'], self.parameters['beta'], self.parameters['a'], self.parameters['j10']))
            
            if (error):            
                self.logger.info("{0:8.3f}  {1:8.3f}  {2:8.3f}  {3:8.3f}  {4:8.3f}  {5:8.3f}  {6:8.3f}  {7:8.3f}  {7:8.3f}".format(self.error['Bx'], self.error['By'], self.error['Bz'], self.error['tau'], 
                self.error['v'], self.error['deltav'], self.error['beta'], self.error['a'], self.error['j10']))
        
        if (self.coordinates_B == 'spherical'):
            self.logger.info("     {0}        {1}        {2}        {3}       {4}       {5}      {6}      {7}      {8}".format('B', 'thB', 'phiB', 'tau', 'v', 'deltav', 'beta', 'a', 'j10'))
            self.logger.info("{0:8.3f}  {1:8.3f}  {2:8.3f}  {3:8.3f}  {4:8.3f}  {5:8.3f}  {6:8.3f}  {7:8.3f}".format(self.parameters['B'], self.parameters['thB'], self.parameters['phiB'], self.parameters['tau'], 
                self.parameters['v'], self.parameters['deltav'], self.parameters['beta'], self.parameters['a'], self.parameters['j10']))
            
            if (error):            
                self.logger.info("{0:8.3f}  {1:8.3f}  {2:8.3f}  {3:8.3f}  {4:8.3f}  {5:8.3f}  {6:8.3f}  {7:8.3f}  {7:8.3f}".format(self.error['B'], self.error['thB'], self.error['phiB'], self.error['tau'], 
                self.error['v'], self.error['deltav'], self.error['beta'], self.error['a'], self.error['j10']))
        

    def synthesize(self,stokes=None, returnRF=False, nlte=None,epsout=None, etaout=None, stimout=None):
        """
        Carry out the synthesis and returns the Stokes parameters directly from python user main program.
        EDGAR:This synthesize of model.py, it does not see n_chromospheres. 

        Parameters
        ----------
        stokes : float
            An array of size [4 x nLambda] with the input Stokes parameter.
                        
        Returns
        -------        
        stokes : float
            Stokes parameters, with the first index containing the wavelength displacement and the remaining
                                    containing I, Q, U and V. Size (4,nLambda)        
        """

        if (self.working_mode == 'inversion'):
            self.nodes_to_model()            
            self.to_physical()


        #--------DELETE ALL THIS BLOCK WHEN VALIDATED---------------------------------------------------------
        #EDGAR : I MOVED ALL THIS BLOCK WAS MOVED TO ADD_PARAMETERS AVOIDING THE REDUNDANT
        #OPERATIONS DONE THERE. PLACE THIS BLOCK THERE ONE GETS HAVING BOTH TYPES OF COORDINATES
        #AND ALSO THE MAGNETIC FIELD IN SPHERICAL COORDINATES AND IN VERTICAL FRAME, AS REQUIRED BY HAZEL
        '''
        # Magnetic field components are entered in Hazel in the vertical reference system
        # Here we do the transformation from one to the other if fields are given in LOS
        if (self.coordinates_B == 'cartesian'):
            # Cartesian - LOS : just carry out the rotation in cartesian geometry and transform to spherical
            if (self.reference_frame == 'line-of-sight'):
                Bx,By,Bz=self.los_to_vertical(self.parameters['Bx'],self.parameters['By'],self.parameters['Bz'])
            else:
            # Cartesian - vertical : just transform to spherical
                Bx = self.parameters['Bx']
                By = self.parameters['By']
                Bz = self.parameters['Bz']

            # Transform to spherical components in the vertical reference frame which are those used in Hazel
            B,thetaB,phiB=self.cartesian_to_spherical(Bx,By,Bz) #in vertical frame

        if (self.coordinates_B == 'spherical'):
            # Spherical - vertical : do nothing
            B = self.parameters['B']
            thetaB = self.parameters['thB']
            phiB = self.parameters['phiB']

            # Spherical - LOS : transform to cartesian, do rotation to vertical and come back to spherical
            if (self.reference_frame == 'line-of-sight'):
                Bx_los,By_los,Bz_los=self.spherical_to_cartesian(B,thetaB,phiB)
                Bx_vert,By_vert,Bz_vert=self.los_to_vertical(Bx_los,By_los,Bz_los)
                # To spherical comps in vertical frame as required in Hazel
                B,thetaB,phiB=self.cartesian_to_spherical(Bx_vert,By_vert,Bz_vert)
        

        B1Input = np.asarray([B,thetaB,phiB])
        '''
        #--------------------------------------------------------------------------------
        
        B1In = np.asarray([self.parameters['B'],self.parameters['thB'], self.parameters['phiB']])

        hIn = self.height
        tau1In = self.parameters['tau']
        anglesIn = self.spectrum.los

        ##EDGAR:haz llegar line_to_index y multiplets hasta aqui una vez que los hayas puesto en add_spectral
        transIn = self.line_to_index[self.active_line]  
        mltp=self.spectrum.multiplets #only for making code shorter below
        lambdaAxisIn = self.wvl_axis - mltp[self.active_line]
        
        nLambdaIn = len(lambdaAxisIn)
                        
        #So confusing, there are two updates fo boundary condition, here and in synthesize_spectral_region
        if (stokes is None):
            boundaryIn  = np.asfortranarray(np.zeros((4,nLambdaIn)))
            # boundaryIn[0,:] = hsra_continuum(mltp[self.active_line])            
            boundaryIn[0,:] = i0_allen(mltp[self.active_line], self.spectrum.mu)
            boundaryIn *= self.spectrum.boundary[:,self.wvl_range[0]:self.wvl_range[1]]
        else:            
            boundaryIn = np.asfortranarray(stokes)

        # Renormalize nbar so that its CLV is the same as that of Allen, but with a decreased I0
        # If I don't do that, fitting profiles in the umbra is not possible. The lines become in
        # emission because the value of the source function, a consequence of the pumping radiation,
        # is too large. In this case, one needs to use beta to reduce the value of the source function.
        ratio = boundaryIn[0,0] / i0_allen(mltp[self.active_line], self.spectrum.mu)
        nbarIn = np.ones(4) * ratio
        omegaIn = np.zeros(4) #when different than zero it is used as reduction factors in  hazel
        betaIn = self.parameters['beta']       

        dopplerWidthIn = self.parameters['deltav']
        dampingIn = self.parameters['a']
        #remind j10 is vector (one val per transition).Inversions do not consider this yet.  
        j10In = self.parameters['j10']
        dopplerVelocityIn = self.parameters['v']
        
        
        #self.index is the index integer of the chromosphere being processed.It changes
        # from 1 to n_chromospheres. Check where is it updated.
        
        args = (self.index, B1In, hIn, tau1In, boundaryIn, transIn, anglesIn, nLambdaIn,
            lambdaAxisIn, dopplerWidthIn, dampingIn, j10In, dopplerVelocityIn,
            betaIn, nbarIn, omegaIn)
        
        #opt coeffs here are still 2D, only for current slab self.index
        l,stokes,epsout,etaout,stimout,error = hazel_code._synth(*args)

        if (error == 1):
            raise NumericalErrorHazel()

        ff = self.parameters['ff']
        
        return ff * stokes, epsout,etaout,stimout,error #/ hsra_continuum(mltp[self.active_line])