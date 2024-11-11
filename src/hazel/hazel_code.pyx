# cython: language_level=3
from numpy cimport ndarray as ar
from numpy cimport npy_bool as nbool
from numpy import empty, linspace, zeros, array

dni=3

cdef extern:
	void c_rtcoeffs(int* index, double* B1Input, double* hInput, int* transInput, double* anglesInput, 
		int* nLambdaInput, double* lambdaAxisInput,double* dopplerWidthInput, double* dampingInput, 
		double* j10Input, double* dopplerVelocityInput, double* nbarInput, double* omegaInput, 
		int* atompolInput,int* magoptInput,int* stimemInput,int* nocohInput, double* dcolInput,
		double* wavelengthOut, nbool* recomputed,double* epsOut,double* etaOut,double* rhoOut, int* error)

	void c_direct_synthesis(int* nl,int* nz,int* nsteps, int* dn, int* method, double* ds, 
		double* eps, double* eta, double* rho,double* stkIn,
		double* stkOut, int* error)

	void c_rt_synthesis(int* index,int* dn, int* synMethIn, double* hIn, double* tauIn, double* betaIn,
		double* boundaryIn, int* nLambdaIn, double* epsIn, double* etaIn, double* rhoIn,
		double* stokesOut, int* error)

	void c_hazel(int* index, int* synMethInput, double* B1Input, double* hInput, double* tau1Input, 
		double* boundaryInput, int* transInput, double* anglesInput, int* nLambdaInput, double* lambdaAxisInput,
		double* dopplerWidthInput, double* dampingInput, double* j10Input, double* dopplerVelocityInput, 
		double* betaInput, double* nbarInput, double* omegaInput, 
		int* atompolInput,int* magoptInput,int* stimemInput,int* nocohInput, double* dcolInput,
		double* wavelengthOut, double* stokesOut,double* epsOut,double* etaOut,double* stimOut, int* error)

	void c_init(int* nchar,char* atomfile, int* verbose,int* ntransOutput) #EDGAR: added nchar, atomfile, and output par ntrans
	void c_exit(int* index)


#NEW routine called by python get_coeffs() to solve local part: SEE AND ALL OPTICAL COEFFS FOR ALL CELLS
def _rtcoeffs(int index=1, ar[double,ndim=1] B1Input=zeros(3), double hInput=3.0, 
	int transInput=1, ar[double,ndim=1] anglesInput=zeros(3), int nLambdaInput=128, 
	ar[double,ndim=1] lambdaAxisInput=linspace(-1.5,2.5,128), double dopplerWidthInput=5.0,
	double dampingInput=0.0, ar[double,ndim=1] j10Input=zeros(4),double dopplerVelocityInput=0.0, 
	ar[double,ndim=1] nbarInput=zeros(4), ar[double,ndim=1] omegaInput=zeros(4),
	int atompolInput=1,int magoptInput=1,int stimemInput=1,int nocohInput=0, 
	ar[double,ndim=1] dcolInput=zeros(3)):

	cdef:
		ar[double,ndim=1] wavelengthOut = empty(nLambdaInput, order='F')
		nbool recomputed = True
		ar[double,ndim=3,mode='fortran'] epsOut = empty((nLambdaInput,1,4), order='F') #this is NO mode='fortran'
		ar[double,ndim=3,mode='fortran'] etaOut = empty((nLambdaInput,1,4), order='F') # and init NOT order fortran 
		ar[double,ndim=3,mode='fortran'] rhoOut = empty((nLambdaInput,1,3), order='F')  # the latter is unimportant
		#ar[double,ndim=3] epsOut = empty((1,4,nLambdaInput), order='F')
		#ar[double,ndim=3] etaOut = empty((1,4,nLambdaInput), order='F')
		#ar[double,ndim=3] rhoOut = empty((1,3,nLambdaInput), order='F')
		int error

	#calls fortran routine c_hazel in hazel_py.f90
	c_rtcoeffs(&index, &B1Input[0], &hInput, &transInput, &anglesInput[0], &nLambdaInput, 
		&lambdaAxisInput[0], &dopplerWidthInput, &dampingInput, &j10Input[0], &dopplerVelocityInput, 
		&nbarInput[0], &omegaInput[0],&atompolInput,&magoptInput,&stimemInput,&nocohInput,&dcolInput[0],
		<double*> wavelengthOut.data, <nbool*> &recomputed, <double*> epsOut.data, 
		<double*> etaOut.data,<double*> rhoOut.data, &error)
    
	return wavelengthOut, recomputed, epsOut, etaOut, rhoOut, error
	"""
	Arrays with more than one dimension (as boundaryInput) are in fortran mode while others do not need
	Args:
		index: (int) index of atmosphere
		B1Input: (float) matrix of size 3 x dni (with dni expected to be 3)--> these are in python dims here
				and with the magnetic field vector in spherical coordinates
		hInput: (float) vector with height
		transInput: (int) transition to compute from the model atom
		anglesInput: (float) vector of size 3 describing the LOS
		lambdaAxisInput: (float) vector of size 2 defining the left and right limits of the wavelength axis
		nLambdaInput: (int) number of wavelength points
		dopplerWidth1Input: (float) Doppler width of the first component
		dampingInput: (float) damping
		dopplerVelocityInput: (float) bulk velocity affecting the first component
		nbarInput: (float) vector of size 4 to define nbar for every transition of the model atom (set them to zero to use Allen's)
		omegaInput: (float) vector of size 4 to define omega for every transition of the model atom (set them to zero to use Allen's)
		
    Returns:
        wavelengthOutput: (float) vector of size nLambdaInput with the wavelength axis
        epsOutput: (float) array of size (4,nLambdaInput) with the emissivity vector at each wavelength
        etaOutput: (float) array of size (7,nLambdaInput) with the independent elements of K matrix at each wavelength
		error: (int) zero if everything went OK
	"""

def _direct_synthesis(int nl=128, int nz=1,int nsteps=1,int dn=dni,
	int method=5, ar[double,ndim=1] ds=zeros(101),
	#ar[double,ndim=3,mode='fortran'] eps=zeros((128,4,100), order='F'),
	#ar[double,ndim=3,mode='fortran'] eta=zeros((128,4,100), order='F'),
	#ar[double,ndim=3,mode='fortran'] rho=zeros((128,3,100), order='F'),
	ar[double,ndim=3,mode='fortran'] eps=zeros((128,100,4), order='F'), 
	ar[double,ndim=3,mode='fortran'] eta=zeros((128,100,4), order='F'),
	ar[double,ndim=3,mode='fortran'] rho=zeros((128,100,3), order='F'),
	ar[double,ndim=2,mode='fortran'] stkIn=zeros((4,128), order='F') ):
	#ar[double,ndim=2,mode='fortran'] stkIn=zeros((128,4), order='F') ):

	cdef:		
		ar[double,ndim=2,mode='fortran'] stkOut = empty((4,nl), order='F') #this IS mode='fortran'
		int error

	c_direct_synthesis(&nl, &nz, &nsteps, &dn, &method, &ds[0], 
		&eps[0,0,0], &eta[0,0,0], &rho[0,0,0], &stkIn[0,0],
		<double*> stkOut.data, &error)
    
	return stkOut, error

def _rt_synthesis(int index=1,int dn=dni, int synMethIn=5, ar[double,ndim=1] hIn=zeros(dni), 
	ar[double,ndim=1] tauIn=zeros(dni), ar[double,ndim=1] betaIn=zeros(dni), 
	ar[double,ndim=2,mode='fortran'] boundaryIn=zeros((4,128)),int nLambdaIn=128,
	ar[double,ndim=3,mode='fortran'] epsIn=zeros((dni,4,128), order='F'),
	ar[double,ndim=3,mode='fortran'] etaIn=zeros((dni,4,128), order='F'),
	ar[double,ndim=3,mode='fortran'] rhoIn=zeros((dni,3,128), order='F') ):

	cdef:		
		ar[double,ndim=2,mode='fortran'] stokesOut = empty((4,nLambdaIn), order='F')
		int error

	c_rt_synthesis(&index, &dn, &synMethIn, &hIn[0], &tauIn[0], &betaIn[0], 
		&boundaryIn[0,0], &nLambdaIn, &epsIn[0,0,0],&etaIn[0,0,0],&rhoIn[0,0,0],
		<double*> stokesOut.data, &error)
    
	return stokesOut, error
	"""
	Arrays with more than one dimension (as boundaryInput) are in fortran mode while others do not need
	Args:
		dn: number of cells, i.e. size of interval to be processed by formal solver
		hInput: (float) vector with height
		tau1Input: (float) vector with optical depth of the first component
		boundaryInput: (float) vector of size 4xnLambda with the boundary condition for (I,Q,U,V)
		nLambdaInput: (int) number of wavelength points
		epsIn,etaIn,rhoIn: are the full opt coeffs in the block of dn cells selected
		betaInput: (float) enhancement factor for the source function of component 1 to allow for emission lines in the disk
    Returns:
        stokesOutput: (float) array of size (4,nLambdaInput) with the emergent Stokes profiles
		error: (int) zero if everything went OK
	"""

#standard routine called by python synthazel to synthesis ONLY 1 CELL PER CALL
def _synth(int index=1, int synMethInput=5, ar[double,ndim=1] B1Input=zeros(3), double hInput=3.0, 
	double tau1Input=1.0, 
	ar[double,ndim=2,mode='fortran'] boundaryInput=zeros((4,128)), int transInput=1, ar[double,ndim=1] anglesInput=zeros(3), 
	int nLambdaInput=128, ar[double,ndim=1] lambdaAxisInput=linspace(-1.5,2.5,128),  
	double dopplerWidthInput=5.0, double dampingInput=0.0, ar[double,ndim=1] j10Input=zeros(4),double dopplerVelocityInput=0.0, 
	double betaInput=1.0, ar[double,ndim=1] nbarInput=zeros(4), 
	ar[double,ndim=1] omegaInput=zeros(4),
	int atompolInput=1,int magoptInput=1,int stimemInput=1,int nocohInput=0, 
	ar[double,ndim=1] dcolInput=zeros(3)):
	
	"""
	Carry out a synthesis with Hazel
	Args: (see the manual for the meaning of all of them)
		index: (int) index of atmosphere
		B1Input: (float) vector of size 3 with the magnetic field vector in spherical coordinates for the first component
		hInput: (float) height
		tau1Input: (float) optical depth of the first component
		boundaryInput: (float) vector of size 4xnLambda with the boundary condition for (I,Q,U,V)
		transInput: (int) transition to compute from the model atom
		anglesInput: (float) vector of size 3 describing the LOS
		lambdaAxisInput: (float) vector of size 2 defining the left and right limits of the wavelength axis
		nLambdaInput: (int) number of wavelength points
		dopplerWidth1Input: (float) Doppler width of the first component
		dampingInput: (float) damping
		dopplerVelocityInput: (float) bulk velocity affecting the first component
		betaInput: (float) enhancement factor for the source function of component 1 to allow for emission lines in the disk
		nbarInput: (float) vector of size 4 to define nbar for every transition of the model atom (set them to zero to use Allen's)
		omegaInput: (float) vector of size 4 to define omega for every transition of the model atom (set them to zero to use Allen's)
		
    Returns:
        wavelengthOutput: (float) vector of size nLambdaInput with the wavelength axis
        stokesOutput: (float) array of size (4,nLambdaInput) with the emergent Stokes profiles
        epsOutput: (float) array of size (4,nLambdaInput) with the emissivity vector at each wavelength
        etaOutput: (float) array of size (7,nLambdaInput) with the independent elements of K matrix at each wavelength
		error: (int) zero if everything went OK
	"""
	
	cdef:
		ar[double,ndim=1] wavelengthOut = empty(nLambdaInput, order='F')
		ar[double,ndim=2] stokesOut = empty((4,nLambdaInput), order='F')
		ar[double,ndim=2] epsOut = empty((4,nLambdaInput), order='F')
		ar[double,ndim=2] etaOut = empty((7,nLambdaInput), order='F')
		ar[double,ndim=2] stimOut = empty((7,nLambdaInput), order='F')
		int error

	#calls fortran routine c_hazel in hazel_py.f90
	c_hazel(&index, &synMethInput, &B1Input[0], &hInput, &tau1Input,  
		&boundaryInput[0,0], &transInput, &anglesInput[0], &nLambdaInput, &lambdaAxisInput[0],  
		&dopplerWidthInput, &dampingInput, &j10Input[0], &dopplerVelocityInput, 
		&betaInput, &nbarInput[0], &omegaInput[0], 
		&atompolInput,&magoptInput,&stimemInput,&nocohInput,&dcolInput[0],
		<double*> wavelengthOut.data, <double*> stokesOut.data,<double*> epsOut.data, <double*> etaOut.data,
		<double*> stimOut.data, &error)
    
	return wavelengthOut, stokesOut, epsOut, etaOut, stimOut, error
	
def _init(str atomfile, int verbose=0):
	"""
	Initialize and do some precomputations that can be avoided in the subsequent calls to the synthesis
	Args:
        atomfile: (str) name of the input atom file .mod to be read. This is a C string
    Returns:
        None
	EDGAR:The line atomfile.encode() converts the input atomfile string to utf8

	"""
	ftmp = atomfile.encode()
	cdef:
		int nchar = len(atomfile)
		char* atomfileInput = ftmp
		#int ntransOutput = 0
		int ntransOutput = 0

	
	#c_init(&nchar, &atomfileInput[0], &verbose) #&ntransOutput
	#this calls init routine in hazel_py.f90 and return ntrans 
	c_init(&nchar, &atomfileInput[0], &verbose, &ntransOutput)

	return ntransOutput #EDGAR:now it returns ntrans when called from model.py

def _exit(int index):
		
	c_exit(&index)


