import hazel
import matplotlib.pyplot as pl
import numpy as np
import sys

#fn='m1_coeffs_7p.hazexp'
fn='m1_coeffs_7p_test.hazexp'
#fn='m1_coeffs_57p.hazexp'
#fn='m1_coeffs_57p_static.hazexp'#--> many points should not boost opacity

if 1==1:
	m1 = hazel.ModelRT(atomfile='sodium_hfs.atom',apmosekc='1110')
	cdic={'ref frame': 'LOS'}#common to all cells
	to=m1.add_funcatmos(7,cdic)#,hzlims=[0.,1500.],hztype='parab')#functional atmosphere 
	m1.add_spectrum('s1', line='5895',wavelength=[5894., 5897., 150], 
		topology=to,los=[0.,0.,90.],boundary=[1,0,0,0])	;m1.setup() 
	#----------------------------------------------------------------------------------
	dlims={'B1':[350.,100.], 'B2': [89.,90.], 'B3':[44.,49.],\
		'tau':[6.,0.1],'v':[0.,4.],'deltav':[4.,7.],'a':[0.2,0.1] ,\
		'j10':[0.01,0.02],'j20f':[1.,1.5],'beta':[1.,1.]} #...'ff':[1,1],'nbar':[1,1]}

	pkws={'plotit':9,'nps':3,'var':'mono','method':1}
	hz=m1.set_funcatm(dlims,orders=4,**pkws) #set atm pars with given-order function
	#m1.synthesize(plot='s1',FtS=fn)#frac=True  muAllen=0.9;
	for mm in ['M2','M1']:m1.synthesize(plot='s1',method=mm,fractional=True)
else: 
	m1,des=hazel.readmodel(fn) #,'Emissivity'
	for mm in ['EvolOp','M1']:m1.synthesize(plot='s1',FtR=fn,method=mm)
	#m1.mutates('s1', apmosekc='0110')

m1.exit_hazel()

'''If you call synthesize repeatedly is ok, the new plots will be overplot and no 
replicant figures will pop up ocuppying memory.
But everytime you open or create a new model, even with the same name, the figure axes
shall be to None or recreated, leading to repeat a new figure every time.
Nothing that one can do to avoid this automatically, the user must just now
how these objects behave.'''

#fname='m1_coeffs_12octC.ehazel'
#hazel.savemodel([m2,dlims],fname)
#mm,dlims,des=hazel.readmodel(fname) 

#hazel.save_RTcoeffs(m2,'s1',dlims,fname) #model,spectrum name,dlims, filename
#eps,eta,rho,dlims,des=hazel.read_RTcoeffs(fname)

#m2.compare_experiments(m1,'s1')

#m1.plot_coeffs('s1',bwc=1.) #,coefs=['epsv','etai','etaq','etav'],scale=2)

#EXAMPLES MUTATES:
#mo,kk=m1.mutates('s1', apmosekc='1110',B1=[46.,48.],j20f=[1.1,1.5],pkws=pkws)
#dd={'B1':[1,34.],'j10'=[1,0.02],'method':'Emissivity'}
#m1.mutates('specname', atmpar1=[layernumber,value],j10=[layernumber,value],apmosekc='value',parsdic=dd)
#m1.mutates('s1', B1=[10,10.,1000],j10=[0,0.01,0.1])
#m1.mutates('s1', apmosekc='0110')
#m1.mutates('s1', apmosekc='0110',B1=[1,40.],j10=[0,0.02],bylayer=True)
#mo,kk=m1.mutates('s1', apmosekc='1111',B2=[1,0.],j10=[0,0.1],bylayer=True)
#mo,kk=m1.mutates('s1', apmosekc='1110',B3=[46.,48.],j20f=[1.1,1.5],pkws=pkws)
#mo2,kk=m1.mutates('s1', apmosekc='1110',v=[4.,8.],j10=[0.01,0.07],pkws=pkws)

#mo2.compare_experiments(mo,'s1')

#m1.fractional_polarization(s1)

#m1.reshow('all')#m1.reshow('1')

#ms=np.zeros(3, dtype=object)#gives a vector of pointers to python objects
#ms[0]=m1 try this to store all models of an experiment
