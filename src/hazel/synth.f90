module synth
use vars
use SEE
use rt_coef
implicit none
  
  integer :: npsf  ! JDLCR: vars for convolution with PSF
  real(kind=8), allocatable :: psf(:)
  !
  PRIVATE :: npsf, psf

contains

  ! ------------------------------------------------------------------------ -
  ! EDGAR: Main synthesis routine of Hazel Experimental with all RT methods
  ! -------------------------------------------------------------------------
 !subroutine synth_methods(nl,ds,epI,epQ,epU,epV,etI,etQ,etU,etV,roQ,roU,roV,stkOut)
 subroutine synth_methods(nl,ds,ep,et,ro,stkOut)
 integer, intent(in) :: nl!ds(kz), but only dn number of points!
 real(kind=8),intent(in) :: ds(:), ep(:,:,:),et(:,:,:),ro(:,:,:)
 !real(kind=8),dimension(:,:),intent(in) :: epI,epQ,epU,epV,etI,etQ,etU,etV,roQ,roU,roV !kw,kz
 real(kind=8),intent(inout) :: stkOut(:,:)!use stkOut as stkIn incom boundary condition and updating it
 
 real(kind=8) ::  I0, eta0, psim, psi0, dtau, source(4)!,StokesM(4),Stokes0(4)
 real(kind=8),dimension(4,4) ::  kappa_star, O_evol, psi_matrix,m2!, m1
 integer :: ii,qq,uu,vv,i,w,kz
 !real(kind=8),dimension(nl) :: epsi,epsq,epsu,epsv,etai,etaq,etau,etav
 !real(kind=8),dimension(nl) :: rhoi,rhoq,rhou,rhov


    !------------ABSURD OPERATION THAT IS REPEATED FOR EVERY CELL OF THE RAY----------
    !define it at intizialization and store globally in vars.f90    
!    identt4 = 0.d0
!    do i = 1, 4
!        identt4(i,i) = 1.d0 
!    enddo

    !----------------------------------------------------------------------------------
    if (synthesis_method == 0) then 
    !****************       
    ! ONLY EMISSIVITY: Accumulation of emission without absorption. Normalize later in Python
    !****************
        do ii = 1, 4 !we are using stkOut as stkIn incomming boundary condition and updating it
            stkOut(:,ii) = stkOut(:,ii) + ep(:,1,ii)!the 1 one here is current height 
        enddo
        !stkOut(:,1) = stkOut(:,1) + epI(:,1)!the 1 in epI here is current height 
        !stkOut(:,2) = stkOut(:,2) + epQ(:,1)!the 1 in epI here is current height 
        !stkOut(:,3) = stkOut(:,3) + epU(:,1)!the 1 in epI here is current height 
        !stkOut(:,4) = stkOut(:,4) + epV(:,1)!the 1 in epI here is current height 
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 1) then 
        synthesis_method = 5
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 2) then 
        synthesis_method = 5
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 3) then 
        synthesis_method = 5
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 4) then 
    
        synthesis_method = 5
    
    endif
    !----------------------------------------------------------------------------------

    if (synthesis_method == 6) then 
        synthesis_method = 5
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 7) then 
           synthesis_method = 5
    endif

    !----------------------------------------------------------------------------------
    if (synthesis_method == 5) then 
    !****************       
    ! Point-by-Point ordinary (constant K) evolution operator
    !****************
        kz=1 !current height
                   
        do w = 1, nl !point by point(frequency) calling... and Not efficient:dimensions should be exchanged
            !call fill_absorption_matrix(kappa_star,etI(w,kz),etQ(w,kz),etU(w,kz),etV(w,kz),roQ(w,kz),roU(w,kz),roV(w,kz))
            call fill_absorption_matrix(kappa_star,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappa_star = kappa_star/et(w,kz,1) !kappa_star / fin%etaI(w)  
            source(:) = ep(w,kz,:)/et(w,kz,1)!beta(kz) was already multiplied in python when !=1
!            kappa_star = kappa_star/etI(w,kz) !kappa_star / fin%etaI(w)  
!            source(1) = epI(w,kz)/etI(w,kz)!beta(kz) was already multiplied in python when !=1
!            source(2) = epQ(w,kz)/etI(w,kz)!beta(kz) was already multiplied in python when !=1
!            source(3) = epU(w,kz)/etI(w,kz)!beta(kz) was already multiplied in python when !=1
!            source(4) = epV(w,kz)/etI(w,kz)!beta(kz) was already multiplied in python when !=1
            
            !dtau = etI(w,kz) * ds(kz) !con kz only the present point i.e. 1   
            dtau = et(w,kz,1) * ds(kz) !con kz only the present point i.e. 1   
        
            ! Evaluate evolution operator
            call evol_operator(kappa_star,dtau,O_evol)
        
            call invert(kappa_star) !kappa_star is now inverted in place instead of doing m1=invert(kappa_star)
            Psi_matrix = matmul(kappa_star, identt4 - O_evol) !m2 = identity - O_evol

            ! Simplified version taking into account that the source function is constant, so that
            !  I0 = exp(-K^* * tau_MO) * I_sun + (PsiM+Psi0)*S  with
            ! PsiM = U0-U1/tau_MO    and PsiO = U1/m
            ! U0 = (K*)^(-1) (1-exp(-K^* tau_MO)    and      U1 = (K*)^(-1) (m*1 - U0)
            !Not efficient:dimensions should be exchanged
            !StokesM(1:4) = stkIn(w,1:4) !StokesM(1:4) = fin%stokes_boundary(0:3,w)
            !Stokes0 = matmul(O_evol,stkIn(w,1:4)) + matmul(Psi_matrix,source * beta(kz))
            !stkOut(w,:) = Stokes0(:)            !Not efficient:dimensions should be exchanged
            stkOut(w,:)= matmul(O_evol,stkOut(w,:)) + matmul(Psi_matrix,source)    
            !EDGAR:Are we here creating unnecesary copy?
        enddo
  
    endif

end subroutine synth_methods


!------------------------------------------------------------
!Simplification of the evolution operator method to ocuppy the least
!------------------------------------------------------------
subroutine synth_evolop(nl,ds,eps,eta,rho,stkOut)
 integer, intent(in) :: nl
 real(kind=8),intent(in) :: eps(:,:,:),eta(:,:,:),rho(:,:,:), ds(:) !ds(kz), but only dn number of points!
 real(kind=8),intent(inout) :: stkOut(:,:)!use stkOut as stkIn incom boundary condition and updating it
 real(kind=8) ::  I0, eta0, psim, psi0, dtau, source(4)!,StokesM(4),Stokes0(4)
 real(kind=8),dimension(4,4) :: identt4, kappa_star, O_evol, psi_matrix,m2!, m1
 integer :: ii,qq,uu,vv,i,w,kz
    
    identt4 = 0.d0
    do i = 1, 4
        identt4(i,i) = 1.d0
    enddo
 
    !****************       
    ! Point-by-Point ordinary (constant K) evolution operator
    !****************
        ii=1 ; qq=2 ;uu=3 ;vv=4 ;kz=1 !Stokes I, Q, U, V, current height

        do w = 1, nl !point by point(frequency) calling... and Not efficient:dimensions should be exchanged
            call fill_absorption_matrix(kappa_star,eta(w,ii,kz),eta(w,qq,kz),eta(w,uu,kz),eta(w,vv,kz),rho(w,qq,kz),rho(w,uu,kz),rho(w,vv,kz))
            kappa_star = kappa_star/eta(w,ii,kz) !kappa_star / fin%etaI(w)  
            source(:) = eps(w,:,kz)/eta(w,ii,kz)!beta(kz) was already multiplied in python when !=1
            dtau = eta(w,ii,kz) * ds(kz) !con kz only the present point i.e. 1   
            call evol_operator(kappa_star,dtau,O_evol)! Evaluate evolution operator
            call invert(kappa_star) !kappa_star is now inverted in place instead of doing m1=invert(kappa_star)
            Psi_matrix = matmul(kappa_star, identt4 - O_evol) !m2 = identity - O_evol
            !stkOut(w,:) = Stokes0(:)            !Not efficient:dimensions should be exchanged
            stkOut(w,:)= matmul(O_evol,stkOut(w,:)) + matmul(Psi_matrix,source)    
        enddo
        
    end subroutine synth_evolop


!------------------------------------------------------------
!------------------------------------------------------------
!------------------------------------------------------------
    subroutine dn_synthesis(pin,in_fixed,output, error)
    type(variable_parameters) :: pin
    type(fixed_parameters) :: in_fixed
    integer :: i, error
    real(kind=8) :: output(0:3,in_fixed%no), I0, Q0, U0, V0, ds, Imax, mu, Ic, factor, eta0, psim, psi0, sh, ds2
    real(kind=8) :: wstep
    real(kind=8) :: StokesM(4), kappa_star(4,4), identity(4,4), source(4), m1(4,4), m2(4,4), Stokes0(4)
    real(kind=8) :: O_evol(4,4), psi_matrix(4,4), Stokes1(4)

    ! More convenient names
    ! subroutine ray_synthesis(nl,pin,fin,output, error)
    ! type(variable_parameters) :: pin
    ! type(fixed_parameters) :: fin
    ! integer, intent(in) :: nl
    ! integer ,intent(out):: error
    ! real(kind=8),intent(out) :: output(0:3,nl)
    ! real(kind=8) :: I0 ds, Imax, eta0, psim, psi0
    ! real(kind=8) :: StokesM(4), kappa_star(4,4), identity(4,4), source(4), m1(4,4), m2(4,4), Stokes0(4)
    ! real(kind=8) :: O_evol(4,4), psi_matrix(4,4)
    ! integer :: i
 
        call init_psf()
        error = 0
        

!----------------------------------------------------------------------------------
!****************       
! Ordinary (order-0) evolution operator with constant K
!****************
        if (.not.associated(in_fixed%epsI)) allocate(in_fixed%epsI(in_fixed%no))
        if (.not.associated(in_fixed%epsQ)) allocate(in_fixed%epsQ(in_fixed%no))
        if (.not.associated(in_fixed%epsU)) allocate(in_fixed%epsU(in_fixed%no))
        if (.not.associated(in_fixed%epsV)) allocate(in_fixed%epsV(in_fixed%no))
        if (.not.associated(in_fixed%etaI)) allocate(in_fixed%etaI(in_fixed%no))
        if (.not.associated(in_fixed%etaQ)) allocate(in_fixed%etaQ(in_fixed%no))
        if (.not.associated(in_fixed%etaU)) allocate(in_fixed%etaU(in_fixed%no))
        if (.not.associated(in_fixed%etaV)) allocate(in_fixed%etaV(in_fixed%no))
        if (.not.associated(in_fixed%rhoQ)) allocate(in_fixed%rhoQ(in_fixed%no))
        if (.not.associated(in_fixed%rhoU)) allocate(in_fixed%rhoU(in_fixed%no))
        if (.not.associated(in_fixed%rhoV)) allocate(in_fixed%rhoV(in_fixed%no))  
                 
        if (.not.associated(in_fixed%dtau)) allocate(in_fixed%dtau(in_fixed%no))
        
         !ABSURD OPERATION THAT IS REPEATED FOR EVERY CELL OF THE RAY----------
        identity = 0.d0
        do i = 1, 4
            identity(i,i) = 1.d0
        enddo
        
                    
! Emission              
        in_fixed%epsI = in_fixed%epsilon(0,:)
        in_fixed%epsQ = in_fixed%epsilon(1,:)
        in_fixed%epsU = in_fixed%epsilon(2,:)
        in_fixed%epsV = in_fixed%epsilon(3,:)
        
! Absorption including stimulated emission
        in_fixed%etaI = in_fixed%eta(0,:)
        in_fixed%etaQ = in_fixed%eta(1,:)
        in_fixed%etaU = in_fixed%eta(2,:)
        in_fixed%etaV = in_fixed%eta(3,:)

! Magneto-optical effects

        in_fixed%rhoQ = in_fixed%mag_opt(1,:)
        in_fixed%rhoU = in_fixed%mag_opt(2,:)
        in_fixed%rhoV = in_fixed%mag_opt(3,:)


        ds = pin%dtau / maxval(in_fixed%etaI)

        in_fixed%dtau = in_fixed%etaI * ds
        
        !print*,in_fixed%etaI(1),in_fixed%stokes_boundary(0,1),pin%dtau

        do i = 1, in_fixed%no

            StokesM(1:4) = in_fixed%stokes_boundary(0:3,i)
            
            call fill_absorption_matrix(kappa_star,in_fixed%etaI(i),in_fixed%etaQ(i),in_fixed%etaU(i),in_fixed%etaV(i),in_fixed%rhoQ(i),in_fixed%rhoU(i),in_fixed%rhoV(i))
            kappa_star = kappa_star / in_fixed%etaI(i)
            source(1) = in_fixed%epsI(i) / in_fixed%etaI(i)
            source(2) = in_fixed%epsQ(i) / in_fixed%etaI(i)
            source(3) = in_fixed%epsU(i) / in_fixed%etaI(i)
            source(4) = in_fixed%epsV(i) / in_fixed%etaI(i)

! Evaluate the evolution operator
            call evol_operator(kappa_star,in_fixed%dtau(i),O_evol)

! Calculate K*^(-1)
            m1 = kappa_star
            call invert(m1)

            m2 = identity - O_evol
            Psi_matrix = matmul(m1,m2)

! Simplified version taking into account that the source function is constant, so that
!  I0 = exp(-K^* * tau_MO) * I_sun + (PsiM+Psi0)*S  with
! PsiM = U0-U1/tau_MO    and PsiO = U1/m
! U0 = (K*)^(-1) (1-exp(-K^* tau_MO)    and      U1 = (K*)^(-1) (m*1 - U0)
            Stokes0 = matmul(O_evol,StokesM) + matmul(Psi_matrix,source * pin%beta)
            
            output(0,i) = Stokes0(1)
            output(1,i) = Stokes0(2)
            output(2,i) = Stokes0(3)
            output(3,i) = Stokes0(4)
            
        enddo
!----------------------------------------------------------------------------------


        !in_fixed%total_forward_modeling = in_fixed%total_forward_modeling + 1

    
        !EDGAR: convolution should not be done until highest layer, so remove it from here!
        !call convolve(in_fixed%no, output)

    
    end subroutine dn_synthesis

!------------------------------------------------------------
! ORIGINAL ROUTINE NOW DEPRECATED
!------------------------------------------------------------
    subroutine do_synthesis(in_params,in_fixed,in_observation,output, error)
    type(variable_parameters) :: in_params, in_trial
    type(type_observation) :: in_observation
    type(fixed_parameters) :: in_fixed
    integer :: i, error
    real(kind=8) :: output(0:3,in_fixed%no), I0, Q0, U0, V0, ds, Imax, mu, Ic, factor, eta0, psim, psi0, sh, ds2
    real(kind=8) :: wstep
    real(kind=8) :: StokesM(4), kappa_star(4,4), identity(4,4), source(4), m1(4,4), m2(4,4), Stokes0(4)
    real(kind=8) :: O_evol(4,4), psi_matrix(4,4), Stokes1(4)

    !
    ! JDLCR: init PSF?
    ! It will only be done the first time internally in the function
    !
        call init_psf()

        error = 0
        
! ! Fill the statistical equilibrium equations
!         call fill_SEE(in_params, in_fixed, 1, error)        

! ! If the solution of the SEE gives an error, return
!         if (error == 1) return
                
! ! Calculate the absorption/emission coefficients for a given transition
!         call calc_rt_coef(in_params, in_fixed, in_observation, 1)
                        
!----------------------------------------------------------------------------------
if (synthesis_method == 0) then 
!****************       
! ONLY EMISSIVITY
!****************

        if (in_fixed%use_atomic_pol == 1) then
            Imax = maxval(in_fixed%epsilon(0,:))
            do i = 0, 3
                output(i,:) = in_fixed%epsilon(i,:) / Imax
            enddo
        else  
            Imax = maxval(in_fixed%epsilon_zeeman(0,:))  !EDGAR:aqui habia solo epsilon pero deberia ser epsilon_zeeman
            do i = 0, 3
                output(i,:) = in_fixed%epsilon_zeeman(i,:) / Imax
            enddo
        endif
    
endif

!----------------------------------------------------------------------------------
if (synthesis_method == 5) then 
!****************       
! Slab case with EXACT SOLUTION
!****************
        if (.not.associated(in_fixed%epsI)) allocate(in_fixed%epsI(in_fixed%no))
        if (.not.associated(in_fixed%epsQ)) allocate(in_fixed%epsQ(in_fixed%no))
        if (.not.associated(in_fixed%epsU)) allocate(in_fixed%epsU(in_fixed%no))
        if (.not.associated(in_fixed%epsV)) allocate(in_fixed%epsV(in_fixed%no))
        if (.not.associated(in_fixed%etaI)) allocate(in_fixed%etaI(in_fixed%no))
        if (.not.associated(in_fixed%etaQ)) allocate(in_fixed%etaQ(in_fixed%no))
        if (.not.associated(in_fixed%etaU)) allocate(in_fixed%etaU(in_fixed%no))
        if (.not.associated(in_fixed%etaV)) allocate(in_fixed%etaV(in_fixed%no))
        if (.not.associated(in_fixed%rhoQ)) allocate(in_fixed%rhoQ(in_fixed%no))
        if (.not.associated(in_fixed%rhoU)) allocate(in_fixed%rhoU(in_fixed%no))
        if (.not.associated(in_fixed%rhoV)) allocate(in_fixed%rhoV(in_fixed%no))           
        if (.not.associated(in_fixed%dtau)) allocate(in_fixed%dtau(in_fixed%no))
        
         !------------NO NEED OF REPEATING THIS FOR EVERY POINT OF THE RAY----------
        identity = 0.d0 
        do i = 1, 4
            identity(i,i) = 1.d0
        enddo
        
    !EDGAR: we add the possibility of using only the zeeman coeffs without atompol
    if (in_fixed%use_atomic_pol == 1 ) then
                    
! Emission              
        in_fixed%epsI = in_fixed%epsilon(0,:)
        in_fixed%epsQ = in_fixed%epsilon(1,:)
        in_fixed%epsU = in_fixed%epsilon(2,:)
        in_fixed%epsV = in_fixed%epsilon(3,:)
        
! Absorption including stimulated emission
        in_fixed%etaI = in_fixed%eta(0,:) - use_stim_emission_RT * in_fixed%eta_stim(0,:)
        in_fixed%etaQ = in_fixed%eta(1,:) - use_stim_emission_RT * in_fixed%eta_stim(1,:)
        in_fixed%etaU = in_fixed%eta(2,:) - use_stim_emission_RT * in_fixed%eta_stim(2,:)
        in_fixed%etaV = in_fixed%eta(3,:) - use_stim_emission_RT * in_fixed%eta_stim(3,:)

! Magneto-optical effects
        if (use_mag_opt_RT == 1) then
            in_fixed%rhoQ = in_fixed%mag_opt(1,:) - use_stim_emission_RT * in_fixed%mag_opt_stim(1,:)
            in_fixed%rhoU = in_fixed%mag_opt(2,:) - use_stim_emission_RT * in_fixed%mag_opt_stim(2,:)
            in_fixed%rhoV = in_fixed%mag_opt(3,:) - use_stim_emission_RT * in_fixed%mag_opt_stim(3,:)
        else
            in_fixed%rhoQ = 0.d0
            in_fixed%rhoU = 0.d0
            in_fixed%rhoV = 0.d0
        endif
    
    else

! Emission
            in_fixed%epsI = in_fixed%epsilon_zeeman(0,:)
            in_fixed%epsQ = in_fixed%epsilon_zeeman(1,:)
            in_fixed%epsU = in_fixed%epsilon_zeeman(2,:)
            in_fixed%epsV = in_fixed%epsilon_zeeman(3,:)

! Absorption including stimulated emission
            in_fixed%etaI = in_fixed%eta_zeeman(0,:) - use_stim_emission_RT * in_fixed%eta_stim_zeeman(0,:) + 1.d-20
            in_fixed%etaQ = in_fixed%eta_zeeman(1,:) - use_stim_emission_RT * in_fixed%eta_stim_zeeman(1,:) + 1.d-20
            in_fixed%etaU = in_fixed%eta_zeeman(2,:) - use_stim_emission_RT * in_fixed%eta_stim_zeeman(2,:) + 1.d-20
            in_fixed%etaV = in_fixed%eta_zeeman(3,:) - use_stim_emission_RT * in_fixed%eta_stim_zeeman(3,:) + 1.d-20

! Magneto-optical terms
            if (use_mag_opt_RT == 1) then
                in_fixed%rhoQ = in_fixed%mag_opt_zeeman(1,:) - use_stim_emission_RT * in_fixed%mag_opt_stim_zeeman(1,:)
                in_fixed%rhoU = in_fixed%mag_opt_zeeman(2,:) - use_stim_emission_RT * in_fixed%mag_opt_stim_zeeman(2,:)
                in_fixed%rhoV = in_fixed%mag_opt_zeeman(3,:) - use_stim_emission_RT * in_fixed%mag_opt_stim_zeeman(3,:)
            else
                in_fixed%rhoQ = 0.d0
                in_fixed%rhoU = 0.d0
                in_fixed%rhoV = 0.d0
            endif
    endif

        ds = in_params%dtau / maxval(in_fixed%etaI)
        in_fixed%dtau = in_fixed%etaI * ds
                
        do i = 1, in_fixed%no

            StokesM(1:4) = in_fixed%stokes_boundary(0:3,i)
            
            call fill_absorption_matrix(kappa_star,in_fixed%etaI(i),in_fixed%etaQ(i),in_fixed%etaU(i),in_fixed%etaV(i),in_fixed%rhoQ(i),in_fixed%rhoU(i),in_fixed%rhoV(i))
            kappa_star = kappa_star / in_fixed%etaI(i)
            source(1) = in_fixed%epsI(i) / in_fixed%etaI(i)
            source(2) = in_fixed%epsQ(i) / in_fixed%etaI(i)
            source(3) = in_fixed%epsU(i) / in_fixed%etaI(i)
            source(4) = in_fixed%epsV(i) / in_fixed%etaI(i)

! Evaluate the evolution operator
            call evol_operator(kappa_star,in_fixed%dtau(i),O_evol)

! Calculate K*^(-1)
            m1 = kappa_star
            call invert(m1)

            m2 = identity - O_evol
            Psi_matrix = matmul(m1,m2)

! Simplified version taking into account that the source function is constant, so that
!  I0 = exp(-K^* * tau_MO) * I_sun + (PsiM+Psi0)*S  with
! PsiM = U0-U1/tau_MO    and PsiO = U1/m
! U0 = (K*)^(-1) (1-exp(-K^* tau_MO)    and      U1 = (K*)^(-1) (m*1 - U0)
            Stokes0 = matmul(O_evol,StokesM) + matmul(Psi_matrix,source * in_params%beta)
            
            output(0,i) = Stokes0(1)
            output(1,i) = Stokes0(2)
            output(2,i) = Stokes0(3)
            output(3,i) = Stokes0(4)
            
        enddo
 
endif
!----------------------------------------------------------------------------------


        in_fixed%total_forward_modeling = in_fixed%total_forward_modeling + 1

        
        ! EDGAR:make no sense to convolve before end of RT, avoid this in new versions
        !spectral convolution is in any case considered in Pyhton
        call convolve(in_fixed%no, output)

    
    end subroutine do_synthesis


  subroutine init_psf()
    implicit none
    character(len=7), parameter :: filename = 'psf.txt'
    integer :: unit, ii
    logical :: psf_exists
    ! -------------------------------------------------------------------------
    ! JDLCR: This function will only read the PSF (hardwired to psf.txt) in the first call.
    ! -------------------------------------------------------------------------
    if(allocated(psf)) return

    psf_exists = .FALSE.
    INQUIRE( FILE=filename, EXIST=psf_exists) 
    if(.not. psf_exists) return

    print *, 'Using PSF'  
    unit = 1
    OPEN(unit, FILE=filename, status='OLD')  ! Open PSF file and read
    read(unit,*) npsf ! Number of elements of the PSF

    allocate(psf(npsf)) ! allocate array to store the PSF
    
    do ii=1,npsf
       read(unit,*) psf(ii)
    end do    
    CLOSE(unit)
  end subroutine init_psf

 ! -------------------------------------------------------------------------
  ! JDLCR: This function computes convolution without FFTs.
  ! It will not pad the arrays,use the part of the PSF inside the range of the spectra.
  ! -------------------------------------------------------------------------
 subroutine convolve(n, sp)
    implicit none
    integer :: n, ii, jj, w0, w1, ww, npsf2, ss
    real(8) :: sp(0:3, n), res(n), psfsum


    if(.not. allocated(psf)) return

    npsf2 = npsf/2

    do ss = 0,3
       do ww = 1, n
          w0 = max(ww - npsf2, 1)
          w1 = min(ww + npsf2, n)
          ii = w0 - (ww - npsf2)
          jj = (ww + npsf2) - w1       
          res(ww) = sum(psf(1+ii:npsf-jj)*sp(ss,w0:w1)) / sum(psf(1+ii:npsf-jj))
       end do
       sp(ss,:) = res(:)
    end do

  end subroutine convolve




end module synth
