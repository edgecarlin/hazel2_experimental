module synth
use vars
use SEE
use rt_coef
!USE OMP_LIB
implicit none
  
contains

! ------------------------------------------------------------------------ -
! Magnus inhomogeneous method 1:  I = Oevol*I_0 + /Phi_1 *E_bar
! -------------------------------------------------------------------------
!subroutine Magnus_FormSol_1(stoks,nl,ds,epI,epQ,epU,epV,etI,etQ,etU,etV,roQ,roU,roV)
 subroutine Magnus_FormSol_1(nl,np,ds,epsZ,etaZ,roZ,stoks)
    integer, intent(in) :: nl,np
    real(kind=8),intent(in) :: ds(:),etaZ(:,:,:),roZ(:,:,:),epsZ(:,:,:)!, eps(:,:,:),eta(:,:,:),rho(:,:,:)
    !real(kind=8),dimension(:,:),intent(in) :: epI,epQ,epU,epV!,etI,etQ,etU,etV,roQ,roU,roV !kw,kz
    real(kind=8),intent(inout) :: stoks(:,:)

    real(kind=8) ::pkpipp(np) !pk,pi
    real(kind=8),dimension(nl,4,4) :: omhat,omtil, evolop  !,intent(out)
    real(kind=8) :: alfa(nl,3),beta(nl,3),emis(nl,4)!coef(0:10,nl)
    real(kind=8),dimension(4,4,nl) :: fomhat,fomhat2,fomtil,fevolop,phi1
    real(kind=8) :: tau(nl),qq(nl),rr(nl)
    real(kind=8),dimension(nl) :: f1,fa,fb,f2, g1,ga,gb,g2
    !real(kind=8),dimension(nl) :: hh,hh_2,bhat_2,bhat,btil_2,btil,dhat,dtil
    !real(kind=8),dimension(nl) :: f1,fa,fb,f2, g1,ga,gb,g2,Chat, Ctil, Shat, Stil,exptau,tau_2,comfac
    integer::kz,kw,ii,jj,kk

    !real(kind=8),dimension(nl) :: Feps, OFeps, Fptil,OFptil,Fphat,OFphat,aux1,aux2,qqsign


    !INIT QUADRATURE WEIGHTS: 
    !Valid for Magnus Piecewise, Linear and Order 4 Schemes    
    ! k(O)*---ds(k)---*i(M)*---ds(k+1)---*k+1(P)
    pkpipp= quadrature_weights(np,ds)!ds is already a small chunk

    !PROGRAM ADAPTATIVE RECURSIVE QUADRATURE AND COMPOSED QUADRATURE RULE
    !add also routine to select the cuadrature coeffs for 1,2, ans 3 points (still with parab interpolants)

    !Integration of optical coefficients alog the ray
            tau(:)=MATMUL(etaZ(:,:,1),pkpipp)
            do kk=1,3  !go from (nw,nz,nquv) to (nw,nquv)
            alfa(:,kk) = MATMUL(etaZ(:,:,kk+1),pkpipp)
            beta(:,kk) = MATMUL(roZ(:,:,kk),pkpipp)
            emis(:,kk) = MATMUL(epsZ(:,:,kk),pkpipp)
            enddo
            emis(:,4) = MATMUL(epsZ(:,:,4),pkpipp)

    !.....................................................................
    !FULL EXACT MAGNUS EVOLUTION OPERATOR UNTIL ORDER 1

            !Build Omega-hat (Lorentz hat) and Omega-tilde (Lorentz tilde) for all frequencies
            do ii=1,3 !Efficient calculation without matrix inversions
                call fill_Lorentz_freqs(Omhat(:,:,:),alfa(:,ii),beta(:,ii),ii) !return 4x4 Omega hat
                call fill_Lorentz_freqs(Omtil(:,:,:),beta(:,ii),-alfa(:,ii),ii) !return 4x4 Ometa tilde
            enddo    !Results are (kw, 4column,4row)

            !Most efficient way I have found to exchange dimensions for speeding up last step
            !$OMP PARALLEL DO !--> try later
            do kk =1,nl
                do jj=1,4
                    do ii=1,4
                        fomhat(ii,jj,kk)=Omhat(kk,ii,jj)
                        fomtil(ii,jj,kk)=Omtil(kk,ii,jj)
                    enddo
                enddo
                fomhat2(1:4,1:4,kk)=MATMUL(fomhat(1:4,1:4,kk),fomhat(1:4,1:4,kk))
            enddo  !Results are (4column,4row,kw)
            !$OMP END PARALLEL DO
            !invomhat= reshape(Omhat, shape(invomhat), order = [2,3,1]) 

            !.................
            qq = 2.0*dot_productF2(alfa,beta) !--> THIS HAS THE KEY SIGN OF f1b
            rr = dot_productF2(alfa,alfa) - dot_productF2(beta,beta)                        
            !call dot_product_signF2(2.0*alfa,beta,qq,qqsign) !--> THIS HAS THE KEY SIGN OF f1b

            !substituting this routine by its operations directly is faster
            !call get_Lorentz_funs(nl,qq,rr,tau,f1,fa,fb,f2,g1,ga,gb,g2)
            call get_Lorentz_funs(nl,qq,rr,tau,f1,fa,fb,f2,g1,ga,gb,g2)

            do kk =1,nl
                fevolop(1:4,1:4,kk) = f1(kk)*identt4(1:4,1:4) + &
                    fa(kk)*fomhat(1:4,1:4,kk) + fb(kk) * fomtil(1:4,1:4,kk) + &
                    f2(kk)*fomhat2(1:4,1:4,kk)  

                !phi1(1:4,1:4,kk) = -(2.d0/qq(kk))*matmul(identt4(1:4,1:4)-fevolop(1:4,1:4,kk), fomtil(1:4,1:4,kk) )

                phi1(1:4,1:4,kk) = g1(kk)*identt4(1:4,1:4) +&
                ga(kk)*fomhat(1:4,1:4,kk) + & 
                gb(kk)*fomtil(1:4,1:4,kk) + &
                g2(kk)*fomhat2(1:4,1:4,kk)  

                !try transposing stokes before multiplying and retransposing again or redefine stokes
                stoks(1:4,kk) = matmul(fevolop(1:4,1:4,kk),stoks(1:4,kk))+matmul(phi1(1:4,1:4,kk),emis(kk,1:4)) 
            enddo
        
 end subroutine Magnus_FormSol_1
! ------------------------------------------------------------------------ -
! Magnus inhomogeneous method 2:  I = I_0 + /Phi_1 * (A_bar * I_0 + E_bar)
! -------------------------------------------------------------------------
 subroutine Magnus_FormSol_1b(nl,np,ds,epsZ,etaZ,roZ,stoks)
    integer, intent(in) :: nl,np
    real(kind=8),intent(in) :: ds(:),etaZ(:,:,:),roZ(:,:,:),epsZ(:,:,:)!, eps(:,:,:),eta(:,:,:),rho(:,:,:)
    !real(kind=8),dimension(:,:),intent(in) :: epI,epQ,epU,epV!,etI,etQ,etU,etV,roQ,roU,roV !kw,kz
    real(kind=8),intent(inout) :: stoks(:,:)

    real(kind=8) ::pkpipp(np) !pk,pi
    real(kind=8),dimension(nl,4,4) :: omhat,omtil, evolop  !,intent(out)
    real(kind=8) :: alfa(nl,3),beta(nl,3),emis(nl,4)!coef(0:10,nl)
    real(kind=8),dimension(4,4,nl) :: fomhat,fomhat2,fomtil,fevolop,phi1
    real(kind=8) :: tau(nl),qq(nl),rr(nl),kappa_bar(4,4)
    real(kind=8),dimension(nl) :: g1,ga,gb,g2
    !real(kind=8),dimension(nl) :: hh,hh_2,bhat_2,bhat,btil_2,btil,dhat,dtil
    !real(kind=8),dimension(nl) :: f1,fa,fb,f2, g1,ga,gb,g2,Chat, Ctil, Shat, Stil,exptau,tau_2,comfac
    integer::kz,kw,ii,jj,kk

    !real(kind=8),dimension(nl) :: Feps, OFeps, Fptil,OFptil,Fphat,OFphat,aux1,aux2,qqsign


    !INIT QUADRATURE WEIGHTS: 
    !Valid for Magnus Piecewise, Linear and Order 4 Schemes    
    ! k(O)*---ds(k)---*i(M)*---ds(k+1)---*k+1(P)
    pkpipp= quadrature_weights(np,ds)!ds is already a small chunk

    !PROGRAM ADAPTATIVE RECURSIVE QUADRATURE AND COMPOSED QUADRATURE RULE
    !add also routine to select the cuadrature coeffs for 1,2, ans 3 points (still with parab interpolants)

    !Integration of optical coefficients alog the ray
            tau(:)=MATMUL(etaZ(:,:,1),pkpipp)
            do kk=1,3  !go from (nw,nz,nquv) to (nw,nquv)
            alfa(:,kk) = MATMUL(etaZ(:,:,kk+1),pkpipp)
            beta(:,kk) = MATMUL(roZ(:,:,kk),pkpipp)
            emis(:,kk) = MATMUL(epsZ(:,:,kk),pkpipp)
            enddo
            emis(:,4) = MATMUL(epsZ(:,:,4),pkpipp)

    !.....................................................................
    !FULL EXACT MAGNUS EVOLUTION OPERATOR UNTIL ORDER 1

            !Build Omega-hat (Lorentz hat) and Omega-tilde (Lorentz tilde) for all frequencies
            do ii=1,3 !Efficient calculation without matrix inversions
                call fill_Lorentz_freqs(Omhat(:,:,:),alfa(:,ii),beta(:,ii),ii) !return 4x4 Omega hat
                call fill_Lorentz_freqs(Omtil(:,:,:),beta(:,ii),-alfa(:,ii),ii) !return 4x4 Ometa tilde
            enddo    !Results are (kw, 4column,4row)

            !Most efficient way I have found to exchange dimensions for speeding up last step
            !$OMP PARALLEL DO !--> try later
            do kk =1,nl
                do jj=1,4
                    do ii=1,4
                        fomhat(ii,jj,kk)=Omhat(kk,ii,jj)
                        fomtil(ii,jj,kk)=Omtil(kk,ii,jj)
                    enddo
                enddo
                fomhat2(1:4,1:4,kk)=MATMUL(fomhat(1:4,1:4,kk),fomhat(1:4,1:4,kk))
            enddo  !Results are (4column,4row,kw)
            !$OMP END PARALLEL DO
            !invomhat= reshape(Omhat, shape(invomhat), order = [2,3,1]) 

            !.................
            qq = 2.0*dot_productF2(alfa,beta) !--> THIS HAS THE KEY SIGN OF f1b
            rr = dot_productF2(alfa,alfa) - dot_productF2(beta,beta)                        
            !call dot_product_signF2(2.0*alfa,beta,qq,qqsign) !--> THIS HAS THE KEY SIGN OF f1b

            !substituting this routine by its operations directly is faster
            call get_phi1_gs(nl,qq,rr,tau,g1,ga,gb,g2)

            !I = I_0 + /Phi_1 * (A_bar * I_0 + E_bar) !SECOND EXPRESSION
            !with A_bar=-K_bar=-(Lorentz_bar+tau*identt4
            do kk =1,nl
                !fevolop(1:4,1:4,kk) = f1(kk)*identt4(1:4,1:4) + &
                !    fa(kk)*fomhat(1:4,1:4,kk) + fb(kk) * fomtil(1:4,1:4,kk) + &
                !    f2(kk)*fomhat2(1:4,1:4,kk)  
         
                phi1(1:4,1:4,kk) = g1(kk)*identt4(1:4,1:4) +&
                ga(kk)*fomhat(1:4,1:4,kk) + & 
                gb(kk)*fomtil(1:4,1:4,kk) + &
                g2(kk)*fomhat2(1:4,1:4,kk)

                kappa_bar=fomhat(1:4,1:4,kk)+tau(kk)*identt4

                !try transposing stokes before multiplying and retransposing again or redefine stokes
                stoks(1:4,kk) = stoks(1:4,kk) + matmul( phi1(1:4,1:4,kk), matmul(-kappa_bar,stoks(1:4,kk)) + emis(kk,1:4) ) 
            enddo
        
 end subroutine Magnus_FormSol_1b
! ------------------------------------------------------------------------ -
! Magnus inhomogeneous method 1 (I = Oevol*I_0 + /Phi_1 *E_bar) INCLUDING second order
! -------------------------------------------------------------------------
 subroutine Magnus_FormSol_2(nl,np,ds,epsZ,etaZ,roZ,stoks)
    integer, intent(in) :: nl,np
    real(kind=8),intent(in) :: ds(:),etaZ(:,:,:),roZ(:,:,:),epsZ(:,:,:)!, eps(:,:,:),eta(:,:,:),rho(:,:,:)
    !real(kind=8),dimension(:,:),intent(in) :: epI,epQ,epU,epV!,etI,etQ,etU,etV,roQ,roU,roV !kw,kz
    real(kind=8),intent(inout) :: stoks(:,:)

    real(kind=8) ::pkpipp(np) !pk,pi
    real(kind=8),dimension(nl,4,4) :: omhat,omtil, evolop  !,intent(out)
    real(kind=8) :: alfa(nl,3),beta(nl,3),emis(nl,4), alfa2(nl,3),beta2(nl,3)!coef(0:10,nl)
    real(kind=8),dimension(4,4,nl) :: fomhat,fomhat2,fomtil,fevolop,phi1
    real(kind=8) :: tau(nl),qq(nl),rr(nl),corr
    real(kind=8),dimension(nl) :: f1,fa,fb,f2, g1,ga,gb,g2
    !real(kind=8),dimension(nl) :: hh,hh_2,bhat_2,bhat,btil_2,btil,dhat,dtil
    !real(kind=8),dimension(nl) :: f1,fa,fb,f2, g1,ga,gb,g2,Chat, Ctil, Shat, Stil,exptau,tau_2,comfac
    integer::kz,kw,ii,jj,kk

    !real(kind=8),dimension(nl) :: Feps, OFeps, Fptil,OFptil,Fphat,OFphat,aux1,aux2,qqsign


    !INIT QUADRATURE WEIGHTS: 
    !Valid for Magnus Piecewise, Linear and Order 4 Schemes    
    ! k(O)*---ds(k)---*i(M)*---ds(k+1)---*k+1(P)
    pkpipp= quadrature_weights(np,ds)!ds is already a small chunk

    !Integration of optical coefficients alog the ray
            tau(:)=MATMUL(etaZ(:,:,1),pkpipp)
            do kk=1,3  !go from (nw,nz,nquv) to (nw,nquv)
            alfa(:,kk) = MATMUL(etaZ(:,:,kk+1),pkpipp)
            beta(:,kk) = MATMUL(roZ(:,:,kk),pkpipp)
            emis(:,kk) = MATMUL(epsZ(:,:,kk),pkpipp)
            enddo
            emis(:,4) = MATMUL(epsZ(:,:,4),pkpipp)

    !...............APPLY ORDER 2 MAGNUS EXPANSION..............................
    !fill cross products to get order-2 corrections alfa2,beta2 and add them to alfa and beta 
    !limiting heights for the commutator depends on the number of points
    ii=1 ;jj=np 
    corr=ds(1)
    do kk=2,np-1 !only enters the loop when np > 2
        corr=corr+ds(kk) !calculate total DeltaS of the interval
    enddo
    !if (synthesis_method==8) corr=ds(1)
    !if (synthesis_method==9) corr=ds(1)+ds(2)
    !corr= -((ds(jj)+ds(ii))**2.0)/12.d0 !here goes the total DeltaS of the interval
    corr= -(corr*corr)/12.d0 !here goes the total DeltaS of the interval
    beta(:,:)=beta(:,:)+corr*(crossp(etaZ(:,ii,2:4),etaZ(:,jj,2:4))-crossp(roZ(:,ii,1:3),roZ(:,jj,1:3)) )
    alfa(:,:)=alfa(:,:)+corr*(crossp(etaZ(:,jj,2:4),roZ(:,ii,1:3))-crossp(etaZ(:,ii,2:4),roZ(:,jj,1:3)) )
    !beta2(:,1:3)=crossprod(etaZ(:,ii,2:4),etaZ(:,jj,2:4)) -crossprod(roZ(:,ii,1:3),roZ(:,jj,1:3)) 
    !alfa2(:,1:3)=crossprod(etaZ(:,jj,2:4),roZ(:,ii,1:3)) -crossprod(etaZ(:,ii,2:4),roZ(:,jj,1:3)) 
    !beta(:,:)=beta(:,:)+corr*beta2(:,:)    ;alfa(:,:)=alfa(:,:)+corr*alfa2(:,:)

    !........BUILD LORENTZ MATRICES................................................

            !Build Omega-hat (Lorentz hat) and Omega-tilde (Lorentz tilde) for all frequencies
            do ii=1,3 !Efficient calculation without matrix inversions
                call fill_Lorentz_freqs(Omhat(:,:,:),alfa(:,ii),beta(:,ii),ii) !return 4x4 Omega hat
                call fill_Lorentz_freqs(Omtil(:,:,:),beta(:,ii),-alfa(:,ii),ii) !return 4x4 Ometa tilde
            enddo    !Results are (kw, 4column,4row)

            !Most efficient way I have found to exchange dimensions for speeding up last step
            !$OMP PARALLEL DO !--> try later
            do kk =1,nl
                do jj=1,4
                    do ii=1,4
                        fomhat(ii,jj,kk)=Omhat(kk,ii,jj)
                        fomtil(ii,jj,kk)=Omtil(kk,ii,jj)
                    enddo
                enddo
                fomhat2(1:4,1:4,kk)=MATMUL(fomhat(1:4,1:4,kk),fomhat(1:4,1:4,kk))
            enddo  !Results are (4column,4row,kw)
            !$OMP END PARALLEL DO
            !invomhat= reshape(Omhat, shape(invomhat), order = [2,3,1]) 

            !.................
            qq = 2.0*dot_productF2(alfa,beta) !--> THIS HAS THE KEY SIGN OF f1b
            rr = dot_productF2(alfa,alfa) - dot_productF2(beta,beta)                        
            !call dot_product_signF2(2.0*alfa,beta,qq,qqsign) !--> THIS HAS THE KEY SIGN OF f1b

            !substituting this routine by its operations directly is faster
            !call get_Lorentz_funs(nl,qq,rr,tau,f1,fa,fb,f2,g1,ga,gb,g2)
            call get_Lorentz_funs(nl,qq,rr,tau,f1,fa,fb,f2,g1,ga,gb,g2)

            do kk =1,nl
                fevolop(1:4,1:4,kk) = f1(kk)*identt4(1:4,1:4) + &
                    fa(kk)*fomhat(1:4,1:4,kk) + fb(kk) * fomtil(1:4,1:4,kk) + &
                    f2(kk)*fomhat2(1:4,1:4,kk)  

                !phi1(1:4,1:4,kk) = -(2.d0/qq(kk))*matmul(identt4(1:4,1:4)-fevolop(1:4,1:4,kk), fomtil(1:4,1:4,kk) )

                phi1(1:4,1:4,kk) = g1(kk)*identt4(1:4,1:4) +&
                ga(kk)*fomhat(1:4,1:4,kk) + & 
                gb(kk)*fomtil(1:4,1:4,kk) + &
                g2(kk)*fomhat2(1:4,1:4,kk)  

                !try transposing stokes before multiplying and retransposing again or redefine stokes
                stoks(1:4,kk) = matmul(fevolop(1:4,1:4,kk),stoks(1:4,kk))+matmul(phi1(1:4,1:4,kk),emis(kk,1:4)) 
            enddo
        
 end subroutine Magnus_FormSol_2
! ------------------------------------------------------------------------ -
! EDGAR: CALCULATE quadrature rules and integrate all optical coefficients 
!        by blocks along the ray.
! -------------------------------------------------------------------------
 subroutine Magnus_FormSol_TEST(nl,np,ds,epsZ,etaZ,roZ,stoks)
    integer, intent(in) :: nl,np
    real(kind=8),intent(in) :: ds(:),etaZ(:,:,:),roZ(:,:,:),epsZ(:,:,:)
    real(kind=8),intent(inout) :: stoks(:,:)

    real(kind=8),dimension(nl,4,4) :: omhat,omtil, evolop  !,intent(out)
    real(kind=8) :: tau(nl),alfa(nl,3),beta(nl,3),emis(nl,4)!coef(0:10,nl)
    real(kind=8),dimension(4,4,nl) :: fomhat,fomhat2,fomtil,fevolop,phi1
    real(kind=8),dimension(nl) :: qq,rr,hh,hh_2,bhat_2,bhat,btil_2,btil,dhat,dtil,dhdt
    real(kind=8),dimension(nl) :: f1h,fah,fbh,f2h, Chat, Ctil, Shat, Stil,exptau,tau_2,comfac
    real(kind=8) ::pkpipp(np) !pk,pi
    integer::kz,kw,ii,jj,kk

    real(kind=8),dimension(nl) :: Feps, OFeps, Fptil,OFptil,Fphat,OFphat,aux1,aux2,qqsign


    !INIT QUADRATURE WEIGHTS: Scheme:      k(O)*---ds(k)---*i(M)*---ds(k+1)---*k+1(P)
    pkpipp= quadrature_weights(np,ds)!ds is already a small chunk

    !PROGRAM ADAPTATIVE RECURSIVE QUADRATURE AND COMPOSED QUADRATURE RULE
    !add also routine to select the cuadrature coeffs for 1,2, ans 3 points (still with parab interpolants)

    !Integration of optical coefficients alog the ray
            tau(:)=MATMUL(etaZ(:,:,1),pkpipp)
            do kk=1,3
            alfa(:,kk) = MATMUL(etaZ(:,:,kk+1),pkpipp)
            beta(:,kk) = MATMUL(roZ(:,:,kk),pkpipp)
            emis(:,kk) = MATMUL(epsZ(:,:,kk),pkpipp)
            enddo
            emis(:,4) = MATMUL(epsZ(:,:,4),pkpipp)
    !.....................................................................
        !COULD BE BETTER TO TRANSPOSE HERE ALFA , BETA Y EMISS, AND THUS WORK DIRECTLY IN (4,4,kw)
        !TO AVOID TRANPOSING THE TWO OM MATRICES BELOW WHICH HAVE 16 ELEMENTS EACH
        !OR JUST FILL LORENTZ ALONG WAVELENGTH, NOT ALONG STOKES!

            ! !$OMP PARALLEL DO --> try this
            ! do kk =1,nl!go from (nw,nquv) to (nquv,nw)
            !         do ii=1,3
            !             alfa(ii,kk)=Nalfa(kk,ii)
            !             beta(ii,kk)=Nbeta(kk,ii)
            !             emis(ii,kk)=Nemis(kk,ii)
            !         enddo
            !         emis(4,kk)=Nemis(kk,4)
            ! enddo  
            ! ! !$OMP END PARALLEL DO

           !composition of Omega hat (Lorentz hat) and Omega tilde (Lorentz tilde) for one frequency:
        ! do kw=1,nl
        !     call  fill_Lorentz_matrix(Omhat(:,:,kw),alfa(:,kw),beta(:,kw)) !return 4x4 Omega hat
        !     call  fill_Lorentz_matrix(Omtilde(:,:,kw),beta(:,kw),-alfa(:,kw)) !return 4x4 Ometa tilde
        !     qq(kw) = dot_product(2.d0*alfa(:,kw),beta(:,kw))
        !     rr(kw) = dot_product(alfa(:,kw),alfa(:,kw)) -dot_product(beta(:,kw),beta(:,kw))
        !     !do rest of calcualtions here frequcny by frequency
        ! end do

    !.....................................................................
    !FULL EXACT MAGNUS EVOLUTION OPERATOR UNTIL ORDER 1

            !Build Omega-hat (Lorentz hat) and Omega-tilde (Lorentz tilde) for all frequencies
            do ii=1,3 !Efficient calculation without matrix inversions
                call fill_Lorentz_freqs(Omhat(:,:,:),alfa(:,ii),beta(:,ii),ii) !return 4x4 Omega hat
                call fill_Lorentz_freqs(Omtil(:,:,:),beta(:,ii),-alfa(:,ii),ii) !return 4x4 Ometa tilde
            enddo    !Results are (kw, 4column,4row)
            !Most efficient way I have found to exchange dimensions for speeding up last step
            !invomhat= reshape(Omhat, shape(invomhat), order = [2,3,1]) 
            !$OMP PARALLEL DO --> try this
            do kk =1,nl
                do jj=1,4
                    do ii=1,4
                        fomhat(ii,jj,kk)=Omhat(kk,ii,jj)
                        fomtil(ii,jj,kk)=Omtil(kk,ii,jj)
                    enddo
                enddo
                fomhat2(1:4,1:4,kk)=MATMUL(fomhat(1:4,1:4,kk),fomhat(1:4,1:4,kk))
            enddo  !Results are (4column,4row,kw)
            ! !$OMP END PARALLEL DO

            !.................ADD SIGNS EDGAR Here and in trigo functions!....................................................
            qq = 2.0*dot_productF2(alfa,beta) !--> THIS HAS THE KEY SIGN OF f1b
            qqsign = get_signsF1(qq)
            !call dot_product_signF2(2.0*alfa,beta,qq,qqsign) !--> THIS HAS THE KEY SIGN OF f1b
            
            rr = dot_productF2(alfa,alfa) - dot_productF2(beta,beta)            
            hh_2 = rr*rr + qq*qq   ;  hh= DSQRT(hh_2) !bhat_2+btil2
            bhat_2= (hh+rr)/2.d0   ;  bhat= DSQRT(bhat_2) ! bhat and btil are modules:
            btil_2= (hh-rr)/2.d0   ;  btil= DSQRT(btil_2) !their signs only matter in f1b and are accounted by qqsign


            Chat=DCOSH(bhat) ; Ctil=DCOS(btil) ; Shat=DSINH(bhat) ; Stil=DSIN(btil)

            exptau=DEXP(-tau)
            comfac=exptau/hh

            !Special functions * hh: 
            f1h = comfac*(btil_2 * Chat +bhat_2*Ctil )!f0h !division by hh is made more efficiently in evolop
            fah= - comfac*(bhat*Shat + btil*Stil)  !fah= - (bhat*Shat + btil*Stil) !f1ah
            fbh= qqsign*comfac*(bhat*Stil - btil*Shat) !f1bh
            f2h = comfac*(Chat - Ctil ) !f2


            do kk =1,nl
                fevolop(1:4,1:4,kk) = f1h(kk)*identt4(1:4,1:4) + &
                    fah(kk)*fomhat(1:4,1:4,kk) + fbh(kk) * fomtil(1:4,1:4,kk) + &
                    f2h(kk)*fomhat2(1:4,1:4,kk)  
            enddo
    !..................................................................... 
    
    !Feps=bhat_2/hh     ; OFeps= 1.d0 - Feps
    Fphat=bhat/tau   ; OFphat= (1.d0 - Fphat*Fphat)*hh
    Fptil=btil/tau   ; OFptil= (1.d0 + Fptil*Fptil)*hh
    
   !................................................................... 
    !CALCULATE FORMAL INHOMOGENEOUS SOLUTION Carlin, Blanes, & Casas (2024)
            
    !CALCULATE PHI_1 FUNCTION REUSING MATRICES AND SOME VARIABLES 
            
    aux1=(exptau*(Ctil+Fptil*Stil)-1.d0)/OFptil
    aux2=(exptau*(Chat+Fphat*Shat)-1.d0)/OFphat

            f1h= bhat*Fphat*aux1 - btil*Fptil*aux2
            f2h= -(aux1 + aux2)/tau
    
    aux1=(exptau*(Shat+Fphat*Chat)-Fphat)/OFphat
    aux2=(exptau*(Stil+Fptil*Ctil)-Fptil)/OFptil
    
            fah= Fphat*aux1 + Fptil*aux2  !for Lhat
            fbh= qqsign * (Fptil*aux1 - Fphat*aux2)   !for Ltil  -->defines signs

            !.....................................................................
            do kk =1,nl
                phi1(1:4,1:4,kk) = f1h(kk)*identt4(1:4,1:4) +&
                 fah(kk)*fomhat(1:4,1:4,kk) + & 
                fbh(kk) * fomtil(1:4,1:4,kk) + &
                f2h(kk)*fomhat2(1:4,1:4,kk)  
        
                !try transposing stokes before multiplying and retransposing again or redefine stokes
                stoks(1:4,kk) = matmul(fevolop(1:4,1:4,kk),stoks(1:4,kk))+matmul(phi1(1:4,1:4,kk),emis(kk,1:4)) 
            enddo
 end subroutine Magnus_FormSol_TEST
    
  ! ------------------------------------------------------------------------ -
  ! EDGAR: Main synthesis routine of Hazel Experimental with all RT methods
  ! -------------------------------------------------------------------------
 !subroutine synth_methods(nl,ds,epI,epQ,epU,epV,etI,etQ,etU,etV,roQ,roU,roV,stkOut)
 subroutine synth_methods(nl,ds,ep,et,ro,stkOut,oldI0,first)
 integer, intent(in) :: nl!ds(kz), but only dn number of points!
 logical, intent(in):: first
 real(kind=8),intent(in) :: ds(:), ep(:,:,:),et(:,:,:),ro(:,:,:)
 !real(kind=8),dimension(:,:),intent(in) :: epI,epQ,epU,epV,etI,etQ,etU,etV,roQ,roU,roV !kw,kz
 real(kind=8),intent(inout) :: stkOut(:,:),oldI0(:,:)!use stkOut as stkIn incom boundary condition and updating it
 
 real(kind=8) ::  I0, eta0, psim, psi0, psip,dtau, source(4)!,StokesM(4),Stokes0(4)
 real(kind=8),dimension(4,4) ::  kappa_star, O_evol, psi_matrix,m2, m1
 integer :: ii,qq,uu,vv,i,w,kz
 real(kind=8) :: kappaM(4,4),kappa0(4,4),kappaP(4,4)!DELO
 real(kind=8) :: dtM,dtP,U0,dta,ff1,ff2,psi0p,psipp,ff,memory(4) !DELO
 !real(kind=8),dimension(nl) :: epsi,epsq,epsu,epsv,etai,etaq,etau,etav
 !real(kind=8),dimension(nl) :: rhoi,rhoq,rhou,rhov

    if (first) synthesis_method = 11

    !----------------------------------------------------------------------------------
    if (synthesis_method == 0) then 
    !****************       
    ! ONLY EMISSIVITY: Accumulation of emission without absorption. Normalize later in Python
    !****************
        !do ii = 1, 4 !we are using stkOut as stkIn incomming boundary condition and updating it
        !    stkOut(:,ii) = stkOut(:,ii) + ep(:,1,ii)!the 1 one here is current height 
        !enddo
        do w = 1, nl !we are using stkOut as stkIn incomming boundary condition and updating it
            stkOut(1:4,w) = stkOut(1:4,w) + ep(w,1,1:4)!the 1 one here is current height 
        enddo

    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 1) then  !TRAPEZOIDAL METHOD

        do w = 1, nl 
            kz=1 !current height
            call fill_absorption_matrix(kappaM,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
             
            kz=2 !current height    
            call fill_absorption_matrix(kappa0,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
                           
            m1 = identt4 - 0.5d0*ds(1) * kappaM
            m2 = identt4 + 0.5d0*ds(1) * kappa0
            call invert(m2)
            
            !beta(kz) was already multiplied in python when !=1
            source(:) = 0.5d0*(ds(1)*ep(w,1,:)+ ds(2)*ep(w,2,:))
 
            stkOut(:,w) = matmul(m2,matmul(m1,stkOut(:,w))+source)
        enddo

    endif

    !----------------------------------------------------------------------------------
    if (synthesis_method == 11) then !DELO (LINEAR)
        kz=1 !current height
                   
        do w = 1, nl !point by point(frequency) calling... and Not efficient:dimensions should be exchanged
            !call fill_absorption_matrix(kappa_star,etI(w,kz),etQ(w,kz),etU(w,kz),etV(w,kz),roQ(w,kz),roU(w,kz),roV(w,kz))
            call fill_absorption_matrix(kappa_star,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappa_star = kappa_star/et(w,kz,1) - identt4  
            source(:) = ep(w,kz,:)/et(w,kz,1)!beta(kz) was already multiplied in python when !=1
 
            dtau = et(w,kz,1) * ds(kz) !con kz only the present point along ray i.e. 1   
            
            psi0 = (dtau - 1.d0 + DEXP(-dtau) ) / dtau
            psim = 1.d0 - DEXP(-dtau) - psi0

                
            m1 = DEXP(-dtau )*identt4 - psim * kappa_star
            m2 = identt4 + psi0 * kappa_star

            call invert(m2)
             
            stkOut(:,w) = matmul(m2,matmul(m1,stkOut(:,w))+(psim+psi0)*source)

        enddo
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 15) then !DELO-PAR in upwind point
        !ds IS HERE ALWAYS EXPECTED TO BE POSITIVE but we force it just in case

        do w = 1, nl !point by point(frequency) calling... 
            kz=1 !current height    
            call fill_absorption_matrix(kappaM,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappaM = kappaM/et(w,kz,1) - identt4  
            dtM = et(w,kz,1) * ABS(ds(kz)) !con kz only the present point along ray i.e. 1   

            kz=2 !current height    
            call fill_absorption_matrix(kappa0,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappa0 = kappa0/et(w,kz,1) - identt4  
            dtP = et(w,kz,1) * ABS(ds(kz)) !con kz only the present point along ray i.e. 1   

            kz=3 !current height    
            call fill_absorption_matrix(kappaP,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappaP = kappaP/et(w,kz,1) - identt4  

            U0= DEXP(-dtP)
            ff=(1.d0-U0)/dtP
            dta = dtM+dtP
            !parabolic terms for k-1,k,k+1 but integrating between k and k+1:
            psim= ((2.d0-dtP)-(2.d0+dtP)*U0)/(dta*dtM) !boundary point
            psi0= ((dta-2.d0)-(dtP*(dtM-1.0)+(dtM - 2.d0))*U0)/(dtM*dtP) 
            psip= 1.d0 + (-2*dtP+(2.d0-dtM)*(1.d0-U0))/(dta*dtP)  !advanced point
            !psip= 1.d0 - (2*dtP+(dtM-2.d0)-(dtM - 2.d0)*U0)/(dta*dtP) !advanced point
            !linear terms integrating between k and k+1:
            psi0p = ff-U0! DELAYED POINT   (1.d0-U0)/dtP -U0
            psipp = 1.d0-ff !ADVANCED POINT   (U0+dtM-1.d0)/dtP

            m1 = U0*identt4 - psi0p * kappa0
            m2 = identt4 + psipp * kappaP
            call invert(m2)

            !beta(kz) was already multiplied in python when !=1
            source(:) = psim*ep(w,1,:)/et(w,1,1)+ psi0*ep(w,2,:)/et(w,2,1)+psip*ep(w,3,:)/et(w,3,1)
            
            stkOut(:,w) = matmul(m2,matmul(m1,stkOut(:,w))+source(:))

        enddo
        
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 2) then !DELO-PARABOLLIC
        do w = 1, nl !point by point(frequency) calling... 
            kz=1 !current height    
            call fill_absorption_matrix(kappaM,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappaM = kappaM/et(w,kz,1) - identt4  
            dtM = et(w,kz,1) * ABS(ds(kz)) !con kz only the present point along ray i.e. 1   

            kz=2 !current height    
            call fill_absorption_matrix(kappa0,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappa0 = kappa0/et(w,kz,1) - identt4  
            dtP = et(w,kz,1) * ABS(ds(kz)) !con kz only the present point along ray i.e. 1   

            kz=3 !current height    
            call fill_absorption_matrix(kappaP,et(w,kz,1),et(w,kz,2),et(w,kz,3),et(w,kz,4),ro(w,kz,1),ro(w,kz,2),ro(w,kz,3))
            kappaP = kappaP/et(w,kz,1) - identt4  

            U0= DEXP(-dtP)
            dta = dtM+dtP

            ff=(1.d0-U0)/dtP
            !!parabolic terms for k-1,k,k+1 but integrating between k and k+1:
            psim= ((2.d0-dtP)-(2.d0+dtP)*U0)/(dta*dtM)!boundary point
            psi0=((dta-2.d0)-(dtP*(dtM-1.0)+(dtM - 2.d0))*U0)/(dtM*dtP)
            psip= 1.d0 + (-2*dtP+(2.d0-dtM)*(1.d0-U0))/(dta*dtP)  !advanced point
            
            !ff=((2.d0-dtM)*(1.d0-U0)-2.d0*dtP)/(dta*dtM)
            !psim= ff + dta*(1.d0-U0)
            !psi0= -dta*ff/dtP -(1.d0-U0)/dtM - U0
            !psip= 1.d0 + ff*dtM/dtP

            m1 = U0*identt4 - psi0 * kappa0
            m2 = identt4 + psip * kappaP
            call invert(m2)

            memory=matmul(kappaM,oldI0(:,w)) !result 4x1
            !beta(kz) was already multiplied in python when !=1
            source(:) = psim*(ep(w,1,:)/et(w,1,1)- memory)+ psi0*ep(w,2,:)/et(w,2,1)+psip*ep(w,3,:)/et(w,3,1)
            
            stkOut(:,w) = matmul(m2,matmul(m1,stkOut(:,w))+source(:))

        enddo
        
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

    if (synthesis_method == 6) then !MAgnus 3 points 
        !call Magnus_FormSol_1(nl,ds,ep,et,ro,stkOut)
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 7) then !Magnus restricted to 1 point as EvolOp
        !call Magnus_FormSol_1(nl,ds,ep,et,ro,stkOut)
        !call Magnus_FormSol_2(stkOut,nl,ds,epI,epQ,epU,epV,etI,etQ,etU,etV,roQ,roU,roV)
        !call Magnus_FormSol_3(nl,ds,epI,epQ,epU,epV,etI,etQ,etU,etV,roQ,roU,roV)
    
    endif

    !----------------------------------------------------------------------------------
    if (synthesis_method == 8) then !Magnus Trapezoidal
        !call Magnus_FormSol_1(nl,ds,ep,et,ro,stkOut)
    endif
    !----------------------------------------------------------------------------------
    if (synthesis_method == 9) then !Magnus including Order2 of expansion
        !call Magnus_FormSol_1(nl,ds,ep,et,ro,stkOut)
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
            stkOut(:,w)= matmul(O_evol,stkOut(:,w)) + matmul(Psi_matrix,source)    
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
            stkOut(:,w)= matmul(O_evol,stkOut(:,w)) + matmul(Psi_matrix,source)    
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
 
        !call init_psf()
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
        !call init_psf()

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
        !call convolve(in_fixed%no, output)

    
    end subroutine do_synthesis





end module synth
