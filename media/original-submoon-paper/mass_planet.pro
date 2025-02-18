;;; GOAL: make plot like Fig 2 of Barnes & O'Brien (2002) to
;;; constrain masses of satellites that could have moons

loadct,0,/silent

;; CGS UNITS
kmcm = 1.d5
yearsec = 3.154d7
G = 6.67d-8


;; Solar System data
rearthkm = 6378. ;km
rearth = 6378.d5 ;km
mearth = 5.974d27

Mmars = 0.107*mearth
Rmars = 0.53*rearth

Mluna = 0.012*mearth
Rluna = 0.273*rearth

mpsyche = 2.72d22
rpsyche = 113.*kmcm

mvesta = 2.589d23 ;g
rvesta = 262.7*kmcm

mceres = 1.5d-4*mearth
rceres = 473.*kmcm

Mjup = 317.8*mearth
Rjup = 69911.*kmcm

Msat = 95.16*mearth
Rsat = 58232.*kmcm

Mnep = 17.15*mearth
Rnep = 24622.*kmcm

Mura = 14.54*mearth
Rura = 25362.*kmcm

;; from Teachey & Kipping (2018, Sci Adv)
MK1625 = 4.*Mjup
RK1625 = 11.4*rearth


if (N_ELEMENTS(a_ss) EQ 0) THEN BEGIN
   readcol,'satellites_all.txt',host_ss,name_ss,a_ss,R_ss,M_ss,format='a,a,f,f,f'
   m_ss = m_ss*1.d3*1.d16 ;grams
   R_ss = 0.5*R_ss ;(file has diameters)
   a_ss = a_ss*kmcm
ENDIF


;; Define age of system
T = 4.6d9*yearsec  ;; 5 Gyr


;; Stability parameter (in fraction of Hill radius; Domingos et al )
f = 0.4895



;; characteristic moon-of-moon properties
;rhomm = 2.
;Rmm = 10.*kmcm
;Mmm = 4./3.*!pi*rhomm*(Rmm*kmcm)^3 ;grams


;; rhomoon = 2.
;; ;Rmoon = findgen(10001) ;km
;; Rmoon = 3000.*kmcm  ;; big-ass moon
;; Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams
;; k2p = 0.25
;; Qmoon = 12.


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; 1. Solve for Mcrit vs ap for a single big moon around Jupiter
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Rmoon = 3000.*kmcm  ;; big-ass moon
;; Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

;; ap = findgen(501)*rjup 
;; mcrit = (2./13.)*((f*ap)^3/(3.*Mjup))^(13./6.)*Mmoon^(8./3.)*Qmoon/(3.*k2p*T*Rmoon^5*sqrt(G))
;; plot,ap/rjup,mcrit/mearth,/ylog,xtitle='ap (Rjup)',ytitle='Mcrit (ME)',charsize=2

;STOP
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; 2. Fix the mass/size of the satellite moon. Calculate the
;;;; critical mmoon-amoon that can host such a moon-of-moon for JSUN
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!p.font=6
ccs = 1.
sss = 2.

;; characteristic moon-of-moon properties: test radii of 1, 10,
;; and 100 km
rhomm = 2.
Rmma = 5.0*kmcm
Mmma = 4./3.*!pi*rhomm*(Rmma)^3 ;grams

Rmmb = 10.*kmcm
Mmmb = 4./3.*!pi*rhomm*(Rmmb)^3 ;grams

Rmmc = 20.*kmcm
Mmmc = 4./3.*!pi*rhomm*(Rmmc)^3 ;grams

Rmmd = 500.0*kmcm
Mmmd = 4./3.*!pi*rhomm*(Rmmd)^3 ;grams


;;; MULTIPLOT PARAMS

x1 = [0.11,0.6,0.11,0.6,0.11,0.6]
x2 = [0.46,0.95,0.46,0.95,0.46,0.95]
y1 = [0.73,0.73,0.4,0.4,0.07,0.07]
y2 = [0.95,0.95,0.62,0.62,0.29,0.29]


;ops,file='moons_moons.eps',form=4,xsize=15.,ysize=7.


;; JUPITER

cc = 0 

;; Make the large moon float
rhomoon = 2.5
k2p = 0.25
Qmoon = 100.
Rmoon = findgen(1001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = Mjup

acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/Rjup,Mmoon/mearth,/ylog,xtitle='ap (Rjup)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 5000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/Rjup le xmax,ntmp)

plot,acrita/Rjup,Rmoon/kmcm,/ylog,xtitle='Orbital distance (Jupiter radii)',$
     ytitle='Radius of moon (km)',$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rjup],thick=5,$
     /nodata,/xstyle,title='Jupiter',yrange=[10.,ymax],/ystyle,$
     position=[x1(cc),y1(cc),x2(cc),y2(cc)]

polyfill,[acritb(tmp)/Rjup,reverse(acritb(tmp)/Rjup)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200

xyouts,40.,2000.,'Moons of moons!C        stable',color=255,charthick=7
xyouts,15.,20.,'No moons of moons',charthick=5

plot,acrita/Rjup,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rjup],$
     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acritb/Rjup,Rmoon/kmcm,thick=5
oplot,acritc/Rjup,Rmoon/kmcm,thick=5,linestyle=2


jupsat = where(host_ss eq 'Jupiter',njupsat)
FOR i=0,njupsat-1 DO $
   oplot,[a_ss(jupsat(i))/rjup],[r_ss(jupsat(i))],psym=sym(1),symsize=alog10(r_ss(jupsat(i)))*0.5

xyouts,65.,300.,'R!dsub!n=20km',charsize=0.75,charthick=20,orientation=-15.
xyouts,65.,152.,'R!dsub!n=10km',charsize=0.75,charthick=20,orientation=-15.
xyouts,65.,70.,'R!dsub!n=5km',charsize=0.75,charthick=20,orientation=-15.



;; SATURN

cc = 1

;; Make the large moon float
rhomoon = 2.5
k2p = 0.25
Qmoon = 100.
Rmoon = findgen(1001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = Msat

acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/Rjup,Mmoon/mearth,/ylog,xtitle='ap (Rjup)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 5000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/Rsat le xmax,ntmp)

plot,acrita/Rsat,Rmoon/kmcm,/ylog,xtitle='Orbital distance (Saturn radii)',$
     ytitle='Radius of moon (km)',/noerase,$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rsat],thick=5,$
     /nodata,/xstyle,title='Saturn',yrange=[10.,ymax],/ystyle,$
     position=[x1(cc),y1(cc),x2(cc),y2(cc)]

polyfill,[acritb(tmp)/Rsat,reverse(acritb(tmp)/Rsat)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200

xyouts,40.,2000.,'Moons of moons!C        stable',color=255,charthick=7
xyouts,15.,20.,'No moons of moons',charthick=5

plot,acrita/Rsat,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rsat],$
     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acritb/Rsat,Rmoon/kmcm,thick=5
oplot,acritc/Rsat,Rmoon/kmcm,thick=5,linestyle=2


satsat = where(host_ss eq 'Saturn',nsatsat)
FOR i=0,nsatsat-1 DO $
   oplot,[a_ss(satsat(i))/rsat],[r_ss(satsat(i))],psym=sym(1),symsize=alog10(r_ss(satsat(i)))*0.5



;; URANUS

cc = 2

;; Make the large moon float
rhomoon = 2.5
k2p = 0.25
Qmoon = 100.
Rmoon = findgen(1001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = Mura

acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/Rura,Mmoon/mearth,/ylog,xtitle='ap (Rura)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 5000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/Rura le xmax,ntmp)

plot,acrita/Rura,Rmoon/kmcm,/ylog,xtitle='Orbital distance (Uranus radii)',$
     ytitle='Radius of moon (km)',/noerase,$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rura],thick=5,$
     /nodata,/xstyle,title='Uranus',yrange=[10.,ymax],/ystyle,$
     position=[x1(cc),y1(cc),x2(cc),y2(cc)]

polyfill,[acritb(tmp)/Rura,reverse(acritb(tmp)/Rura)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200

xyouts,40.,2000.,'Moons of moons!C        stable',color=255,charthick=7
xyouts,15.,20.,'No moons of moons',charthick=5

plot,acrita/Rura,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rura],$
     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acritb/Rura,Rmoon/kmcm,thick=5
oplot,acritc/Rura,Rmoon/kmcm,thick=5,linestyle=2


urasat = where(host_ss eq 'Uranus',nurasat)
FOR i=0,nurasat-1 DO $
   oplot,[a_ss(urasat(i))/rura],[r_ss(urasat(i))],psym=sym(1),symsize=alog10(r_ss(urasat(i)))*0.5



;; NEPTUNE

cc = 3

;; Make the large moon float
rhomoon = 2.5
k2p = 0.25
Qmoon = 100.
Rmoon = findgen(1001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = Mnep

acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/Rnep,Mmoon/mearth,/ylog,xtitle='ap (Rnep)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 5000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/Rnep le xmax,ntmp)

plot,acrita/Rnep,Rmoon/kmcm,/ylog,xtitle='Orbital distance (Neptune radii)',$
     ytitle='Radius of moon (km)',/noerase,$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rnep],thick=5,$
     /nodata,/xstyle,title='Neptune',yrange=[10.,ymax],/ystyle,$
     position=[x1(cc),y1(cc),x2(cc),y2(cc)]

polyfill,[acritb(tmp)/Rnep,reverse(acritb(tmp)/Rnep)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200


xyouts,40.,2000.,'Moons of moons!C        stable',color=255,charthick=7
xyouts,15.,20.,'No moons of moons',charthick=5

plot,acrita/Rnep,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rnep],$
     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acritb/Rnep,Rmoon/kmcm,thick=5
oplot,acritc/Rnep,Rmoon/kmcm,thick=5,linestyle=2


nepsat = where(host_ss eq 'Neptune',nnepsat)
FOR i=0,nnepsat-1 DO $
   oplot,[a_ss(nepsat(i))/rnep],[r_ss(nepsat(i))],psym=sym(1),symsize=alog10(r_ss(nepsat(i)))*0.5



;; EARTH

cc = 4

;; Make the large moon float
rhomoon = 3.34
Rmoon = findgen(1001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = Mearth

acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

ymax = 5000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/Rearth le xmax,ntmp)

plot,acrita/Rearth,Rmoon/kmcm,/ylog,xtitle='Orbital distance (Earth radii)',$
     ytitle='Radius of moon (km)',/noerase,$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rearth],thick=5,$
     /nodata,/xstyle,title='Earth',yrange=[10.,ymax],/ystyle,$
     position=[x1(cc),y1(cc),x2(cc),y2(cc)]

polyfill,[acritb(tmp)/Rearth,reverse(acritb(tmp)/Rearth)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200


xyouts,68.,2700.,'Moons of!C  moons!C  stable',color=255,charthick=7
xyouts,15.,20.,'No moons of moons',charthick=5

plot,acrita/Rearth,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rearth],$
     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acritb/Rearth,Rmoon/kmcm,thick=5
oplot,acritc/Rearth,Rmoon/kmcm,thick=5,linestyle=2


earthsat = where(host_ss eq 'Earth',nearthsat)
FOR i=0,nearthsat-1 DO $
   oplot,[a_ss(earthsat(i))/rearth],[r_ss(earthsat(i))],psym=sym(1),symsize=alog10(r_ss(earthsat(i)))*0.5



;; KEPLER-1625

cc = 5

;; Make the large moon float
rhomoon = 1.64
k2p = 0.12
Qmoon = 1000.
Rmoon = findgen(10001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = MK1625

acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

acritd = (1./f)*(3.*Mplan*(13./2.*Mmmd*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/RK1625,Mmoon/mearth,/ylog,xtitle='ap (RK1625)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 50000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/RK1625 le xmax,ntmp)

plot,acrita/RK1625,Rmoon/kmcm,/ylog,xtitle='Orbital distance (Kepler-1625b radii)',$
     ytitle='Radius of moon (km)',/noerase,$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/RK1625],thick=5,$
     /nodata,/xstyle,title='Kepler-1625b',yrange=[10.,ymax],/ystyle,$
     position=[x1(cc),y1(cc),x2(cc),y2(cc)]

polyfill,[acritb(tmp)/RK1625,reverse(acritb(tmp)/RK1625)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200


xyouts,45.,5000.,'Moons of moons!C        stable',color=255,charthick=7
xyouts,13.,20.,'No moons of moons',charthick=5

plot,acrita/RK1625,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/RK1625],$
     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acritb/RK1625,Rmoon/kmcm,thick=5
oplot,acritc/RK1625,Rmoon/kmcm,thick=5,linestyle=2


K1625sat = where(host_ss eq 'K1625b',nK1625sat)
FOR i=0,nK1625sat-1 DO $
   oplot,[a_ss(K1625sat(i))/rK1625],[r_ss(K1625sat(i))],psym=sym(1),symsize=alog10(r_ss(K1625sat(i)))*0.5


;cps

STOP

;;; MORE ON KEPLER-1625b


;; Define age of system
T = 9.0d9*yearsec  ;; 5 Gyr


loadct,12,/silent

;; Make the large moon float
rhomoon = 1.64
k2p = 0.12
Qmoon = 1000.
Rmoon = findgen(10001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = MK1625

;; characteristic moon-of-moon properties: 
Rmma = rvesta
Mmma = mvesta
acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

Rmmb = rceres
Mmmb = mceres
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

Rmmc = rluna;3000.*kmcm
Mmmc = mluna;4./3.*!pi*rhomm*(Rmmc)^3 ;grams
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

Rmmd = rmars;3000.*kmcm
Mmmd = mmars;4./3.*!pi*rhomm*(Rmmc)^3 ;grams
acritd = (1./f)*(3.*Mplan*(13./2.*Mmmd*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

Rmme = rpsyche;3000.*kmcm
Mmme = mpsyche;4./3.*!pi*rhomm*(Rmmc)^3 ;grams
acrite = (1./f)*(3.*Mplan*(13./2.*Mmme*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/RK1625,Mmoon/mearth,/ylog,xtitle='ap (RK1625)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 50000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/RK1625 le xmax,ntmp)

;ops,file='submoons_size.eps',form=4

ccs = 1.5

ywords = ['10!u4!n','2x10!u4!n','3x10!u4!n','4x10!u4!n','5x10!u4!n']

plot,acrita/RK1625,Rmoon/kmcm,xtitle='Orbital distance (Kepler-1625b radii)',$
     ytitle='Radius of moon (km)',$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/RK1625],thick=5,$
     /nodata,/xstyle,title='Effect of submoon size (Q!dmoon!n=1000)',yrange=[1000.,ymax],/ystyle,ytickname=ywords;,/ylog

;polyfill,[acritb(tmp)/RK1625,reverse(acritb(tmp)/RK1625)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200

xyouts,55.,25000.,'Submoons!C   stable',charthick=5,charsize=1.4
xyouts,3.,3000.,'Submoons unstable',charthick=5,charsize=1.4

arrow,30.,1.2d4,23.,8000.,thick=4,/data,/solid,hsize=350
arrow,50.,2.d4,57.,2.4d4,thick=4,/data,/solid,hsize=350

;plot,acrita/RK1625,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
;     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/RK1625],$
;     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acrita/RK1625,Rmoon/kmcm,thick=10
oplot,acritb/RK1625,Rmoon/kmcm,thick=10,color=30
oplot,acritc/RK1625,Rmoon/kmcm,thick=10,color=200
oplot,acritd/RK1625,Rmoon/kmcm,thick=10,color=100
oplot,acrite/RK1625,Rmoon/kmcm,thick=10,color=150


xyouts,3.,4.5d4,'Submoon!Cwith size,!Cmass of:',charthick=4
xyouts,9.,3.9d4,'Psyche',charthick=4,color=150
xyouts,21,3.7d4,'Vesta',charthick=4
xyouts,36.,3.9d4,'Ceres',charthick=4,color=30
xyouts,57.,3.9d4,"Earth's!CMoon",charthick=4,color=200
xyouts,87.,3.9d4,'Mars',charthick=4,color=100


K1625sat = where(host_ss eq 'K1625b',nK1625sat)
FOR i=0,nK1625sat-1 DO BEGIN
loadct,0,/silent
oplot,[a_ss(K1625sat(i))/rK1625],[r_ss(K1625sat(i))],psym=sym(1),symsize=alog10(r_ss(K1625sat(i)))*0.6,color=150
   oplot,[a_ss(K1625sat(i))/rK1625],[r_ss(K1625sat(i))],psym=sym(23),symsize=alog10(r_ss(K1625sat(i)))*0.6
loadct,12,/silent
ENDFOR

;cps

STOP


;;; MORE ON KEPLER-1625b: PARAMETER STUDY 2 -- EFFECT OF TIDAL Q


;; Define age of system
T = 9.0d9*yearsec  ;; 5 Gyr


loadct,12,/silent

;; Make the large moon float
rhomoon = 1.6
k2p = 0.12
Rmoon = findgen(10001)*10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = MK1625

;; characteristic moon-of-moon properties: 
rhomm = 2.0 ;; close to Ceres' value of 2.08
Rmma = rceres
Mmma = mceres 
Qmoon = 1.d2
acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

Rmmb = Rmma
Mmmb = Mmma
Qmoon = 1.d3
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

Rmmc = Rmma
Mmmc = Mmma
Qmoon = 1.d4
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

Rmmd = rmma
Mmmd = mmma
Qmoon = 1.d5

acritd = (1./f)*(3.*Mplan*(13./2.*Mmmd*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/RK1625,Mmoon/mearth,/ylog,xtitle='ap (RK1625)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 50000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/RK1625 le xmax,ntmp)


;ops,file='submoons_qmoon.eps',form=4

ccs = 1.5

ywords = ['10!u4!n','2x10!u4!n','3x10!u4!n','4x10!u4!n','5x10!u4!n']

plot,acrita/RK1625,Rmoon/kmcm,xtitle='Orbital distance (Kepler-1625b radii)',$
     ytitle='Radius of moon (km)',$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/RK1625],thick=5,$
     /nodata,/xstyle,title='Effect of Q!dmoon!n (Ceres-like submoon)',yrange=[1000.,ymax],/ystyle,ytickname=ywords;,/ylog

;polyfill,[acritb(tmp)/RK1625,reverse(acritb(tmp)/RK1625)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200

xyouts,75.,37000.,'Submoons!C   stable',charthick=5,charsize=1.4
xyouts,3.,3000.,'Submoons unstable',charthick=5,charsize=1.4

arrow,25.,1.2d4,18.,8000.,thick=4,/data,/solid,hsize=350
arrow,65.,3.d4,72.,3.4d4,thick=4,/data,/solid,hsize=350

;plot,acrita/RK1625,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
;     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/RK1625],$
;     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acrita/RK1625,Rmoon/kmcm,thick=10
oplot,acritb/RK1625,Rmoon/kmcm,thick=10,color=30
oplot,acritc/RK1625,Rmoon/kmcm,thick=10,color=200
oplot,acritd/RK1625,Rmoon/kmcm,thick=10,color=100
;oplot,acrite/RK1625,Rmoon/kmcm,thick=10,color=150


xyouts,3.,4.3d4,'Q!dmoon!n=',charthick=4,charsize=1.25
;xyouts,9.,3.9d4,'Psyche',charthick=4,color=150
xyouts,10,3.9d4,'10!u5!n',charthick=4,color=100,charsize=1.25
xyouts,25.,3.9d4,'10!u4!n',charthick=4,color=200,charsize=1.25
xyouts,36.,3.9d4,'10!u3!n',charthick=4,color=30,charsize=1.25
xyouts,51.,3.9d4,'100!n',charthick=4,charsize=1.25


K1625sat = where(host_ss eq 'K1625b',nK1625sat)
FOR i=0,nK1625sat-1 DO BEGIN
loadct,0,/silent
oplot,[a_ss(K1625sat(i))/rK1625],[r_ss(K1625sat(i))],psym=sym(1),symsize=alog10(r_ss(K1625sat(i)))*0.6,color=150
   oplot,[a_ss(K1625sat(i))/rK1625],[r_ss(K1625sat(i))],psym=sym(23),symsize=alog10(r_ss(K1625sat(i)))*0.6
loadct,12,/silent
ENDFOR


;cps



STOP

;; from Teachey & Kipping (2018, Sci Adv)

MK1625 = 4.*Mjup
RK1625 = 11.4*rearth

K1625sat = where(host_ss eq 'K1625b',nK1625sat)
MK1625bI = m_ss(K1625sat(0))
RK1625bI = r_ss(K1625sat(0))*kmcm
aK1625bI = a_ss(K1625sat(0))

k2p = 0.12
Rmoon = RK1625bI
Mmoon = MK1625bI
rhomoon = 3.*Mmoon/(4.*!pi*Rmoon^3) ; 1.64
amoon = aK1625bI
Mplan = MK1625

tmp = (1.+findgen(100))/100.
Qmoon = 10.^(1.+tmp*5.)

MminK1625 = (2./13.)*((f*amoon)^3/(3.*Mplan))^(13./6.)*$
            (4./3.*!pi*rhomoon)^(8./3.)*Qmoon*Rmoon^3/(3.*k2p*T*sqrt(G))


;rhomoon=1.6
RminK1625 = (3.*MminK1625/(4.*!pi*rhomoon))^(1/3.)

plot,qmoon,RminK1625/kmcm,xthick=4,ythick=4,xtitle='Q!dMoon!n',$
     ytitle='R!dmax!n (km)',thick=4,charsize=ccs,/xlog,/ylog

;plot,qmoon,MminK1625/mearth,xthick=4,ythick=4,xtitle='Q!dMoon!n',$
;     ytitle='M!dmax!n',thick=4,charsize=ccs,/xlog,/ylog



STOP

;;;; Max submoon mass calculation

Rmoon = rearth
rhomoon = 5.5
qmoon = 100.
k2moon = 0.47

T = 5.0d9*yearsec

frh = 0.4


STOP
;;;; SUBSUBMOON CALCULATION: take K-1625 with Ceres-mass submoon


;; characteristic moon-of-moon properties: test radii of 1, 10,
;; and 100 km
rhomm = 2.
Rmma = 5.0*kmcm
Mmma = 4./3.*!pi*rhomm*(Rmma)^3 ;grams

Rmmb = 10.*kmcm
Mmmb = 4./3.*!pi*rhomm*(Rmmb)^3 ;grams

Rmmc = 20.*kmcm
Mmmc = 4./3.*!pi*rhomm*(Rmmc)^3 ;grams

;; Make the large moon float
rhomoon = 2.5
k2p = 0.25
Qmoon = 100.
Rmoon = findgen(1001)/10.*kmcm
Mmoon = 4./3.*!pi*rhomoon*(Rmoon)^3 ;grams

Mplan = mceres

acrita = (1./f)*(3.*Mplan*(13./2.*Mmma*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritb = (1./f)*(3.*Mplan*(13./2.*Mmmb*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)
acritc = (1./f)*(3.*Mplan*(13./2.*Mmmc*(3.*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8./3.)*Qmoon)))^(6./13.))^(1/3.)

;plot,acrit/Rjup,Mmoon/mearth,/ylog,xtitle='ap (Rjup)',ytitle='Mcrit (ME)',charsize=ccs

ymax = 5000.
xmax = 100.

tmp = where(rmoon/kmcm lt ymax and acritb/Rceres le xmax,ntmp)

plot,acrita/Rceres,Rmoon/kmcm,/ylog,xtitle='Orbital distance (Ceres radii)',$
     ytitle='Radius of moon (km)',$
     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rceres],thick=5,$
     /xstyle,title='Nope',yrange=[0.1,ymax],/ystyle

polyfill,[acritb(tmp)/Rceres,reverse(acritb(tmp)/Rjup)],[Rmoon(tmp)/kmcm,dblarr(ntmp)+ymax],color=200

xyouts,10.,2000.,'Moons of moons!C        stable',color=255,charthick=7
xyouts,50.,10000.,'No moons of moons',charthick=5

;plot,acrita/Rjup,Rmoon/kmcm,/ylog,/noerase,position=[x1(cc),y1(cc),x2(cc),y2(cc)],$
;     charsize=ccs,xthick=4,ythick=4,charthick=3,xrange=[0.,max(acritb(tmp))/Rjup],$
;     thick=5,/xstyle,linestyle=1,yrange=[10.,ymax],/ystyle

oplot,acritb/Rceres,Rmoon/kmcm,thick=5
oplot,acritc/Rceres,Rmoon/kmcm,thick=5,linestyle=2

xyouts,65.,300.,'R!dsub!n=20km',charsize=0.75,charthick=20,orientation=-15.
xyouts,65.,152.,'R!dsub!n=10km',charsize=0.75,charthick=20,orientation=-15.
xyouts,65.,70.,'R!dsub!n=5km',charsize=0.75,charthick=20,orientation=-15.





END


