
  fd3 v.3.1 (Florianapolis, 25 July 2014)

  SECTION 1: LOADING OBSERVED SPECTRA

  observed spectra master file = resampled_spectra.dat
  loaded 9 spectra with 125999 bins per spectrum
  lnlambda range from 8.33806695 to 8.80985824
  rv step 1.12258 km/s per bin
  use data from lnlambda = 8.4
             to lnlambda = 8.43
  selected 8011 bins per spectrum

  SECTION 2: OUTPUT FILES

  output files root name = temp.obs

  used part of observed spectra ------------> "temp.obs.obs"
  model spectra ----------------------------> "temp.obs.mod"
  residuals (obs. - calc.) -----------------> "temp.obs.res"
  radial velocities (in bin units) ---------> "temp.obs.rvs"
  optimisation log -------------------------> "temp.obs.log"

  SECTION 3: COMPONENT SPECTRA MODELING SWITCHES (0/1)

  A = 1
  B = 1
  C = 0

  number of components to be resolved is 2

  SECTION 4: DESCRIPTORS TO OBSERVED SPECTRA
             AND LIGHT-FACTORS ASSIGNED TO COMPONENTS

           t_obs     rv_corr   noise_rms      lf_A      lf_B
           [day]      [km/s]         [1]       [1]       [1]

     59689.62207      0.0000      0.5000    1.0000    1.0000
     59690.62698      0.0000      0.5000    1.0000    1.0000
     59696.52644      0.0000      0.5000    1.0000    1.0000
     59697.45133      0.0000      0.5000    1.0000    1.0000
     59721.47611      0.0000      0.5000    1.0000    1.0000
     59731.41817      0.0000      0.5000    1.0000    1.0000
     59736.50107      0.0000      0.5000    1.0000    1.0000
     59757.43809      0.0000      0.5000    1.0000    1.0000
     59758.43028      0.0000      0.5000    1.0000    1.0000

  SECTION 5: PARAMETERS OF ORBITS

  wide (AB--C) orbit

  period of AB--C [day] = 1 +/- 0
  time of periast pass in AB--C [day] = 0 +/- 0
  eccentricity of AB--C [1] = 0 +/- 0
  periast long of AB in AB--C [deg] = 0 +/- 0
  a sin(incl) of AB in AB--C [light-day] = 0 +/- 0
  a sin(incl) of C in AB--C [light-day] = 0 +/- 0

  tight (A--B) orbit

  period A--B [day] = 2.74 +/- 0
  time of periast pass in A--B [day] = 43563.5 +/- 0.0001 *** free ***
  eccentricity of A--B [1] = 0.001 +/- 0.001 *** free ***
  periast long of A in A--B [deg] = 60 +/- 5 *** free ***
  rv semiamp of A in A--B [km/s] = 10 +/- 5 *** free ***
  rv semiamp of B in A--B [km/s] = 100 +/- 5 *** free ***
  periast long adv per cycle in A--B [deg] = 0 +/- 0

  number of free orbital parameters is 5

  SECTION 6: OPTIMISATION DETAILS

  number of independent optimisation runs = 30
  number of allowed iterations per run = 400
  stop when simplex shrinked by factor = 1e-05

  SECTION 7: OPTIMISATION

  separation at the starting point:  chi2=88.7797  gof=1.00
  converged disentangling runs reported only if chi2 decreases

  irun=3  iter=289  sxshrnkf=7.7e-06  chi2=38.9408  gof=1.00
              time of periast pass in A--B = 43563.5002 day
                      eccentricity of A--B = 0.00334326144
                 periast long of A in A--B = 60.7763597 deg
                   rv semiamp of A in A--B = 13.6873498 km/s
                   rv semiamp of B in A--B = 78.0483214 km/s

  irun=8  iter=313  sxshrnkf=5.1e-06  chi2=38.2937  gof=1.00
              time of periast pass in A--B = 43563.5001 day
                      eccentricity of A--B = 0.00317759091
                 periast long of A in A--B = 60.4792772 deg
                   rv semiamp of A in A--B = 0.0928618331 km/s
                   rv semiamp of B in A--B = 77.6794042 km/s

  completed 30 optimisation runs

  EXITING REGULARLY

