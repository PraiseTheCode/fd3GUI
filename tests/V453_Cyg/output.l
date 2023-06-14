
  fd3 v.3.1 (Florianapolis, 25 July 2014)

  SECTION 1: LOADING OBSERVED SPECTRA

  observed spectra master file = V453_Cyg.master.obs
  loaded 8 spectra with 1344 bins per spectrum
  lnlambda range from 8.36567238 to 8.4104673
  rv step 9.99981 km/s per bin
  use data from lnlambda = 8.36
             to lnlambda = 8.41
  selected 1329 bins per spectrum

  SECTION 2: OUTPUT FILES

  output files root name = V453_Cyg

  used part of observed spectra ------------> "V453_Cyg.obs"
  model spectra ----------------------------> "V453_Cyg.mod"
  residuals (obs. - calc.) -----------------> "V453_Cyg.res"
  radial velocities (in bin units) ---------> "V453_Cyg.rvs"
  optimisation log -------------------------> "V453_Cyg.log"

  SECTION 3: COMPONENT SPECTRA MODELING SWITCHES (0/1)

  A = 1
  B = 1
  C = 0

  number of components to be resolved is 2

  SECTION 4: DESCRIPTORS TO OBSERVED SPECTRA
             AND LIGHT-FACTORS ASSIGNED TO COMPONENTS

           t_obs     rv_corr   noise_rms      lf_A      lf_B
           [day]      [km/s]         [1]       [1]       [1]

         2.48002      0.0000      0.5000    1.0000    1.0000
         6.53181      0.0000      0.5000    1.0000    1.0000
         3.42439      0.0000      0.5000    1.0000    1.0000
         7.65200      0.0000      0.5000    1.0000    1.0000
         4.47209      0.0000      0.5000    1.0000    1.0000
         8.47470      0.0000      0.5000    1.0000    1.0000
         8.61560      0.0000      0.5000    1.0000    1.0000
         1.48987      0.0000      0.5000    1.0000    1.0000

  SECTION 5: PARAMETERS OF ORBITS

  wide (AB--C) orbit

  period of AB--C [day] = 1 +/- 0
  time of periast pass in AB--C [day] = 0 +/- 0
  eccentricity of AB--C [1] = 0 +/- 0
  periast long of AB in AB--C [deg] = 0 +/- 0
  a sin(incl) of AB in AB--C [light-day] = 0 +/- 0
  a sin(incl) of C in AB--C [light-day] = 0 +/- 0

  tight (A--B) orbit

  period A--B [day] = 3.8898128 +/- 0
  time of periast pass in A--B [day] = 0 +/- 0.5 *** free ***
  eccentricity of A--B [1] = 0.2 +/- 0.05 *** free ***
  periast long of A in A--B [deg] = 270 +/- 10 *** free ***
  rv semiamp of A in A--B [km/s] = 170 +/- 5 *** free ***
  rv semiamp of B in A--B [km/s] = 220 +/- 5 *** free ***
  periast long adv per cycle in A--B [deg] = 0 +/- 0

  number of free orbital parameters is 5

  SECTION 6: OPTIMISATION DETAILS

  number of independent optimisation runs = 50
  number of allowed iterations per run = 500
  stop when simplex shrinked by factor = 1e-05

  SECTION 7: OPTIMISATION

  separation at the starting point:  chi2=6.7451  gof=1.00
  converged disentangling runs reported only if chi2 decreases

  irun=1  iter=461  sxshrnkf=8.7e-06  chi2=1.32331  gof=1.00
              time of periast pass in A--B = -0.85527713 day
                      eccentricity of A--B = 2.94521055e-11
                 periast long of A in A--B = 202.401154 deg
                   rv semiamp of A in A--B = 174.035916 km/s
                   rv semiamp of B in A--B = 217.062131 km/s

  irun=4  iter=209  sxshrnkf=8.3e-06  chi2=1.32325  gof=1.00
              time of periast pass in A--B = -0.323124382 day
                      eccentricity of A--B = 0.00158856551
                 periast long of A in A--B = 251.64628 deg
                   rv semiamp of A in A--B = 174.037067 km/s
                   rv semiamp of B in A--B = 217.325034 km/s

  irun=7  iter=293  sxshrnkf=7.7e-06  chi2=1.32167  gof=1.00
              time of periast pass in A--B = -0.141575014 day
                      eccentricity of A--B = 0.00522099116
                 periast long of A in A--B = 268.437851 deg
                   rv semiamp of A in A--B = 173.965631 km/s
                   rv semiamp of B in A--B = 218.184623 km/s

  irun=15  iter=309  sxshrnkf=9.1e-06  chi2=1.31973  gof=1.00
              time of periast pass in A--B = 0.0634664405 day
                      eccentricity of A--B = 0.00606560607
                 periast long of A in A--B = 287.385711 deg
                   rv semiamp of A in A--B = 173.844606 km/s
                   rv semiamp of B in A--B = 218.623823 km/s

  irun=17  iter=215  sxshrnkf=9e-06  chi2=1.31919  gof=1.00
              time of periast pass in A--B = 0.127437302 day
                      eccentricity of A--B = 0.00599578745
                 periast long of A in A--B = 293.300067 deg
                   rv semiamp of A in A--B = 173.826292 km/s
                   rv semiamp of B in A--B = 218.596241 km/s

  irun=24  iter=302  sxshrnkf=7.8e-06  chi2=1.31702  gof=1.00
              time of periast pass in A--B = 0.418174164 day
                      eccentricity of A--B = 0.00659486304
                 periast long of A in A--B = 320.173293 deg
                   rv semiamp of A in A--B = 173.77114 km/s
                   rv semiamp of B in A--B = 218.463859 km/s

  irun=35  iter=485  sxshrnkf=9.8e-06  chi2=1.30626  gof=1.00
              time of periast pass in A--B = 1.18778683 day
                      eccentricity of A--B = 0.0295717404
                 periast long of A in A--B = 391.665089 deg
                   rv semiamp of A in A--B = 173.566593 km/s
                   rv semiamp of B in A--B = 217.338903 km/s

  irun=47  iter=369  sxshrnkf=7.4e-06  chi2=1.30626  gof=1.00
              time of periast pass in A--B = 1.18778762 day
                      eccentricity of A--B = 0.0295717195
                 periast long of A in A--B = 391.665164 deg
                   rv semiamp of A in A--B = 173.566579 km/s
                   rv semiamp of B in A--B = 217.338917 km/s

  completed 50 optimisation runs

  EXITING REGULARLY

