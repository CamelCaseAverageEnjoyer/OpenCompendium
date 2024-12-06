# Near earth coordinates

The module contains two functions: translation of Cartesian 
coordinates and velocities into orbital elements and vice versa.

Use the following way to use the functions:

    _import near_earth_coordinates as nec_

    _a,e,i,Om,w,nu = nec.cartesian2keplerian(r,v,mu)_
    _r,v = nec.keplerian2cartesian(a,e,i,Om,w,nu,mu)_

