"""
Core physics functions for STOPX energy loss calculations.

Functions:
    _elspwr: Electronic stopping power calculation
    _spwrhi: Heavy ion stopping power calculation  
    _nuspwr: Nuclear stopping power calculation
    _stopp: Total stopping power calculation
    _range: Range calculation using 4th-order Runge-Kutta
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def _elspwr(energy_kev, izt, zt, at, ioneff, c2s, a0s, a1s, a2s, a3s, a4s, 
           tmcc2, pi, e4, isw4):
    """
    Calculate electronic stopping power using the bimodal model.
    
    This is an exact translation of the Fortran ELSPWR function, implementing:
    - Low energy regime (E/A < 1000 keV/u): Varelas-Biersack formula
    - High energy regime (E/A >= 1000 keV/u): Bethe formula with shell corrections
    - Smooth blending between regimes using harmonic sum
    
    Args:
        energy_kev: Energy per nucleon in keV/u
        izt: Target atomic number
        zt: Target atomic number (float)
        at: Target mass number
        ioneff: Effective ionization potential
        c2s: C2S parameter array
        a0s-a4s: Shell correction coefficient arrays
        tmcc2: 2*mc^2 constant
        pi: Pi constant
        e4: e^4 constant
        isw4: Control flag for shell corrections
    
    Returns:
        Electronic stopping power in MeV/(mg/cm^2)
    """
    # Initialize parameters (equivalent to Fortran lines 1254-1266)
    c2 = c2s[izt - 1]
    indx = (izt + 15) // 10 - 1  # Convert to 0-based indexing
    a0 = -a0s[indx]
    a1 = a1s[indx]
    a2 = -a2s[indx]
    a3 = a3s[indx]
    a4 = -a4s[indx] * 1.0e-04
    
    # Conversion factor to MeV/(mg/cm^2)
    cfact = 0.6023 / at
    
    # Low energy regime (E/A < 1000 keV/u)
    if energy_kev < 1000.0:
        slow = c2 * (energy_kev ** 0.45)
        
        if energy_kev < 10.0:
            return slow * cfact
        else:
            # High energy form for blending
            shigh = (243.0 - 0.375 * zt) * zt / energy_kev * np.log(
                1.0 + 500.0 / energy_kev + 2.195e-06 * energy_kev / ioneff)
            
            # Harmonic blending (exact Fortran formula)
            return slow * shigh / (slow + shigh) * cfact
    
    # High energy regime (E/A >= 1000 keV/u) - Bethe formula
    # Shell correction calculation
    czt = 0.0
    if not isw4 and energy_kev <= 40000.0:
        ale = np.log(energy_kev)
        czt = a0 + (a1 + (a2 + (a3 + a4 * ale) * ale) * ale) * ale
    
    # Beta squared calculation
    beta2 = 1.0 - 1.0 / (1.0 + energy_kev / 931189.0) ** 2
    
    # Bethe stopping power formula
    spwr = (8.0 * pi * e4 * zt / (beta2 * tmcc2) * 
            (np.log(beta2 * tmcc2 / ((1.0 - beta2) * ioneff * 1.0e-06)) - beta2 - czt))
    
    # Effective charge for hydrogen
    ex = 0.2 * np.sqrt(energy_kev) + 0.0012 * energy_kev + 1.443e-05 * energy_kev * energy_kev
    zh = 1.0
    if ex < 20.0:
        zh = 1.0 - np.exp(-ex)
    
    # Target-dependent correction for effective charge
    zfact = 1.0
    if not isw4 and energy_kev <= 1999.0:
        b = 1.0
        if izt < 35:
            b = (zt - 1.0) / 34.0
        zfact = 1.0 + b * (0.1097 - 5.561e-05 * energy_kev)
    
    # Final stopping power
    return spwr * zh * zh * cfact * zfact


@jit(nopython=True)
def _spwrhi(energy_kev, izp, zp, zt, spwrh):
    """
    Calculate heavy ion stopping power by scaling proton stopping power.
    
    This is an exact translation of the Fortran SPWRHI function, which scales
    the cached proton stopping power using effective charge formulas for
    different ion types.
    
    Args:
        energy_kev: Energy per nucleon in keV/u
        izp: Projectile atomic number
        zp: Projectile atomic number (float)
        zt: Target atomic number (float)
        spwrh: Cached proton stopping power
    
    Returns:
        Heavy ion stopping power in MeV/(mg/cm^2)
    """
    spwrhi = spwrh
    
    if izp < 2:
        return spwrhi
    
    if izp <= 3:
        x = np.log(energy_kev)
        ex = (7.6 - x) ** 2
        gamma = 1.0
        if ex < 20.0:
            gamma = 1.0 + (0.007 + 0.00005 * zt) * np.exp(-ex)
        
        if izp == 2:  # Helium
            ex = 0.7446 + 0.1429 * x + 0.01562 * x * x - 0.00267 * x ** 3 + 1.325e-06 * x ** 8
            zhezh = 2.0 * gamma
            if ex < 20.0:
                zhezh = zhezh * (1.0 - np.exp(-ex))
            spwrhi = spwrh * zhezh * zhezh
            
        elif izp == 3:  # Lithium
            zlizh = 3.0 * gamma
            ex = 0.7138 + 0.002797 * energy_kev + 1.348e-06 * energy_kev * energy_kev
            if ex < 20.0:
                zlizh = zlizh * (1.0 - np.exp(-ex))
            spwrhi = spwrh * zlizh * zlizh
    else:
        # Heavy ions (Z > 3)
        b = 0.886 / 5.0 * np.sqrt(energy_kev) * (zp ** (-0.666666))
        a = b + 0.0378 * np.sin(1.5708 * b)
        zhizh = zp
        if a < 20.0:
            zhizh = zp * (1.0 - np.exp(-a) * (1.034 - 0.1777 * np.exp(-0.08114 * zp)))
        spwrhi = spwrh * zhizh * zhizh
    
    return spwrhi


@jit(nopython=True)
def _nuspwr(energy_kev, zt, zp, at, ap, cfact):
    """
    Calculate nuclear stopping power using Coulomb scattering.
    
    This is an exact translation of the Fortran NUSPWR function, which
    calculates the nuclear contribution to stopping power.
    
    Args:
        energy_kev: Energy per nucleon in keV/u
        zt: Target atomic number
        zp: Projectile atomic number
        at: Target mass number
        ap: Projectile mass number
        cfact: Conversion factor
    
    Returns:
        Nuclear stopping power in MeV/(mg/cm^2)
    """
    az = (at + ap) * np.sqrt(zt ** 0.666666 + zp ** 0.666666)
    rede = 32.53 * at * ap * energy_kev / (zt * zp * az)
    sn = 0.5 * np.log(1.0 + rede) / (rede + 0.10718 * (rede ** 0.37544))
    nuspwr = sn * (8.462 * zt * zp * ap) * cfact / az
    
    return nuspwr


@jit(nopython=True)
def _stopp(energy_kev, izt, izp, zt, zp, at, ap, ioneff, c2s, a0s, a1s, a2s, a3s, a4s,
          tmcc2, pi, e4, isw4, cache_isw1, cache_isw2, cache_spwrh):
    """
    Calculate total stopping power (electronic + nuclear).
    
    This is an exact translation of the Fortran STOPP function, which serves
    as the main dispatcher for stopping power calculations.
    
    Args:
        energy_kev: Energy per nucleon in keV/u
        izt: Target atomic number
        izp: Projectile atomic number
        zt: Target atomic number (float)
        zp: Projectile atomic number (float)
        at: Target mass number
        ap: Projectile mass number
        ioneff: Effective ionization potential
        c2s: C2S parameter array
        a0s-a4s: Shell correction coefficient arrays
        tmcc2: 2*mc^2 constant
        pi: Pi constant
        e4: e^4 constant
        isw4: Control flag for shell corrections
        cache_isw1, cache_isw2: Cache control variables
        cache_spwrh: Cached proton stopping power
    
    Returns:
        Total stopping power in MeV/(mg/cm^2)
    """
    ie = int(energy_kev)
    
    # Check cache for electronic stopping power (exact Fortran logic)
    if cache_isw1 == izt and cache_isw2 == ie:
        # Use cached value
        spwrh = cache_spwrh
    else:
        # Calculate electronic stopping power for proton
        spwrh = _elspwr(energy_kev, izt, zt, at, ioneff, c2s, a0s, a1s, a2s, a3s, a4s,
                       tmcc2, pi, e4, isw4)
    
    # Calculate total stopping power
    if izp > 1:
        # Heavy ion - scale proton stopping power
        stopp = _spwrhi(energy_kev, izp, zp, zt, spwrh)
    else:
        # Proton
        stopp = spwrh
    
    # Add nuclear stopping power for low energies (exact Fortran logic)
    if not isw4 and energy_kev <= 2000.0:
        cfact = 0.6023 / at
        stopp += _nuspwr(energy_kev, zt, zp, at, ap, cfact)
    
    return stopp


@jit(nopython=True)
def _range(energy_kev, step_size, izt, izp, zt, zp, at, ap, ioneff, c2s, a0s, a1s, a2s, a3s, a4s,
          tmcc2, pi, e4, isw4, emin, range_r0, range_e0):
    """
    Calculate particle range using 4th-order Runge-Kutta integration.
    
    This is an exact translation of the Fortran RANGE function, implementing
    the same numerical integration algorithm with identical step size logic
    and the minimum step count rule.
    
    Args:
        energy_kev: Energy per nucleon in keV/u
        step_size: Integration step size in keV/u
        izt: Target atomic number
        izp: Projectile atomic number
        zt: Target atomic number (float)
        zp: Projectile atomic number (float)
        at: Target mass number
        ap: Projectile mass number
        ioneff: Effective ionization potential
        c2s: C2S parameter array
        a0s-a4s: Shell correction coefficient arrays
        tmcc2: 2*mc^2 constant
        pi: Pi constant
        e4: e^4 constant
        isw4: Control flag for shell corrections
        emin: Minimum energy for reliable calculations
        range_r0: Previous range value (for stateful integration)
        range_e0: Previous energy value (for stateful integration)
    
    Returns:
        Range in mg/cm^2
    """
    r = range_r0
    
    # Check if this is the first call or if energy is below minimum
    if range_r0 == 0.0:
        if energy_kev > emin:
            # Simple estimate for first call
            r = (energy_kev / 2.0) / _stopp(energy_kev, izt, izp, zt, zp, at, ap, ioneff,
                                          c2s, a0s, a1s, a2s, a3s, a4s, tmcc2, pi, e4, isw4,
                                          0, 0, 0.0)
        else:
            # Extrapolate below minimum energy
            r = energy_kev / (2.0 * _stopp(emin, izt, izp, zt, zp, at, ap, ioneff,
                                         c2s, a0s, a1s, a2s, a3s, a4s, tmcc2, pi, e4, isw4,
                                         0, 0, 0.0))
    else:
        if energy_kev < emin:
            # Extrapolate below minimum energy
            r = energy_kev / (2.0 * _stopp(emin, izt, izp, zt, zp, at, ap, ioneff,
                                         c2s, a0s, a1s, a2s, a3s, a4s, tmcc2, pi, e4, isw4,
                                         0, 0, 0.0))
        else:
            # Runge-Kutta integration
            n = int((energy_kev - range_e0) / step_size)
            if n < 4:
                n = 4  # Minimum step count rule (exact Fortran logic)
            
            w = (energy_kev - range_e0) / n
            
            # Initialize with previous energy
            e_current = range_e0
            k1 = 1.0 / _stopp(e_current, izt, izp, zt, zp, at, ap, ioneff,
                            c2s, a0s, a1s, a2s, a3s, a4s, tmcc2, pi, e4, isw4,
                            0, 0, 0.0)
            
            # Runge-Kutta integration loop
            for i in range(n):
                k2 = 1.0 / _stopp(e_current + w / 2.0, izt, izp, zt, zp, at, ap, ioneff,
                                c2s, a0s, a1s, a2s, a3s, a4s, tmcc2, pi, e4, isw4,
                                0, 0, 0.0)
                k4 = 1.0 / _stopp(e_current + w, izt, izp, zt, zp, at, ap, ioneff,
                                c2s, a0s, a1s, a2s, a3s, a4s, tmcc2, pi, e4, isw4,
                                0, 0, 0.0)
                
                # 4th-order Runge-Kutta step
                r += (k1 + 4.0 * k2 + k4) * w / 6.0
                
                # Update for next iteration
                k1 = k4
                e_current += w
    
    return r
