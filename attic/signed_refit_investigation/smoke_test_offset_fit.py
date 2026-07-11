import numpy as np
from dipole_new import fitTrapIntensity, intensity_function_offset

rng = np.random.default_rng(7)
seconds = np.array([750, 1200, 2000, 3000, 5000, 8000, 18000, 28000, 45000, 70000,
                    100000, 170000, 260000, 400000, 650000, 1000000, 1500000, 2500000]) / 15e6


def make(coeff, tau, off, sig):
    I = intensity_function_offset(seconds, coeff, tau, off) + rng.normal(0, sig, seconds.size)
    return {'seconds': seconds, 'intensities': I, 'intensity_err': np.full(seconds.size, sig),
            'image_sigma': 200.0, 'dtphs': (seconds * 15e6).astype(int),
            'poisson_err': np.full(seconds.size, sig), 'patch_sigma': np.full(seconds.size, sig * 2.5)}


taus = [0.5, 0.05, 0.005, 5e-4, 5e-5]
temps = [140, 150, 160, 170, 180]
offs = [0, -50, -200, -700, -700]

d = {0: {(100, 50): {}}}
for T, tau, off in zip(temps, taus, offs):
    d[0][(100, 50)][T] = make(+0.4, tau, off, 35.0)
d[0][(200, 60)] = {T: make(0.005, tau, off, 35.0) for T, tau, off in zip(temps, taus, offs)}

fitTrapIntensity(d, useIntensityErr=True, wellBehavedThreshold=4, fit_offset=True)

t1 = d[0][(100, 50)]
print('bright trap good temps:', [T for T in temps if t1[T]['GoodIntensityFit']])
print('true taus:           ', taus)
print('fitted taus:', {T: float(round(t1[T]['fit_tau'], 6)) for T in temps if t1[T]['GoodIntensityFit']})
print('fitted offsets:', {T: float(round(t1[T].get('fit_offset', np.nan), 1)) for T in temps})
print('WellBehaved:', t1['WellBehavedTrap'], 'EnergyFitFailed:', t1.get('EnergyFitFailed'),
      'GoodEnergyFit:', t1.get('GoodEnergyFit'))
t2 = d[0][(200, 60)]
print('faint trap good temps:', [T for T in temps if t2[T]['GoodIntensityFit']], '(expect none)')
print('faint significance e.g.:', float(round(t2[140].get('amplitude_significance', -1), 2)))
