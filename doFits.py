import sys
from dipole import getDipoleList2,getDipoleSpectra2,fitTrapIntensity
import pickle
goodQuads = [0,1,2,3]
image_dir = 'proc/'
test_temps = [125, 130, 135, 140, 145, 150, 155, 165, 175, 180, 183, 185, 187,
       190, 193, 195, 197, 200, 203, 207, 210]


try:
    with open('dipole_coord_list.pkl','rb') as infile:
        full_dipole_coord_list = pickle.load(infile)
except FileNotFoundError:
    full_dipole_coord_list = getDipoleList2(image_dir,test_temps,goodQuads)
    with open('dipole_coord_list.pkl','wb') as outfile:
        pickle.dump(full_dipole_coord_list,outfile)
# print(f'# of images at {test_temp}',Counter(temperatures_strs)[str(test_temp)])
try:
    with open('dipole_spectra.pkl','rb') as infile:
        dipole_spectra = pickle.load(infile)
except FileNotFoundError:
    dipole_spectra = getDipoleSpectra2(image_dir,goodQuads,full_dipole_coord_list)
    with open('dipole_spectra.pkl','wb') as outfile:
        pickle.dump(dipole_spectra,outfile)


for useIntensityErr in [True,False]:
    intensity_str = '_err' if useIntensityErr else ''
    fit_dipole_spectra = fitTrapIntensity(dipole_spectra,useIntensityErr=useIntensityErr)
    with open(f'fit_dipole_spectra{intensity_str}.pkl','wb') as outfile:
        pickle.dump(fit_dipole_spectra,outfile)
