import sys
import os
import re
import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, resample
from scipy.fftpack import rfft, irfft, rfftfreq
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import efield_denoising.hdf5lib.hdf5fileinout as hdf5io

#######################################
##Konstant with a K
kc = 2.997924580e8
kepsilon_0 = 8.85418782e-12
kvolt_ev = 1.6e-19
#######################################

def get_ce_pola(E_vec_, ant_vec_):

    #compute the polarization vector of the charge excess emission in SHOWER PLANE!
    _ant_vec_norm = ant_vec_/np.linalg.norm(ant_vec_)
    _Ece = np.dot(E_vec_ ,_ant_vec_norm)
    _pola_vec = -  _ant_vec_norm #* max(np.abs(_Ece))

    return _Ece, _pola_vec

def get_geo_pola(E_vec_, k_, inclination, declination):

    #compute the polarization vector of the geomagnetic emission in SHOWER PLANE!
    #_B = np.array([np.cos(declination)*np.sin(inclination), np.sin(declination)*np.sin(inclination),np.cos(inclination)])
    #_kxB = np.cross(k_,_B)
    #_Egeo = np.dot(E_vec_ ,_kxB)
    _Egeo = E_vec_[:,0]
    #_pola_vec = _kxB * max(_Egeo)
    _pola_vec = np.array([1, 0, 0]) #* max(np.abs(_Egeo))

    return _Egeo, _pola_vec

def get_pola_max(E_vec_):

    _E = np.array([np.linalg.norm(_vec) for _vec in E_vec_])
    _pola_vec = E_vec_[np.where(_E==max(np.abs(_E))),:]

    return _E, _pola_vec

#Real antennes positions to airshower plane positions
def get_in_shower_plane(pos_, k_, core_decay_, inclination_, declination_):

    _pos = (pos_ - core_decay_).T
    _B = np.array([np.cos(declination_)*np.sin(inclination_), np.sin(declination_)*np.sin(inclination_),np.cos(inclination_)])
    _kxB = np.cross(k_,_B)
    _kxB /= np.linalg.norm(_kxB)
    _kxkxB = np.cross(k_,_kxB)
    _kxkxB /= np.linalg.norm(_kxkxB)
    #print("k_", k_, "_kxB = ", _kxB, "_kxkxB = ", _kxkxB)

    return np.array([np.dot(_kxB, _pos), np.dot(_kxkxB, _pos), np.dot(k_, _pos)])

def get_efield_shower(E_, k_, inclination_, declination_):

    _B = np.array([np.cos(declination_)*np.sin(inclination_), np.sin(declination_)*np.sin(inclination_),np.cos(inclination_)])
    _kxB = np.cross(k_,_B)
    _kxB /= np.linalg.norm(_kxB)
    _kxkxB = np.cross(k_,_kxB)
    _kxkxB /= np.linalg.norm(_kxkxB)

    return np.array([np.dot(_kxB,E_), np.dot(_kxkxB,E_), np.dot(k_,E_)])

def get_CE_GeoM_pola(E_vec_, ant_vec_):
    return

def getpeak_time_polarization(path, output_list, ant_flag, plot_flag) :

    peaktime= np.zeros(len(output_list))
    peakamp= np.zeros(len(output_list))
    peakamp_x = np.zeros(len(output_list))
    peaktime_x = np.zeros(len(output_list))
    peakamp_y = np.zeros(len(output_list))
    peaktime_y = np.zeros(len(output_list))
    peakamp_z = np.zeros(len(output_list))
    peaktime_z = np.zeros(len(output_list))
    nbe = np.zeros(len(output_list))

    i = 0
    for output in output_list: # loop over all antennas

        #Read traces or voltages
        if ant_flag == "trace" :
            nbe[i] =  int(output.split('/a')[-1].split('.trace')[0])
            convert_ns = 1.e-9											#traces are in ns
        elif ant_flag == "volt":
            nbe[i] =  int(output.split('out_')[-1].split('.txt')[0])
            convert_ns = 1												#voltages are in s
        elif ant_flag == "volt_noise_filt":
            nbe[i] =  int(output.split('out_')[-1].split('_')[0])
            convert_ns = 1												#voltages are in s
        else :
            print("Error output format unknow, please use : trace / volt / volt_noise_filt")
            sys.exit()

        signal=np.loadtxt(output)																	#simulated signals
        signal.T[0,:] = signal.T[0,:]*convert_ns
        peakamp_x[i] = signal.T[1,np.where(np.abs(signal.T[1,:])==max(np.abs(signal.T[1,:])))[0][0]]		#peak amplitude of x-polarization
        peaktime_x[i] = signal.T[0,np.where(signal.T[1,:] == peakamp_x[i])[0][0]] 					#peak time of x-polarization
        peakamp_y[i] = signal.T[2,np.where(np.abs(signal.T[2,:])==max(np.abs(signal.T[2,:])))[0][0]]		#peak amplitude of y-polarization
        peaktime_y[i] = signal.T[0,np.where(signal.T[2,:] == peakamp_y[i])[0][0]] 					#peak time of y-polarization
        peakamp_z[i] = signal.T[3,np.where(np.abs(signal.T[3,:])==max(np.abs(signal.T[3,:])))[0][0]]		#peak amplitude of z-polarization
        peaktime_z[i] = signal.T[0,np.where(signal.T[3,:] == peakamp_z[i])[0][0]] 					#peak time of z-polarization
        sum_pola = np.sqrt(signal.T[1,:]**2 + signal.T[2,:]**2 + signal.T[3,:]**2)
        peakamp[i] = max(sum_pola)																	#peak amplitude of summed polarization
        peaktime[i]=signal.T[0,np.where(sum_pola == peakamp[i])[0][0]] 							#peak time of summed polarization

        #PLOT
        if plot_flag == 1 :
            if peakamp[i] > 0. :

                plt.plot(signal.T[0], sum_pola, label = '|E|', color='C0')
                plt.scatter(peaktime[i], peakamp[i], color='C0')
                plt.plot(signal.T[0], signal.T[1], label = 'Ex', color='C1')
                plt.scatter(peaktime_x[i], peakamp_x[i], color='C1')
                plt.plot(signal.T[0], signal.T[2], label = 'Ey', color='C2')
                plt.scatter(peaktime_y[i], peakamp_y[i], color='C2')
                plt.plot(signal.T[0], signal.T[3], label = 'Ez', color='C3')
                plt.scatter(peaktime_z[i], peakamp_z[i], color='C3')
                plt.axvline(peaktime[i], color='r', label = 'Timepeak')
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude (muV) (s)")
                plt.legend(loc = 'best')
                plt.show()
                input()

        i+=1
    nbe = list(map(int, nbe))
    return [nbe, peaktime, peakamp, peaktime_x, peakamp_x, peaktime_y, peakamp_y, peaktime_z, peakamp_z]


def Getpeak_time_amp_hilbert(path, output_list, ant_flag, plot_flag) :

    peaktime= np.zeros(len(output_list))
    peakamplitude= np.zeros(len(output_list))
    nbe = np.zeros(len(output_list))
    i = 0
    for output in output_list: # loop over all antennas

        #Read traces or voltages
        if ant_flag == "trace" :
            nbe[i] =  int(output.split('/a')[-1].split('.trace')[0])
            convert_ns = 1.e-9											#traces are in ns
        elif ant_flag == "volt":
            nbe[i] =  int(output.split('out_')[-1].split('.txt')[0])
            convert_ns = 1												#voltages are in s
        elif ant_flag == "volt_noise_filt":
            nbe[i] =  int(output.split('out_')[-1].split('_')[0])
            convert_ns = 1												#voltages are in s
        else :
            print("Error output format unknow, please use : trace / volt / volt_noise_filt")
            sys.exit()

        txt=np.loadtxt(output) 																		#simulated voltages
        hilbert_amp = np.abs(hilbert(txt.T[1:])) 													#enveloppe de hilbert x, y, z channels
        peakamplitude[i]=max([max(hilbert_amp[0,:]), max(hilbert_amp[1,:]), max(hilbert_amp[2,:])]) #find best peakamp for the 3 channels
        peaktime[i]=txt.T[0,np.where(hilbert_amp == peakamplitude[i])[1][0]] *convert_ns 			# get the time of the peak amplitude
        #peaktime[i] = peaktime[i] + np.random.normal(0.0, 5)*1.e-9 									#add std of  5 ns offset

        #PLOT
        if plot_flag == 1 :
            print(peakamplitude[i])
            if peakamplitude[i] > 0. :
                #plt.plot(txt.T[0], hilbert_amp[0,:], label = 'Hilbert env channel x')
                #plt.plot(txt.T[0], hilbert_amp[1,:], label = 'Hilbert env channel y')
                #plt.plot(txt.T[0], hilbert_amp[2,:], label = 'Hilbert env channel z')
                plt.plot(txt.T[0], np.sqrt(txt.T[1]**2 + txt.T[2]**2 + txt.T[3]**2), label = 'Time traces')
                print('peaktime = ',peaktime[i])
                #plt.axvline(peaktime[i], color='r', label = 'Timepeak')
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude (muV) (s)")
                plt.legend(loc = 'best')
                plt.show()
                input()

        i+=1
    nbe = list(map(int, nbe))
    return [nbe, peaktime, peakamplitude]


def Getpeak_time_amp_minmax(path, nbe_antennas) :

    peaktime= np.zeros(nbe_antennas)
    peakamplitude= np.zeros(nbe_antennas)
    for i in range(0, nbe_antennas): # loop over all antennas
        #txt=np.loadtxt(path + '/split/' + 'out_'+str(i)+'.txt') #simulated traces
        txt=np.loadtxt(path + '/' + 'out_'+str(i)+'.txt') #simulated traces
        # NOTE : along y axis only
        amplitude_min = min(txt[2])
        amplitude_max = max(txt[2])
        t_min = txt[0, np.where(txt[2] == amplitude_min)]
        t_max = txt[0, np.where(txt[2] == amplitude_max)]
        peakamplitude[i]=amplitude_max
        peaktime[i]= t_min[0][0] + np.abs(t_min[0][0] - t_max[0][0]) /2.

    return [peaktime, peakamplitude]


def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """subfunction of filt
    """
    b, a = butter(5, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')  # (order, [low, high], btype)

    return lfilter(b, a, data)


def filters(voltages, FREQMIN=50.e6, FREQMAX=200.e6):
  """ Filter signal v(t) in given bandwidth
  Parameters
  ----------
   : voltages
      The array of time (s) + voltage (muV) vectors to be filtered
   : FREQMIN
      The minimal frequency of the bandpass filter (Hz)
   : FREQMAX:
      The maximal frequency of the bandpass filter (Hz)


  Raises
  ------
  Notes
  -----
  At present Butterworth filter only is implemented
  Examples
  --------
  ```
  >>> from signal_treatment import _butter_bandpass_filter
  ```
  """

  t = voltages[:,0]
  v = np.array(voltages[:, 1:])  # Raw signal

  fs = 1 / np.mean(np.diff(t))  # Compute frequency step
  #print("Trace sampling frequency: ",fs/1e6,"MHz")
  nCh = np.shape(v)[1]
  vout = np.zeros(shape=(len(t), nCh))
  res = t
  for i in range(nCh):
        vi = v[:, i]
        #vout[:, i] = _butter_bandpass_filter(vi, FREQMIN, FREQMAX, fs)
        res = np.append(res,_butter_bandpass_filter(vi, FREQMIN, FREQMAX, fs))

  res = np.reshape(res,(nCh+1,len(t)))  # Put it back inright format
  return res

def get_filtered_peakAmpTime_Hilbert(InputFilename_, EventName_, AntennaInfo_, f_min_, f_max_):
    # get the path to directory containing InputFilename_
    _path = os.path.dirname(InputFilename_)

    _NumberOfAntennas = hdf5io.GetNumberOfAntennas(AntennaInfo_)
    _peakamp, _peaktime = np.zeros(_NumberOfAntennas), np.zeros(_NumberOfAntennas)
    i = 0
    for ant_id in hdf5io.GetAntIDFromAntennaInfo(AntennaInfo_):

        _Efield_trace = hdf5io.GetAntennaEfield(InputFilename_,EventName_,ant_id,OutputFormat="numpy")
        _Efield_trace[:,0] *= 1.e-9 #from ns to s

        _Efield_filt = filters(_Efield_trace, FREQMIN=f_min_, FREQMAX=f_max_)

        _hilbert_amp = np.abs(hilbert(_Efield_filt[1:4,:]))
        _peakamp[i]=max([max(_hilbert_amp[0,:]), max(_hilbert_amp[1,:]), max(_hilbert_amp[2,:])])
        _peaktime[i]=_Efield_filt[0,np.where(_hilbert_amp == _peakamp[i])[1][0]]

        # _Emodulus = np.sqrt(_Efield_filt[1,:]**2 + _Efield_filt[2,:]**2 + _Efield_filt[3,:]**2)
        # _peakamp[i]=max(_Emodulus)
        # _peaktime[i]=_Efield_filt[0,np.where(_Emodulus == _peakamp[i])[0][0]]
        
        label_amp = r'Peak Amplitude = ' + str(round(_peakamp[i],2)) + r' $\rm \mu V/m$'
        label_time = r'Peak Time = %.2e ns' % (_peaktime[i]*1.e9)
        g, gx = plt.subplots()
        gx.plot(_Efield_filt.T[:,0]*1e9, _Efield_filt.T[:,1], label='X', color='C0', lw=2.0)
        #gx.axhline(y=max(_hilbert_amp[0,:]), linestyle='--', label='X', color='C0')
        gx.plot(_Efield_filt.T[:,0]*1e9, _Efield_filt.T[:,2], label='Y', color='C1',lw=2.0)
        #gx.axhline(y=max(_hilbert_amp[1,:]), linestyle='--', label='Y', color='C1')
        gx.plot(_Efield_filt.T[:,0]*1e9, _Efield_filt.T[:,3], label='Z', color='C2',lw=2.0)
        #gx.axhline(y=max(_hilbert_amp[2,:]), linestyle='--', label='Z', color='C2')
        gx.axvline(x=_peaktime[i]*1e9, linestyle='--', label=label_time, color='r', lw=1.0)
        gx.axhline(y=_peakamp[i], linestyle='--', label=label_amp, color='r', lw=1.0)
        gx.legend()
        gx.set_xlabel(r'Time (ns)')
        gx.set_ylabel(r'Amplitude ($\rm \mu V/m$)')
        plt.savefig(_path + '/ant_' + str(ant_id) + '_filtered.png')
        plt.close()

        i+=1

    return _peaktime, _peakamp


def get_filtered_fluence(InputFilename_, EventName_, AntennaInfo_, f_min_, f_max_):

    _NumberOfAntennas = hdf5io.GetNumberOfAntennas(AntennaInfo_)
    _fluence_x, _fluence_y, _fluence_z = np.zeros(_NumberOfAntennas), np.zeros(_NumberOfAntennas), np.zeros(_NumberOfAntennas)
    i = 0
    for ant_id in hdf5io.GetAntIDFromAntennaInfo(AntennaInfo_):

        _Efield_trace = hdf5io.GetAntennaEfield(InputFilename_,EventName_,ant_id,OutputFormat="numpy")
        _Efield_trace[:,0] *= 1.e-9 #from ns to s

        _Efield_filt = filters(_Efield_trace, FREQMIN=f_min_, FREQMAX=f_max_)

        _sampling = np.mean(np.diff(_Efield_filt[0,:]))
        _fluence_x[i] = np.sum(_Efield_filt[1,:]**2)*kepsilon_0*kc*_sampling*(1.e-6/kvolt_ev)
        _fluence_y[i] = np.sum(_Efield_filt[2,:]**2)*kepsilon_0*kc*_sampling*(1.e-6/kvolt_ev)
        _fluence_z[i] = np.sum(_Efield_filt[3,:]**2)*kepsilon_0*kc*_sampling*(1.e-6/kvolt_ev)


        # g, gx = plt.subplots()
        # gx.set_xlabel(r"$\rm time\ (ns)$")
        # gx.set_xlabel(r"$\rm Amplitude\ (\mu V/m)$")
        # gx.plot(_Efield_filt.T[:,0], _Efield_filt.T[:,1], label='X', color='C0')
        # #gx.axhline(y=_fluence_x, linestyle='--', label='X', color='C0')
        # gx.plot(_Efield_filt.T[:,0], _Efield_filt.T[:,2], label='Y', color='C1')
        # #gx.axhline(y=_fluence_y, linestyle='--', label='Y', color='C1')
        # gx.plot(_Efield_filt.T[:,0], _Efield_filt.T[:,3], label='Z', color='C2')
        # #gx.axhline(y=_fluence_z, linestyle='--', label='Z', color='C2')
        # gx.legend()
        # plt.show()

        i+=1

    return _fluence_x, _fluence_y, _fluence_z


def GetRefractionIndexAtXmax(x0, y0, z0, ns,kr):
    rearth=6370949.0
    R02=x0*x0+y0*y0  #notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada
    h0=(np.sqrt((z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.E3 #altitude of emission, in km

    rh0 = ns*np.exp(kr*h0) #refractivity at emission
    n_h0=1.E0+1.E-6*rh0 #n at
    return n_h0
