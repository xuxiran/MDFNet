import torch
import config as cfg
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import butter, filtfilt

KUL_2D = [[-1 ,-1 ,-1 ,-1 , 1 ,33 ,34 ,-1, -1 ,-1 ,-1],
         [-1 ,-1, -1 , 2  ,3, 37 ,36, 35, -1 ,-1 ,-1],
         [-1, 7 , 6 , 5 , 4 ,38 ,39 ,40 ,41 ,42 ,-1],
         [-1 , 8 , 9 ,10 ,11 ,47 ,46 ,45 ,44 ,43 ,-1],
         [-1 ,15, 14 ,13 ,12 ,48 ,49 ,50 ,51 ,52, -1],
         [-1, 16 ,17 ,18, 19, 32 ,56 ,55 ,54, 53, -1],
         [24 ,23 ,22 ,21 ,20, 31 ,57 ,58 ,59 ,60 ,61],
         [-1, -1 ,-1, 25, 26 ,30 ,63 ,62 ,-1, -1, -1],
         [-1, -1 ,-1, -1 ,27 ,29 ,64, -1, -1 ,-1 ,-1],
         [-1 ,-1 ,-1 ,-1 ,-1 ,28 ,-1 ,-1,-1 ,-1 ,-1]]

DTU_2D = [[-1 ,-1 ,-1 ,-1 , 1 ,33 ,34 ,-1, -1 ,-1 ,-1],
          [-1 ,-1, -1 , 2  ,3, 37 ,36, 35, -1 ,-1 ,-1],
          [-1, 7 , 6 , 5 , 4 ,38 ,39 ,40 ,41 ,42 ,-1],
          [-1 , 8 , 9 ,10 ,11 ,47 ,46 ,45 ,44 ,43 ,-1],
          [-1 ,15, 14 ,13 ,12 ,48 ,49 ,50 ,51 ,52, -1],
          [-1, 16 ,17 ,18, 19, 32 ,56 ,55 ,54, 53, -1],
          [24 ,23 ,22 ,21 ,20, 31 ,57 ,58 ,59 ,60 ,61],
          [-1, -1 ,-1, 25, 26 ,30 ,63 ,62 ,-1, -1, -1],
          [-1, -1 ,-1, -1 ,27 ,29 ,64, -1, -1 ,-1 ,-1],
          [-1 ,-1 ,-1 ,-1 ,-1 ,28 ,-1 ,-1,-1 ,-1 ,-1]]



def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_psd_de(data, window, fs, f_bands=None):
    """
    compute  DE (differential entropy) and PSD (power spectral density) features

    input:
    data-[n, m] n channels, m points of each time course,
    window-integer, window lens of each segment in seconds, such as 1s
    fs-integer, frequency of singal sampling rate, such as 200Hz
    optional  f_bands, default delta, theta, aplha, beta, gamma

    output:
        psd,de  [bands, channels, samples]
    """
    # segment the data
    channels, lens = np.shape(data)
    segment_lens = int(window * fs)
    samples = lens // segment_lens
    data = data[:, :samples*segment_lens]
    data = data.reshape(channels, samples, -1)

    if f_bands == None:
        f_bands = [(1,4), (4,7), (8,13), (14,29), (30, 47)] # delta, theta, aplha, beta, gamma

    # compute the magnitudes
    fxx = np.fft.fft(data)
    timestep = 1 / fs
    f = np.fft.fftfreq(segment_lens, timestep)[:segment_lens//2]  # only use the positive frequency
    fxx = np.abs(fxx[:,:,:segment_lens//2])

    psd_bands = []
    de_bands = []
    for f_band1, f_band2 in f_bands:
        f_mask = (f >= f_band1) & (f <= f_band2)
        data_bands = fxx[:, :, f_mask]

        # psd = np.sum(data_bands**2 / (segment_lens//2), axis=-1)  # same with scipy.signal.periodogram * fs, divide the number of total frequency bands like 100
        psd = np.mean(data_bands**2, axis=-1)  # only divide the number of frequency band1-band2 like 1-4, maybe 4 points with window==1s or 7 points with window==2s
        de = np.log2(2*np.pi*np.exp(1)*np.maximum(data_bands.var(axis=-1), np.finfo(np.float64).tiny)) / 2

        psd_bands.append(psd)
        de_bands.append(de)
    psd = np.stack(psd_bands)
    de = np.stack(de_bands)
    return psd, de


class AADdataset(Dataset):
    def __init__(self, eeg, label,sample_rate,decision_window,dataset_name,csp_pipe=None):
        self.eeg = np.asarray(eeg, dtype=np.float64)
        self.label = label

        self.trnum = self.eeg.shape[0]
        self.segnum = self.eeg.shape[1]//decision_window
        self.sample_rate = sample_rate
        self.decision_window = decision_window
        usable = self.segnum * decision_window
        if self.segnum < 1:
            raise ValueError("Trial shorter than the decision window")

        self.dataset_name = dataset_name

        self.eeg_t1d = self.eeg[:, :usable].reshape(self.trnum, self.segnum, self.decision_window, self.eeg.shape[2])
        self.label = self.label[:, :usable].reshape(self.trnum, self.segnum, self.decision_window)
        self.label = self.label[:, :, 0]

        self.eeg_f1d = np.zeros((self.trnum, self.segnum, 10, self.eeg.shape[2]))
        for i in range(self.trnum):
            for j in range(self.segnum):
                tmp = self.eeg_t1d[i, j, :, :]
                tmp = tmp.transpose(1, 0)
                psd, de = compute_psd_de(tmp,self.decision_window/sample_rate,sample_rate)
                f_tmp = np.concatenate((psd, de)).squeeze(2)
                self.eeg_f1d[i, j, :, :] = f_tmp

        self.eeg_csp = torch.zeros((self.trnum, self.segnum, self.decision_window, self.eeg.shape[2]))
        if csp_pipe is not None:
            eeg_t = self.eeg.transpose(0,2,1)
            eeg_csp = csp_pipe.transform(eeg_t)
            eeg_csp = eeg_csp.transpose(0,2,1)
            self.eeg_csp = eeg_csp[:, :usable].reshape(self.trnum, self.segnum, self.decision_window, self.eeg.shape[2])
            self.eeg_csp = torch.tensor(self.eeg_csp, dtype=torch.float32)

        self.eeg_t2d = np.zeros((self.trnum, self.segnum, self.decision_window,10, 11))
        self.eeg_f2d = np.zeros((self.trnum, self.segnum, 10, 10, 11))
        self.eeg_csp2d = np.zeros((self.trnum, self.segnum, self.decision_window,10, 11))

        if dataset_name == 'KUL':
            map2d = KUL_2D
        elif dataset_name == 'DTU':
            map2d = DTU_2D
        else:
            raise ValueError('Supported datasets: KUL, DTU')


        for i_ch in range(10):
            for j_ch in range(11):
                if map2d[i_ch][j_ch] > 0:
                    self.eeg_t2d[:,:,:, i_ch, j_ch] = self.eeg_t1d[:,:,:, map2d[i_ch][j_ch] - 1]
                    self.eeg_f2d[:,:,:, i_ch, j_ch] = self.eeg_f1d[:,:,:, map2d[i_ch][j_ch] - 1]
                    if csp_pipe is not None:
                        self.eeg_csp2d[:,:,:, i_ch, j_ch] = self.eeg_csp[:,:,:, map2d[i_ch][j_ch] - 1]

        # to tensor

        self.eeg_t1d = torch.tensor(self.eeg_t1d, dtype=torch.float32)
        self.eeg_f1d = torch.tensor(self.eeg_f1d, dtype=torch.float32)
        self.eeg_t2d = torch.tensor(self.eeg_t2d, dtype=torch.float32)
        self.eeg_f2d = torch.tensor(self.eeg_f2d, dtype=torch.float32)
        self.eeg_csp2d = torch.tensor(self.eeg_csp2d, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.long)

        filtered_data = []
        data = self.eeg
        fs = sample_rate
        f_bands = [(1,4), (4,7), (8,13), (14,29), (30, 47)]
        for lowcut, highcut in f_bands:
            band_data = np.empty_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    band_data[i, :, j] = bandpass_filter(data[i, :, j], lowcut, highcut, fs)
            filtered_data.append(band_data)
        # print(1)
        delta_data = filtered_data[0][:, :usable].reshape(self.trnum, self.segnum, self.decision_window, self.eeg.shape[2])
        theta_data = filtered_data[1][:, :usable].reshape(self.trnum, self.segnum, self.decision_window, self.eeg.shape[2])
        alpha_data = filtered_data[2][:, :usable].reshape(self.trnum, self.segnum, self.decision_window, self.eeg.shape[2])
        beta_data = filtered_data[3][:, :usable].reshape(self.trnum, self.segnum, self.decision_window, self.eeg.shape[2])
        gamma_data = filtered_data[4][:, :usable].reshape(self.trnum, self.segnum, self.decision_window, self.eeg.shape[2])


        delta_2d = np.zeros((self.trnum, self.segnum, self.decision_window,10, 11))
        theta_2d = np.zeros((self.trnum, self.segnum, self.decision_window,10, 11))
        alpha_2d = np.zeros((self.trnum, self.segnum, self.decision_window,10, 11))
        beta_2d = np.zeros((self.trnum, self.segnum, self.decision_window,10, 11))
        gamma_2d = np.zeros((self.trnum, self.segnum, self.decision_window,10, 11))

        for i_ch in range(10):
            for j_ch in range(11):
                if map2d[i_ch][j_ch] > 0:
                    delta_2d[:,:,:, i_ch, j_ch] = delta_data[:,:,:, map2d[i_ch][j_ch] - 1]
                    theta_2d[:,:,:, i_ch, j_ch] = theta_data[:,:,:, map2d[i_ch][j_ch] - 1]
                    alpha_2d[:,:,:, i_ch, j_ch] = alpha_data[:,:,:, map2d[i_ch][j_ch] - 1]
                    beta_2d[:,:,:, i_ch, j_ch] = beta_data[:,:,:, map2d[i_ch][j_ch] - 1]
                    gamma_2d[:,:,:, i_ch, j_ch] = gamma_data[:,:,:, map2d[i_ch][j_ch] - 1]

        delta_data = torch.tensor(delta_data, dtype=torch.float32)
        theta_data = torch.tensor(theta_data, dtype=torch.float32)
        alpha_data = torch.tensor(alpha_data, dtype=torch.float32)
        beta_data = torch.tensor(beta_data, dtype=torch.float32)
        gamma_data = torch.tensor(gamma_data, dtype=torch.float32)


        self.delta_2d = torch.tensor(delta_2d, dtype=torch.float32)
        self.theta_2d = torch.tensor(theta_2d, dtype=torch.float32)
        self.alpha_2d = torch.tensor(alpha_2d, dtype=torch.float32)
        self.beta_2d = torch.tensor(beta_2d, dtype=torch.float32)
        self.gamma_2d = torch.tensor(gamma_2d, dtype=torch.float32)

        # # unsqueeze 1d
        # delta_2d = delta_2d.unsqueeze(2)
        # theta_2d = theta_2d.unsqueeze(2)
        # alpha_2d = alpha_2d.unsqueeze(2)
        # beta_2d = beta_2d.unsqueeze(2)
        # gamma_2d = gamma_2d.unsqueeze(2)
        #
        # self.fre = torch.cat((delta_2d, theta_2d, alpha_2d, beta_2d, gamma_2d), dim=2).contiguous()

        # print("finish loading data,get eeg_t1d")


    def __len__(self):
        return self.eeg.shape[0]*self.segnum



    def __getitem__(self, index):

        tr = index//self.segnum
        seg = index-tr*self.segnum
        # eeg
        t1d = self.eeg_t1d[tr, seg, :, :]
        f1d = self.eeg_f1d[tr, seg, :, :]
        t2d = self.eeg_t2d[tr, seg, :, :, :]
        f2d = self.eeg_f2d[tr, seg, :, :, :]
        tcsp = self.eeg_csp[tr, seg, :, :]
        tcsp2d = self.eeg_csp2d[tr, seg, :, :, :]
        delta2d = self.delta_2d[tr, seg, :, :, :]
        theta2d = self.theta_2d[tr, seg, :, :, :]
        alpha2d = self.alpha_2d[tr, seg, :, :, :]
        beta2d = self.beta_2d[tr, seg, :, :, :]
        gamma2d = self.gamma_2d[tr, seg, :, :, :]

        x = [t1d, f1d, t2d, f2d,tcsp,tcsp2d,delta2d,theta2d,alpha2d,beta2d,gamma2d]
        y = self.label[tr,seg]
        z = torch.tensor(tr, dtype=torch.long)

        return x,y,z
