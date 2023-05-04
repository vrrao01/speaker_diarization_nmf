import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import IPython.display as ipd
import scipy.io.wavfile as wavfile
import scipy

def plot_NMF_iter(W, H,beta,sr = 5512,hop = 16,iteration = None):
    """
    Plots the results of a single iteration of Non-negative Matrix Factorization (NMF).

    Parameters:
    -----------
    W : numpy array
        Dictionary matrix of shape (n_features, n_components).
    H : numpy array
        Temporal activations matrix of shape (n_components, n_samples).
    beta : float
        Beta divergence value used for NMF.
    sr : int
        Sample rate of original audio recording
    hop: int
        Hop length used in performing STFT
    iteration : int, optional
        Iteration number, default is None.

    Returns:
    --------
    None

    """

    f = plt.figure(figsize=(5,5))
    f.suptitle(f"NMF Iteration {iteration}, for beta = {beta}", fontsize=8,)
    
    # definitions for the axes
    V_plot = plt.axes([0.35, 0.1, 1, 0.6])
    H_plot = plt.axes([0.35, 0.75, 1, 0.15])
    W_plot = plt.axes([0.1, 0.1, 0.2, 0.6])

    D = librosa.amplitude_to_db(W@H, ref = np.max)

    librosa.display.specshow(W,y_axis = 'hz', sr=sr, hop_length=hop,x_axis ='time',cmap= matplotlib.cm.jet, ax=W_plot)
    librosa.display.specshow(H,y_axis = 'hz', sr=sr, hop_length=hop,x_axis ='time',cmap= matplotlib.cm.jet, ax=H_plot)
    librosa.display.specshow(D,y_axis = 'hz', sr=sr, hop_length=hop,x_axis ='time',cmap= matplotlib.cm.jet, ax=V_plot)

    W_plot.set_title('Dictionary W', fontsize=10)
    H_plot.set_title('Temporal activations H', fontsize=10)

    W_plot.axes.get_xaxis().set_visible(False)
    H_plot.axes.get_xaxis().set_visible(False)
    V_plot.axes.get_yaxis().set_visible(False)


def NMF_custom(V, S, beta = 2,  threshold = 0.05, MAXITER = 1000, display = True , displayEveryNiter = None): 
    """
    Perform Non-negative Matrix Factorization (NMF) using the given beta divergence measure
    to decompose the mixture signal V into S sources.

    Parameters
    ----------
    V : array_like
        The mixture signal matrix of shape (K, N).
    S : int
        The number of sources to extract.
    beta : float, optional
        The divergence measure to use. Default is 2, which corresponds to Euclidean distance.
        Common values are 0 (Itakura-Saito), 1 (Kullback-Leibler) and 2 (Euclidean distance).
    threshold : float, optional
        The stopping criterion. When the relative improvement in the cost function
        falls below this threshold, the algorithm stops. Default is 0.05.
    MAXITER : int, optional
        The maximum number of iterations. Default is 1000.
    display : bool, optional
        Whether to display the NMF iterations. Default is True.
    displayEveryNiter : int or None, optional
        If display=True, only display the NMF iteration at every N-th iteration.
        If set to None, display all iterations. Default is None.

    Returns
    -------
    W : array_like
        The dictionary matrix of shape (K, S) with non-negative elements.
    H : array_like
        The activation matrix of shape (S, N) with non-negative elements.
    cost_function : list
        The cost function (beta divergence) values for each iteration.

    Algorithm
    ---------
    1) Randomly initialize W and H matrices.
    2) Multiplicative update of W and H.
    3) Repeat step 2 until convergence or after MAXITER iterations.

    """
    counter  = 0
    cost_function = []
    beta_divergence = 1
    
    K, N = np.shape(V)
    
    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))
    
    # Plotting the first initialization
    if display == True : plot_NMF_iter(W,H,beta,counter)


    while beta_divergence >= threshold and counter < MAXITER:
        
        # Update of W and H
        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
        
        
        # Compute cost function
        beta_divergence =  divergence(V,W,H, beta = 2)
        cost_function.append( beta_divergence )
        counter +=1

        if  display == True  and counter%displayEveryNiter == 0  : plot_NMF_iter(W,H,beta,counter)

        
    
    if counter == MAXITER : print(f"Stop after {MAXITER} iterations.")
    else : print(f"Convergence after {counter} iterations.")
        
    return W,H, cost_function 


def divergence(V,W,H, beta = 2):
    
    """
    Computes the beta divergence between V and W.H. This is used as the cost function
    
    Parameters:
        V (ndarray): the input matrix of shape (F, T)
        W (ndarray): the dictionary matrix of shape (F, K)
        H (ndarray): the activation matrix of shape (K, T)
        beta (float): the parameter controlling the type of divergence to compute (default=2)
        
    Returns:
        div (float): the beta divergence between V and W.H
    """
    
    if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
    
    if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
    
    if beta == 2 : return 1/2*np.linalg.norm(W@H-V)

    div = np.sum((beta / (beta - 1)) * (V**beta - np.dot(W, H)**beta - V**(beta-1) * np.dot(W, H)))
    return div
    
    

def plot_filtered_spectrograms(V, W, H, S, sr = 5512, hop = 16):
    """
    Plots the frequency mask of each audio source S over time after non-negative matrix factorization.
    
    Parameters:
    V (ndarray): the magnitude spectrogram of the mixture signal
    W (ndarray): the learned spectral basis matrix of shape (F, K)
    H (ndarray): the learned activation matrix of shape (K, T)
    S (int): the number of audio sources
    sr (int): the sample rate of the audio signal
    hop (int): the hop length in samples
    
    Returns:
    filtered_spectrograms (list): a list of filtered spectrograms for each audio source
    """
    
    f, axs = plt.subplots(nrows=S, ncols=1, figsize=(10, 20))
    filtered_spectrograms = []
    
    for i in range(S):
        axs[i].set_title(f"Frequency Mask of Audio Source s = {i+1}")
        
        # Filter each source component
        WsHs = W[:, [i]] @ H[[i], :]
        filtered_spectrogram = WsHs / (W @ H) * V 
        
        # Compute the filtered spectrogram
        D = librosa.amplitude_to_db(filtered_spectrogram, ref=np.max)
        
        # Show the filtered spectrogram
        librosa.display.specshow(D, y_axis='hz', sr=sr, hop_length=hop, x_axis='time', cmap=matplotlib.cm.jet, ax=axs[i])
        
        filtered_spectrograms.append(filtered_spectrogram)
        
    return filtered_spectrograms