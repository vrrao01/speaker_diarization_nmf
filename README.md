# Speaker Diarization

## Team Name
Error 404

## Team Members
* Akshat Arun - 190101007 - Computer Science and Engineering - BTech 
* Shreyansh Meena - 190101084 - Computer Science and Engineering - BTech 
* Siddharth Charan - 190101085 - Computer Science and Engineering - BTech
* Vineet Agarwal - 190101099 - Computer Science and Engineering - BTech 
* Vivek Kumar - 190101100 - Computer Science and Engineering - BTech 
* Vignesh Ravichandra Rao - 190101109 -Computer Science and Engineering - BTech 

## Project Title
Speaker Diarization using Non-negative Matrix Factorization


## Objective
The objective of this project is to use a non-negative matrix factorization (NMF) based approach to solve the problem of Speaker Diarization that involves identifying and separating different speakers in a recording(audio signal).

## Background
Speaker diarization is a challenging problem in audio signal processing, with applications in automatic transcription, audio segmentation, speaker recognition, and speech enhancement [1], among others. Various methods have been adopted to tackle this problem, including Bayesian Source Separation and Separation by Hilbert Spectrum Subspace Decomposition [3]. In this project, we propose to use Non-Negative Matrix Factorization (NMF) as a matrix decomposition technique to address the speaker diarization task.

NMF is a widely used method for analyzing and extracting meaningful features from non-negative data, such as audio signals. It has been successfully applied in various speech and audio processing tasks. NMF allows for factorizing an audio signal into a set of basis vectors and corresponding activation coefficients, which can represent different speakers or sources in the audio. 

## Methodology
We will use NMF to separate multiple speakers in an audio signal. We believe that by decomposing the magnitude spectrogram of the audio signal using NMF, we can identify and separate different sources based on their unique spectral characteristics.

While NMF can separate an audio signal into different sources, it does not explicitly provide information about which source corresponds to which speaker [1,2]. Therefore, a clustering algorithm is needed to group the separated sources based on the similarity of their spectral features and assign each group to a different speaker. The clustering algorithm can also help to remove any spurious sources that may have been extracted by NMF but are not related to any speaker.

Our proposed methodology can be summarized in the following major steps:
1. Preprocessing: The audio signal is pre-processed to remove any noise or unwanted sounds using filtering and noise reduction techniques.
2. Short-Time Fourier Transform: Short-Time Fourier Transform (STFT) is applied to the pre-processed audio signal to obtain the spectrogram, which represents the magnitude of the frequencies over time.
3. Non-negative Matrix Factorization: NMF is applied to the magnitude spectrogram to separate the audio signal into different sources. The output of NMF is a basis matrix and an activation matrix.
4. Clustering: The separated sources are clustered using a clustering algorithm to group the sources based on their spectral features. The output of clustering is a set of clusters corresponding to different speakers.
5. Evaluation: We can evaluate our system using measures such as Speaker Error Rate (SER), Diarization Error Rate (DER), etc.

We will consider these steps in our project to accurately separate speakers in audio signals and evaluate the performance of our proposed methodology using appropriate metrics.

## Deliverables
* Abstract of the work
* Model based on the idea
    * Trying different NMF strategies
    * Trying out some clustering algorithms
* GitHub repository with code
* Presentation showing the work done


## Dataset to be used
The potential datasets for our project include:

- ICSI Meeting Corpus: This corpus contains recordings of meetings conducted in English and includes transcriptions, speaker labels, and annotations for various acoustic and linguistic events.
- VoxConverse Speaker Diarisation Dataset: VoxConverse is an audio-visual diarisation dataset that consists of multispeaker clips of human speech extracted from YouTube videos. The dataset provides speaker labelling in .rttm format.

However, it's worth noting that the audio clips in these datasets can be quite lengthy. Therefore, we may need to crop them into shorter segments containing multiple speakers for our analysis.

## References
1. Y. Chen, "Single channel blind source separation based on NMF and its application to speech enhancement," 2017 IEEE 9th International Conference on Communication Software and Networks (ICCSN), Guangzhou, China, 2017, pp. 1066-1069, doi: 10.1109/ICCSN.2017.8230274.
2. Rohlfing, Christian, Julian M. Becker, and Mathias Wien. "NMF-based informed source separation." 2016 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2016.
3. Patki, Kedar. "Review of single channel source separation techniques." Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR 2013).

