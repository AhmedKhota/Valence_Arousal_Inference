# Valence_Arousal_Inference
Valence, Arousal Inference of sounds (robot sounds) using random forest model

In order to study and quantify the emotional meaning of non-linguistic utterances (NLUs), sounds of characters making such sounds were extracted from various movies, tv-shows and video games (see table below for details of the extracted sounds). An experiment was done whereby subjects rated the emotional Valence and Arousal of the sounds. Features of the sounds were extracted using the openSMILE and pyAudioAnalysis feature sets and factor analysis was done to identify important features where applicable. Using the ratings gathered from the experiment, a random forest model was trained to infer the valence and arousal of NLUs and achieved a Mean Absolute Error (MAE) of 0.107 for valence and 0.097 for arousal.

![560 sounds sources table](https://github.com/AhmedKhota/Valence_Arousal_Inference/assets/139664971/16ff7b49-5256-4384-a10a-45d1a2258411)

The distribution of ratings gathered from the experiment is shown in the figures below.

![560ExpResults](https://github.com/AhmedKhota/Valence_Arousal_Inference/assets/139664971/80263746-6d94-4258-a66d-46aa54ca2151)

Scatterplots of predicted vs. actual valence and arousal values are shown in the figures below. The correlation for valence is 0.63 and for arousal 0.75.

![pvsascatters](https://github.com/AhmedKhota/Valence_Arousal_Inference/assets/139664971/5bfb37e6-cd81-46b9-ab15-ebdf7c6d0742)


