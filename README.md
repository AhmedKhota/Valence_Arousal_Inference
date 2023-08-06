# Valence_Arousal_Inference
Valence, Arousal Inference of sounds (robot sounds) using random forest model

In order to study and quantify the emotional meaning of non-linguistic utterances (NLUs), sounds of characters making such sounds were extracted from various movies, tv-shows and video games (see table below for details of the extracted sounds). An experiment was done whereby subjects rated the emotional Valence and Arousal of the sounds. Features of the sounds were extracted using the openSMILE and pyAudioAnalysis feature sets and factor analysis was done to identify important features where applicable. Using the ratings gathered from the experiment, a random forest model was trained to infer the valence and arousal of NLUs and achieved a Mean Absolute Error (MAE) of 0.107 for valence and 0.097 for arousal.

![560 sounds sources table](https://github.com/AhmedKhota/Valence_Arousal_Inference/assets/139664971/16ff7b49-5256-4384-a10a-45d1a2258411)
