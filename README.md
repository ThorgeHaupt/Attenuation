# Neural response attenuates with decreasing inter-onset intervals between sounds in a natural soundscape.
This code repository contains the scripts for this paper

## Abstract
Sensory attenuation of auditory evoked potentials (AEPs), particularly N1 and P2 components, has been widely demonstrated in response to simple, repetitive stimuli sequences of isolated synthetic sounds. It remains unclear, however, whether these effects generalize to complex soundscapes where temporal and acoustic features vary more broadly and dynamically.
In this study, we investigated whether the inter-onset interval (IOI), the time between successive sound events, modulates AEP amplitudes in a complex auditory scene.
We derived acoustic onsets from a naturalistic soundscape and applied temporal response function (TRF) analysis to EEG data recorded from normal hearing human listeners (N = 22, 16 females, 6 males).
Our results showed that shorter IOIs are associated with attenuated N1 and P2 amplitudes, replicating classical adaptation effects in a naturalistic soundscape. These effects remained stable when controlling for other acoustic features such as intensity and envelope sharpness and across different TRF model specifications. Integrating IOI information into predictive modelling revealed that neural dynamics were captured more effectively than simpler onset models when training data were matched. 
These findings highlight the brain’s sensitivity to temporal structure even in highly variable auditory environments, and show that classical lab findings generalize to naturalistic soundscapes. Our results underscore the need to include temporal features alongside acoustic ones in models of real-world auditory processing. 


### Toolboxes
It requires several toolboxes to run: <br/>
EEGlab - https://de.mathworks.com/matlabcentral/fileexchange/56415-eeglab <br/>
mtrftoolbox - https://github.com/mickcrosse/mTRF-Toolbox <br/>
audionovelty - https://github.com/ThorgeHaupt/Audionovelty <br/>
Data - https://zenodo.org/records/7147701 <br/>
functions from - https://github.com/ThorgeHaupt/RelevantFeaturesSoundPerception <br/>
field trip - https://www.fieldtriptoolbox.org/ <br/>
violinplot - https://github.com/bastibe/Violinplot-Matlab <br/>
sigstar - https://de.mathworks.com/matlabcentral/fileexchange/39696-raacampbell-sigstar <br/>


