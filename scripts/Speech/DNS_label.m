%testing of the different onset thresholds and how it impacts the
%computation for the different dns conditions
MAINPATH = 'O:\projects\thh_ont\auditory-attention-in-complex-work-related-auditory-envrionments\data files'
addpath(genpath(MAINPATH));

OT_setup

thresh = logspace(log(0.22),log(0.8),18);

direc = 1; %specifies the forward modeling
t = [-0.5 1];
base = [-0.2 -0.01]
erp_time = t(1):0.01:t(2) %0.1 is the sample rate here
base_idx = dsearchn(erp_time',base');

na_idx = zeros(length(sbj),length(task),length(thresh));


hr_dat = cell(length(task),length(thresh));
lr_dat=  cell(length(task),length(thresh));

win_l = 20;
win_h = 5;

sounds_h = [];
sounds_l = [];
classes_h = [];
classes_l = [];

fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Label\'

%% load the Yament
downloadFolder = fullfile(tempdir,'YAMNetDownload');
loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/yamnet.zip');
YAMNetLocation = tempdir;
unzip(loc,YAMNetLocation)
addpath(fullfile(YAMNetLocation,'yamnet'))

net = yamnet

%%


for s=1:length(sbj)

    for k=1:2

        
        [EEG,PATH] = OT_preprocessing(s,k,sbj,40);

        cd(PATH)
        
        %get the audio files
        if k == 1
            [audioIn,fs] =audioread(sprintf('narrow_audio_game_%s.wav',sbj{s}));
        else
            [audioIn,fs] =audioread(sprintf('wide_audio_game_%s.wav',sbj{s}));
            
        end
        
        %sorted according to interonset distance
        stim = extract_stimulus2(EEG, PATH,'onset', k, sbj{s});
        env = extract_stimulus2(EEG, PATH,'mTRF envelope', k, sbj{s});
        win_lo = win_l*EEG.srate;
        win_ho = win_h*EEG.srate;

        % Compute the number of output samples
        num_output_samples = floor((length(stim) - win_lo)/win_ho) + 1;

        % Initialize the output signal
        ons_dist = zeros(num_output_samples,1);
        ons_dns = zeros(1,num_output_samples);
        

        % Apply the moving average filter to the input signal
        for i = 1:num_output_samples
            start_index = (i-1)*win_ho + 1;
            end_index = start_index + win_lo - 1;
            ons_dns(1,i) = mean(stim(start_index:end_index,1));
            ons_dist(i,1) = mean(diff(find(stim(start_index:end_index,1))));
        end
        audio_dns = ons_dns;
        %select the 5% highest and lowest peaks
        [audio_dns_sort, audio_sort_idx] = sort(audio_dns,'descend'); % <-- crucial change happens here
        audio_rms_sort = audio_dns_sort;
        sel_thrsh = 10;%ceil(length(audio_dns)*0.05);
        %select the persons rms values for the high and low peaks
%         rms_sbj(s,k,:) = [mean(audio_rms_sort(1,1:sel_thrsh)), mean(audio_rms_sort(1,end+1-sel_thrsh:end))];
        ons_dns = ons_dns(1,audio_sort_idx);

        %find the conversion to the real data ....
        time_win = [linspace(0,EEG.xmax-win_l,size(audio_dns,2))' linspace(win_l,EEG.xmax,size(audio_dns,2))'];
        time_eeg = linspace(EEG.xmin,EEG.xmax,EEG.pnts)';
        
        %adjust to the actual EEG data by finding the minimal offset
        time_avec = [];
        for i=1:size(time_win,1)
            [~, time_idx1] = min(abs(time_eeg - time_win(i,1)));
            [~, time_idx2] = min(abs(time_eeg - time_win(i,2)));
            
            time_avec(i,:) = [time_idx1 time_idx2];
        end
        time_avec_sort = time_avec(audio_sort_idx,:);
        
        
        %select the low peaks
        pks_ldns = [time_avec_sort(end-sel_thrsh+1:end,1) time_avec_sort(end-sel_thrsh+1:end,2)];
        pks_ldns = pks_ldns(pks_ldns(:,1)>=0,:);
        
        %select the high peaks
        pks_dns = [time_avec_sort(1:sel_thrsh,1) time_avec_sort(1:sel_thrsh,2)];
        pks_dns = pks_dns(pks_dns(:,1)>=0,:);
            
        pks_dns441 = (pks_dns/EEG.srate)*fs;
        pks_ldns441 = (pks_ldns/EEG.srate)*fs;
        
        %sort the envelope accordingly
        fl = figure;
%         title(fl, 'Low Density', 'FontSize', 14); % Title for the low density figure
        fh = figure;
%         title(fh, 'High Density', 'FontSize', 14); % Title for the high density figure
        for i = 1:length(pks_ldns)
            
            env_l{s,k,i,:} = env(pks_ldns(i,1):pks_ldns(i,2));
            stim_l{s,k,i,:} = stim(pks_ldns(i,1):pks_ldns(i,2));
            
            temp_classdat_l = audioIn(pks_ldns441(i,1):pks_ldns441(i,2));
            [sounds,timeStamps] = classifySound(temp_classdat_l,fs);
            sounds_l = [sounds_l sounds];
            %
            melSpectYam = yamnetPreprocess(temp_classdat_l,fs);
            classes = classify(net,melSpectYam);
            classes_l = [classes_l classes];
            figure(fl)
            if i < 6
                subplot(5,1,i)
                plot(env_l{s,k,i,:},'r')
                hold on
                plot(stim_l{s,k,i,:},'k')
                %plot the classes
                timeStamps = timeStamps*EEG.srate
                textHeight = 1.1;
                for idx = 1:numel(sounds)
                    patch([timeStamps(idx,1),timeStamps(idx,1),timeStamps(idx,2),timeStamps(idx,2)], ...
                        [-1,1,1,-1], ...
                        [0.3010 0.7450 0.9330], ...
                        'FaceAlpha',0.2);
                    text(timeStamps(idx,1),textHeight+0.05*(-1)^idx,sounds(idx))
                end
            end
            
            
            env_h{s,k,i,:} = env(pks_dns(i,1):pks_dns(i,2));
            stim_h{s,k,i,:} = stim(pks_dns(i,1):pks_dns(i,2));
            
            temp_classdat_h = audioIn(pks_dns441(i,1):pks_dns441(i,2));
            [sounds,timeStamps] = classifySound(temp_classdat_h,fs);
            sounds_h = [sounds_h sounds];
            
            %
            melSpectYam = yamnetPreprocess(temp_classdat_h,fs);
            classes = classify(net,melSpectYam);
            classes_h = [classes_h classes];
            figure(fh)
            if i < 6
                subplot(5,1,i)
                plot(env_h{s,k,i,:},'b')
                hold on
                plot(stim_h{s,k,i,:},'k')
                %plot the classes
                timeStamps = timeStamps*EEG.srate
                textHeight = 1.1;
                for idx = 1:numel(sounds)
                    patch([timeStamps(idx,1),timeStamps(idx,1),timeStamps(idx,2),timeStamps(idx,2)], ...
                        [-1,1,1,-1], ...
                        [0.3010 0.7450 0.9330], ...
                        'FaceAlpha',0.2);
                    text(timeStamps(idx,1),textHeight+0.05*(-1)^idx,sounds(idx))
                end
            end




        end
        %% extract the neural response epochs
        
        %is that necessary?
        %             EEG.data = zscore(EEG.data,[],'all');
        
%         [epo_dat,epo_stim,~,stim_z] = OT_epochize(EEG,stim,t,0);
%         stim_idx = find(stim_z==1);
%         
%         %find the epochs that fall into the range
%         h_idx = [];
%         l_idx = [];
%         h_rank=[];
%         l_rank = [];
%         for id = 1:length(stim_idx)
%             
%             %check if the epoch is in the highest segments
%             for sl = 1:sel_thrsh
%                 if stim_idx(id) > time_avec_sort(sl,1) && stim_idx(id) < time_avec_sort(sl,2)
%                     h_idx = [h_idx; find(stim_idx == stim_idx(id))];
%                     h_rank = [h_rank;sl];
%                 elseif stim_idx(id) > time_avec_sort(end-sl+1,1) && stim_idx(id) < time_avec_sort(end-sl+1,2)
%                     l_idx = [l_idx; find(stim_idx == stim_idx(id))];
%                     l_rank = [l_rank;sl];
%                     
%                 end
%             end
%         end
%         %delete double values
%         [hu_idx,ih] = unique(h_idx);
%         [lu_idx,il] = unique(l_idx);
%         
%         %adjust rank
%         hr_un = h_rank(ih);
%         lr_un = l_rank(il);
%         
%         %sort the ranks
%         [hr_sort,sh_idx] = sort(hr_un,'ascend');
%         [lr_sort,sl_idx] = sort(lr_un,'ascend');
%         
%         %sort the epochs idx according to rank
%         h_sort = hu_idx(sh_idx);
%         l_sort = lu_idx(sl_idx);
%         
%         %correct for different number of onsets -> throw them out
%         if size(l_sort,1) > size(h_sort,1)
%             l_dif = size(l_sort,1) - ((size(l_sort,1) - size(h_sort,1)));
%             l_sort = l_sort(1:l_dif);
%         elseif size(l_sort,1) < size(h_sort,1)
%             h_dif = size(h_sort,1) - ((size(h_sort,1) - size(l_sort,1)));
%             h_sort = h_sort(1:h_dif);
%         end
%         
%         %should i baseline correct either of the two ERPs?
%         h_epo = epo_dat(:,:,h_sort);
%         l_epo = epo_dat(:,:,l_sort);
%         
%         for tr = 1:length(h_sort)
%             for ch = 1:EEG.nbchan
%                 %remove the base line
%                 h_epo(ch,:,tr) = squeeze(h_epo(ch,:,tr)) - mean(h_epo(ch,base_idx(1):base_idx(2),tr),2);
%                 l_epo(ch,:,tr) = squeeze(l_epo(ch,:,tr)) - mean(l_epo(ch,base_idx(1):base_idx(2),tr),2);
%             end
%         end
%         
%         
%         high_epo = mean(h_epo,3);
%         low_epo =  mean(l_epo,3);
%         
%         if any(isnan(high_epo),'all') || any(isnan(low_epo),'all')
%             na_idx (s,k,tsh) = 1;
%             
%         end
%         
%         h_dat(s,k,:,:) = high_epo;
%         l_dat(s,k,:,:) =low_epo;
%         
%         %get the difference curve
%         h_l_dif(s,k,:,:) =high_epo - low_epo;
%         
%         nr_ons(s,k,:) = length(h_sort);
%         
%         %save the number of onsets
%         sum_stim(s,k) = sum(stim);
%         
        
        
        
        
        
            
    end
end


cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results\')
DNS_epoch = struct()
DNS_epoch.sounds_h = sounds_h;
DNS_epoch.sounds_l = sounds_l;
DNS_epoch.classes_l = classes_l;
DNS_epoch.classes_h = classes_h;
DNS_epoch.env_h = env_h;
DNS_epoch.env_l = env_l;
DNS_epoch.stim_h = stim_h;
DNS_epoch.stim_l = stim_l;
DNS_epoch.t = 'labeling of segments -> discrepance between high and low classes';
save('DNS40_epo_label.mat','-struct','DNS_epoch')


%% plotting

figure
wordcloud(sounds_h)
sum(strcmp(sounds_h,'Speech'))/length(sounds_h)

figure
wordcloud(sounds_l)
sum(strcmp(sounds_l,'Speech'))/length(sounds_l)

figure
wordcloud(classes_h)
[numOccurrences,uniqueWords] = histcounts(classes_h);
speech_idx = find(strcmp(uniqueWords,'Speech'));
numOccurrences(speech_idx)/sum(numOccurrences)

figure
wordcloud(classes_l)
[numOccurrences,uniqueWords] = histcounts(classes_l);
speech_idx = find(strcmp(uniqueWords,'Speech'));
numOccurrences(speech_idx)/sum(numOccurrences)



