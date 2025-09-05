%% range of the onset intensities and rates of change
%start the loop
OT_setup
auditory = {'envelope onset','mTRF envelope'};

for s=1:length(sbj)
    
    
    for k=1:2
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        cd(PATH)
        
        %get the neural data
        resp = double(EEG.data');
        
        novelty_ultm = load(sprintf('ons_ult_%s',task{k}));
        
        fs_new = EEG.srate;
        
        peak = smooth_peak(novelty_ultm.novelty_ultm,fs_new,'sigma',4);
        
        if size(resp,1)>length(peak)
            resp = resp(1:length(peak),:);
        elseif size(resp,1)<length(peak)
            peak = peak(:,1:size(resp,1));
        end
                    
       
        
        %start the trifecta of happiness
        for ao = 1:length(auditory)
            
            %extract the stimulus
            menv = extract_stimulus2(EEG, PATH, auditory{ao}, k,sbj{s},task);
            
            %normalize the envelope
            menv_norm = (menv - min(menv)) / ...
                (max(menv) - min(menv));
            
            %rather than binning the envelope -> get the values of
            %the envelope onset at the onsets indicies and bin those
            
            %get the onset peaks
            ons_idx = find(peak);
            
            %normalize those values -> will that lead to a
            %different distribution accorss participants?
            ons_env(s,k,ao,1) = min(menv_norm(ons_idx));
            ons_env(s,k,ao,2) = max(menv_norm(ons_idx));
            ons_env(s,k,ao,3) = range(menv_norm(ons_idx));
            ons_env(s,k,ao,4) = mean(menv_norm(ons_idx));
            ons_env(s,k,ao,5) = std(menv_norm(ons_idx));
            ons_env(s,k,ao,6) = prctile(menv_norm(ons_idx),2.5);
            ons_env(s,k,ao,7) = prctile(menv_norm(ons_idx),97.5);
            ons_env(s,k,ao,8) = length(menv_norm(ons_idx));
            % u_bound = prctile(data,97.5);
            
            
        end
    end
end

%plot the min and max values for each participant
figure
t = tiledlayout(2,2)
for ao = 1:2
    for k = 1:2
        
        nexttile
        plot(squeeze(ons_env(:,k,ao,1)),'linew',2)
        hold on 
        plot(squeeze(ons_env(:,k,ao,2)),'linew',2)
        title(sprintf('%s and %s',auditory{ao},task{k}))
        
        
    end
end
legend({'min','max'},'Box','off')
xlabel(t,'subjects')
ylabel(t,'a.u.')

figure
t = tiledlayout(2,2)
for ao = 1:2
    for k = 1:2
        
        nexttile
        plot(squeeze(ons_env(:,k,ao,6)),'linew',2)
        hold on 
        plot(squeeze(ons_env(:,k,ao,7)),'linew',2)
        hold on
        plot(squeeze(ons_env(:,k,ao,4)),'linew',2)
        title(sprintf('%s and %s',auditory{ao},task{k}))
%         errorbar(squeeze(ons_env(:,k,ao,4)), (squeeze(ons_env(:,k,ao,4))'-squeeze(ons_env(:,k,ao,6))'), 'horizontal', '-o', 'MarkerSize', 6, 'LineWidth', 1.5, 'CapSize', 10);

        
    end
end
legend({'2.5','97.5'},'Box','off')
xlabel(t,'subjects')
ylabel(t,'a.u.')
title(t,'Distribution of values')
                    