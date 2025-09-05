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


for s=1:length(sbj)

    for k=1:2

        
        [EEG,PATH] = OT_preprocessing(s,k,sbj,40);

        cd(PATH)
        
        
        %% compute the new onsets
        if k ==1
            wav = load(sprintf('%s_narrow_audio_strct.mat',sbj{s}));
            env = extract_stimulus2(EEG, PATH,'mTRF envelope', k, sbj{s});
        else
            wav = load(sprintf('%s_wide_audio_strct.mat',sbj{s}));
            env = extract_stimulus2(EEG, PATH,'mTRF envelope', k, sbj{s});
        end
        
        %start the threshold loop
        for tsh = 1:length(thresh)
            %my own function
            [novelty, fs_new] = energy_novelty(double(wav.audio_strct.data)',wav.audio_strct.srate,'H',441);
            pks = simp_peak(novelty,thresh(tsh));
            if length(pks) > EEG.pnts
                pks = pks(1,1:EEG.pnts)';
            elseif length(pks) < EEG.pnts
                x(s,1) = EEG.pnts - length(pks);
                pks = [pks zeros(1,x(s,1))]';
            end
            
            
            %get the neural data
            resp = EEG.data';
            stim = pks;
            
            if size(resp,1)>size(stim,1)
                resp = resp(1:size(stim,1),:);
            end
            
            %% compute here the event density for the peaks
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
            
            
            
            
            %% extract the neural response epochs
            
            %is that necessary?
%             EEG.data = zscore(EEG.data,[],'all');

            [epo_dat,epo_stim,~,stim_z] = OT_epochize(EEG,stim,t,0);
            stim_idx = find(stim_z==1);

            %find the epochs that fall into the range
            h_idx = [];
            l_idx = [];
            h_rank=[];
            l_rank = [];
            for id = 1:length(stim_idx)
                
                %check if the epoch is in the highest segments
                for sl = 1:sel_thrsh
                    if stim_idx(id) > time_avec_sort(sl,1) && stim_idx(id) < time_avec_sort(sl,2)
                        h_idx = [h_idx; find(stim_idx == stim_idx(id))]; 
                        h_rank = [h_rank;sl];
                    elseif stim_idx(id) > time_avec_sort(end-sl+1,1) && stim_idx(id) < time_avec_sort(end-sl+1,2)
                        l_idx = [l_idx; find(stim_idx == stim_idx(id))];
                        l_rank = [l_rank;sl];

                    end
                end
            end
            %delete double values 
            [hu_idx,ih] = unique(h_idx);
            [lu_idx,il] = unique(l_idx);
            
            %adjust rank
            hr_un = h_rank(ih);
            lr_un = l_rank(il);
            
            %sort the ranks
            [hr_sort,sh_idx] = sort(hr_un,'ascend');
            [lr_sort,sl_idx] = sort(lr_un,'ascend');
            
            %sort the epochs idx according to rank 
            h_sort = hu_idx(sh_idx);
            l_sort = lu_idx(sl_idx);

            %correct for different number of onsets -> throw them out
            if size(l_sort,1) > size(h_sort,1)
                l_dif = size(l_sort,1) - ((size(l_sort,1) - size(h_sort,1)));
                l_sort = l_sort(1:l_dif);
            elseif size(l_sort,1) < size(h_sort,1)
                h_dif = size(h_sort,1) - ((size(h_sort,1) - size(l_sort,1)));
                h_sort = h_sort(1:h_dif);
            end
            
            %should i baseline correct either of the two ERPs?
            h_epo = epo_dat(:,:,h_sort);
            l_epo = epo_dat(:,:,l_sort);

            for tr = 1:length(h_sort)
                for ch = 1:EEG.nbchan
                    %remove the base line
                    h_epo(ch,:,tr) = squeeze(h_epo(ch,:,tr)) - mean(h_epo(ch,base_idx(1):base_idx(2),tr),2);
                    l_epo(ch,:,tr) = squeeze(l_epo(ch,:,tr)) - mean(l_epo(ch,base_idx(1):base_idx(2),tr),2);
                end
            end
                

            high_epo = mean(h_epo,3);
            low_epo =  mean(l_epo,3);
            
            if any(isnan(high_epo),'all') || any(isnan(low_epo),'all')
                na_idx (s,k,tsh) = 1;
                
            end
            
            h_dat(s,k,tsh,:,:) = high_epo;
            l_dat(s,k,tsh,:,:) =low_epo;
            
            %get the difference curve
            h_l_dif(s,k,tsh,:,:) =high_epo - low_epo;
            
            nr_ons(s,k,tsh,:) = length(h_sort);
            
            %save the number of onsets
            sum_stim(s,k,tsh) = sum(stim);
            
            %save the average density estimate [low high]
            avg_dns(s,k,tsh,:) = [mean(ons_dns(1,end-sel_thrsh+1:end)) mean(ons_dns(1,1:sel_thrsh)) ];
           
        end
        
        
            
    end
end


cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results\')
DNS_epoch = struct()
DNS_epoch.h_dat = h_dat;
DNS_epoch.l_dat = l_dat;
DNS_epoch.thresh = thresh;
DNS_epoch.nr_ons = nr_ons;
DNS_epoch.h_l_dif = h_l_dif;
DNS_epoch.na_idx = na_idx;
DNS_epoch.erp_time = erp_time;
DNS_epoch.avg_dns = avg_dns;
DNS_epoch.sum_stim = sum_stim;
DNS_epoch.t = 'intial analysis to test whether density difference is a factor of selected onsets';
save('DNS40_epo_thresh.mat','-struct','DNS_epoch')


%% plotting
le = length(erp_time);
fig_pos = [57         185        1424         793]


for ao = 1:15%length(thresh)
    temp_dns = squeeze(mean(mean(avg_dns(:,:,ao,:),1),2));
    temp_sum = squeeze(mean(mean(sum_stim(:,:,ao),1),2));
    figure
    temp_dat = squeeze(mean(mean(h_dat(:,:,ao,:,:),2,'omitnan'),4,'omitnan'));
    
    dnsMean = squeeze(mean(temp_dat,1,'omitnan'));
    h = plot(dnsMean,'b','linew',2);
    hold on
    N = size(h_dat,1);
    ySEM = std(temp_dat,1)/sqrt(N);
    CI95 = tinv([0.025 0.975],N-1);
    yCI95 = bsxfun(@times,ySEM,CI95(:));
    conv = yCI95 + dnsMean ;
    x2 = [linspace(1,le,le) fliplr(linspace(1,le,le))];
    inbe = [conv(1,:) fliplr(conv(2,:))];
    f = fill(x2,inbe,'b');
    f.FaceAlpha = 0.2;
    f.EdgeAlpha = 0.4;
    f.LineWidth = 0.5;
    hold on
    
    temp_datl = squeeze(mean(mean(l_dat(:,:,ao,:,:),2,'omitnan'),4,'omitnan'));
    dnslMean = squeeze(mean(temp_datl,1,'omitnan'));
    l = plot(dnslMean,'r','linew',2);
    hold on
    ylSEM = std(temp_datl,1)/sqrt(N);
    CI95 = tinv([0.025 0.975],N-1);
    yCI95 = bsxfun(@times,ylSEM,CI95(:));
    conv = yCI95 + dnslMean;
    x2 = [linspace(1,le,le) fliplr(linspace(1,le,le))];
    inbe = [conv(1,:) fliplr(conv(2,:))];
    f = fill(x2,inbe,'r');
    f.FaceAlpha = 0.1;
    f.EdgeAlpha = 0.2;
    f.LineWidth = 0.5;

    l = legend([h,l],'high dns','low dns','Fontsize',24,'Location','southeast');
    set(l,'Box','off')

    set(gca,'XTick', linspace(1,le,16),'XTickLabel',linspace(-500,1000,16),'Xlim',[40 102],'Fontsize',24)
    title(sprintf('ERP %0.3f [%0.2f %0.2f] sum = %d',thresh(ao),temp_dns(1), temp_dns(2),temp_sum),'FontSize',30)
    
    xlabel('Time (ms)','Fontsize',24)
    ylabel('[\muV]', 'Interpreter', 'tex','Fontsize',24);
    
    set(gcf,'Position',fig_pos)
    box off
    
    save_fig(gcf,'\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\thresh_dns\',sprintf('DNS_ERP_%0.2f',thresh(ao)))
    
end








