%test the different onsets, and whether removing onsets closer than 500ms
%from each other will help 

%lets do this shit manually
%i want to write a script that extracts the erps and arranges them
%according to the event density

OT_setup


direc = 1; %specifies the forward modeling
lambda = 0.05;
t = [-0.5 1];
base = [-0.2 -0.01]
erp_time = t(1):0.01:t(2) %0.1 is the sample rate here
base_idx = dsearchn(erp_time',base');

auditory = {'onset'}%,'alarm','irregular','odd'};
na_idx = zeros(length(sbj),length(task),length(auditory));


hr_dat = cell(length(task),length(auditory));
lr_dat=  cell(length(task),length(auditory));

win_l = 20;
win_h = 5;

fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Dumbons\';

for s=1:length(sbj)
    
    
    for k=1:2
        
        EEG = [];
        [EEG,PATH] = OT_preprocessing(s,k,sbj,40);
        
%         EEG.data = detrend(EEG.data)
        
        %get the event density
        fs_new = 16000;
       

        %sorted according to interonset distance
        label = string(auditory);
        stim = extract_stimulus2(EEG, PATH,'onset', k, sbj{s});
        
%         stim_idx = find(stim);
%         stim_dif = diff(stim_idx);
%         stim_clean = stim_idx(1);
%         thresh = 0.5*EEG.srate;
%         %remove the onsets that are closer than X ms
%         i = 1;
%         ons_del_idx = [];
%         while i < sum(stim_idx_ad)
%             ons_del = find(stim_idx_ad(i) > stim_idx);
%             if ons_del(end) > i
%                 ons_del_idx = [ons_del_idx; ons_del(ons_del>i)]
%                 i = ons_del_idx(end) +1;
%             else
%                 i = i+1;
%             end
%         end
%                 
%             for k = 2:sum(stim_idx)
%                 if  0<stim_idx_ad(i) -stim_idx(k)
%                     
%             end
%             if stim_idx - stim_idx_ad(i)
%             
%             
%             
%             if stim_idx(i)-stim_clean(end) > thresh
%                 stim_clean(end+1) = stim_idx(i);
%                 
%             end
%         end
%         stim_adap = zeros(size(stim));
%         stim_adap(stim_clean) = 1;
%         
%         stim = stim_adap;
        
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
        
         for ao=1:length(auditory)
            
            
            %extract the stimulus
            label = string(auditory(ao));
%             stim = extract_stimulus2(EEG, PATH, label, k, sbj{s},task);
            [epo_dat,epo_stim,~,stim_z] = OT_epochize(EEG,stim,t,1);
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
                na_idx (s,k,ao) = 1;
                
            end
            
            h_dat(s,k,ao,:,:) = high_epo;
            l_dat(s,k,ao,:,:) =low_epo;
            
            %get the difference curve
            h_l_dif(s,k,ao,:,:) =high_epo - low_epo;
            
            nr_ons(s,k,ao,:) = length(h_sort);
            

            
         end
    end
end

%% Save your stuff
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results\')
DNS_epoch = struct()
DNS_epoch.h_dat = h_dat;
DNS_epoch.l_dat = l_dat;
DNS_epoch.auditory = auditory;
DNS_epoch.nr_ons = nr_ons;
DNS_epoch.h_l_dif = h_l_dif;
DNS_epoch.na_idx = na_idx;
DNS_epoch.erp_time = erp_time;
DNS_epoch.t = 'removed onsets following 500ms within each other, sorted to the new stimulus';
save('DNS40_epo_rmdumbons.mat','-struct','DNS_epoch')


%% plotting
erp_time = linspace(-500,1000,size(h_dat,5));
base_win = [-400 -200];
base_idx = dsearchn(erp_time',base_win');

%window of interest
win_int = [-100 500];
win_idx = dsearchn(erp_time',win_int');

% c_idx = 8; %this is Cz
% EEGl = EEG;
% EEGh = EEG;

t_erp_time = 0:10:500;
%% plotting the average ERPS and standard deviations
le = length(erp_time);
fig_pos = [57   418   749   560]
%figure,clf

for ao = 1:length(auditory)
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

    l = legend([h,l],'high dns','low dns','Fontsize',24,'Location','southeast')
    set(l,'Box','off')

    set(gca,'XTick', linspace(1,le,16),'XTickLabel',linspace(-500,1000,16),'Xlim',[40 102],'Fontsize',24)
    title(sprintf('ERP %s',auditory{ao}),'FontSize',30)
    
    xlabel('Time (ms)','Fontsize',24)
    ylabel('[\muV]', 'Interpreter', 'tex','Fontsize',24);
    
    set(gcf,'Position',fig_pos)
    box off
    
%     save_fig(gcf,fig_path,'DNS_rmdumbones')
    
end



%% plot difference curve

figure
tiledlayout(length(task),length(auditory))
for k=1:2
    for ao = 1:length(auditory)
        nexttile
        plot(erp_time(win_idx(1):win_idx(2)), squeeze(mean(mean(h_l_dif(:,k,ao,:,win_idx(1):win_idx(2)),4,'omitnan'),1,'omitnan'))','r')
        hold on
        yline(0)
        title(sprintf('%s %s',task{k},auditory{ao}))
    end
end


%% permutation testing 

t_win = [0 500];
perm = 10000;
dif_t = []

figure,clf
tiledlayout(length(task),length(auditory))
sgtitle(sprintf('Permutation test, win:[%d %d] ms. ',squeeze(t_win(1,1)),squeeze(t_win(1,2))))

for k=1:2
    for ao = 1:length(auditory)
        tw_idx = dsearchn(erp_time',[0 500]');%squeeze(t_win(k,ao,:)));

        %compute the t-statistic
        %1. get the mean difference between the condtions
        dif_val = squeeze(mean(h_l_dif(:,k,ao,:,tw_idx(1,1):tw_idx(2,1)),4,'omitnan'));
        dif_val = dif_val(~isnan(dif_val(:,1)),:);
        
        %2.compute the t-values for each sample point
        for t = 1:size(dif_val,2)
            dif_t(k,ao,t) = mean(dif_val(:,t))/(std(dif_val(:,t))/sqrt(size(dif_val,1)));
        end 
        %get the critical t-value
        t_val = tinv([0.025 0.975],size(dif_val,1)-1);
        dif_ti = squeeze(dif_t(k,ao,:));
        %4. find t-vals that are more extreme than crit. t-val
        lowb = abs(sum(dif_ti(find(dif_ti<t_val(1,1)))));
        highb = abs(sum(dif_ti(find(dif_ti>t_val(1,2)))));
        
        %select largest cluster
        test_s = max([lowb highb]);
        
        %set up container
        t0_dist = zeros(perm,1);
        %start the permutation
        for prm = 1:perm
            rh_idx = randperm(size(sbj,2),size(sbj,2)/2);
            rl_idx = setdiff(1:1:size(sbj,2),rh_idx);
            hr_dat = squeeze(cat(1,h_dat(rh_idx,k,ao,:,tw_idx(1):tw_idx(2)),l_dat(rl_idx,k,ao,:,tw_idx(1):tw_idx(2))));
            lr_dat = squeeze(cat(1,h_dat(rl_idx,k,ao,:,tw_idx(1):tw_idx(2)),l_dat(rh_idx,k,ao,:,tw_idx(1):tw_idx(2))));
            
            hr_erp = squeeze(mean(hr_dat,2,'omitnan'));
            lr_erp = squeeze(mean(lr_dat,2,'omitnan'));
            dif_val_r = hr_erp - lr_erp;
            %check for nan values
            dif_val_r = dif_val_r(~isnan(dif_val_r(:,1)),:);
            %compute random t-statistics
            dif_t_r = zeros(1,size(dif_val_r,2));
            for l = 1:size(dif_val_r,2)
                dif_t_r(1,l) = mean(dif_val_r(:,l))/(std(dif_val_r(:,l))/sqrt(size(dif_val_r,1)));
            end
            lowb_r = abs(sum(dif_t_r(find(dif_t_r<t_val(1,1)))));
            highb_r = abs(sum(dif_t_r(find(dif_t_r>t_val(1,2)))));
            t0_dist(prm,1) = max([lowb_r highb_r]);
        end
        
        r_sort = sort(t0_dist);
        t_rej = length(t0_dist)*0.95;
        t_cv = r_sort(t_rej);
        
        sig_t(k,ao) = test_s > t_cv;
        
        nexttile
        dt = histogram(t0_dist,15)
        hold on
        tv = plot(test_s,[0 0],'d')
        hold on
        cv = xline(t_cv,'r')
        ylabel('cluster counts')
        xlabel('summed cluster t values')
        title(sprintf('%s, %s',task{k},auditory{ao}))
        legend('permutation',sprintf('critical value: %0.2f',t_cv),sprintf('t-value: %0.2f',test_s))
        
        
        
    end 
end


        
%% plotting

%plot this shit
figure
t1=tiledlayout(length(task),length(auditory))
for k =1:2
    for ao = 1:length(auditory)
        
      
        %prepare data
        h_plt = squeeze(mean(h_dat(:,k,ao,:,:),1,'omitnan'));
%         h_plt= bsxfun(@rdivide,h_plt, abs(mean(h_plt(:,base_idx(1):base_idx(2)))))
%         
        l_plt = squeeze(mean(l_dat(:,k,ao,:,:),1,'omitnan'));
        lc_plt= bsxfun(@rdivide,l_plt(8,:), abs(mean(l_plt(8,base_idx(1):base_idx(2)))))
        
        t4=nexttile(t1)
        plot(erp_time(:,win_idx(1):win_idx(2)),mean(h_plt(:,win_idx(1):win_idx(2)),1),'b')
        hold on
        plot(erp_time(:,win_idx(1):win_idx(2)),mean(l_plt(:,win_idx(1):win_idx(2)),1),'r')
        hold on
        title(t4,sprintf('%s %s',task{k},auditory{ao}))

        %plot sig        
        dif_ti = squeeze(dif_t(k,ao,:));
        t_val = tinv([0.025 0.975],size(dif_ti,1)-1);
        dif_tma = dif_ti>t_val(1,2);
        dif_tmi = dif_ti<t_val(1,1);        
        dif_sum = dif_tma + dif_tmi;
        
        %plot significance line
        if sig_t(k,ao)
            idx1 = find(diff([~dif_sum(1) dif_sum']));
            idx2 = circshift(idx1-1,[0 -1]);
            idx2(end) = numel(dif_sum);
            pcs = [idx1; idx2];
            
            pcs_a = [];
            %update pcs to only contain 1
            for p = 1:length(pcs)
                if any(find(dif_sum==1) == pcs(1,p)) && pcs(1,p) ~= pcs(2,p)
                    pcs_a=[pcs_a pcs(:,p)];
                end
            end
            
            %get min value
            y_min = min([mean(l_plt(:,win_idx(1):win_idx(2)),1) ; mean(h_plt(:,win_idx(1):win_idx(2)),1)],[],'all');
            figure('units','normalized','outerposition',[0 0 1 1])
            t2=tiledlayout(length(pcs_a)+1,2)
            t3 = nexttile(t2,[1 2])
            plot(erp_time(:,win_idx(1):win_idx(2)),mean(h_plt(:,win_idx(1):win_idx(2)),1),'b')
            hold on
            plot(erp_time(:,win_idx(1):win_idx(2)),mean(l_plt(:,win_idx(1):win_idx(2)),1),'r')
            xlabel(gca,'time (ms.)')
            ylabel(gca,'amplitude a.u.')
            title(t2,sprintf('%s %s',auditory{ao},task{k}))
            
            for pa = 1:size(pcs_a,2)
                
                p_time = t_erp_time(pcs_a(:,pa)');
                line(t3,[p_time(1,1) p_time(1,2)],[y_min-0.5 y_min-0.5],'linew',3,'color','k')
                line(t4,[p_time(1,1) p_time(1,2)],[y_min-0.5 y_min-0.5],'linew',3,'color','k')
                
                
                pa_idx = dsearchn(erp_time',p_time');
                %plot the topography
                nexttile(t2)
                topoplot(mean(l_plt(:,pa_idx(1,1):pa_idx(2,1)),2),EEG.chanlocs)
                title(sprintf('low dns Window [%d %d]',p_time(1,1),p_time(1,2)))
                nexttile(t2)
                topoplot(mean(h_plt(:,pa_idx(1,1):pa_idx(2,1)),2),EEG.chanlocs)
                title(sprintf('high dns Window [%d %d]',p_time(1,1),p_time(1,2)))
                
            end
        end
        
%         if ao == 5
%             set(t1,'Ylim',[-10 10])
%         else
%             set(t1,'Ylim',[-5 4])
%         end
        
        
        
    end
end

%% t-testing this biatch 
t_wins(1,1,:) = [70 110];
t_wins(1,2,:) = [60 100];
t_wins(1,3,:) = [30 130];
t_wins(1,4,:) = [100 180];

t_wins(2,1,:) = [70 110];
t_wins(2,2,:) = [110 160];
t_wins(2,3,:) = [40 140];
t_wins(2,4,:) = [70 140];

p=[]
for k=1:2
    for ao = 1:length(auditory)
        tw_idx = dsearchn(erp_time',squeeze(t_wins(k,ao,:)));

        %compute the t-statistic
        %1. get the mean difference between the condtions
        temp_dat = mean(mean(h_l_dif(:,k,ao,:,tw_idx(1,1):tw_idx(2,1)),5,'omitnan'),4);
        [h,p(k,ao)] = ttest(temp_dat);        
    end 
end

%correct for multiple testing
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p,0.05,'dep','yes')
           





            
