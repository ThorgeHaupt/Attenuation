OT_setup 
DNS_setup


%partition the data set
nfold = 12;
testfold = 1;

%number of bins
min_value = 1;       % Minimum value of the range
max_value = 1000;    % Maximum value of the range

% uniform distribution
numBins = linspace(2,8,7); % Number of bins

%load the distance collection vector
load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\dns_dist_descriptives.mat')

%clean it first
dns_dist(dns_dist > max_value) = [];


% uniform distribution
for i = 1:length(numBins)
    binEdges{i} = quantile(dns_dist, linspace(0, 1, numBins(i) + 1));
end


for s=1:length(sbj)
      
    
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab('nogui');

    for k=1:2
        
        %% compute the envelopes
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        eeg_sav{k} = EEG; 
        
        cd(PATH)
        
        
        novelty_ultm = load(sprintf('ons_ult_%s',task{k}));
        
        fs_new = EEG.srate;
        
        peak = smooth_peak(novelty_ultm.novelty_ultm,fs_new,'sigma',4);
        
        peak_sav{k} = peak;
        
    end
    
    EEG = pop_mergeset(eeg_sav{1},eeg_sav{2});
    
    peak = [peak_sav{1} peak_sav{2}]';
    
    %get the indicies
    ons_idx = find(peak);
    
    %find the distance
    ons_dif = diff(ons_idx);
    
    %remove the first one
    ons_idx(1) = [];
    
    %remove the onsets above the cutoff of
    ons_del = find(ons_dif > max_value);
    ons_idx(ons_del) = [];
    ons_dif(ons_del) = [];
    
    
    
%     for ed = 1:length(numBins)
%         
%         
%         [counts, bins, binIndices] = histcounts(ons_dif,binEdges{ed});
%         
%         %create the feature vector based on the random entries
%         ons_bin = zeros(length(counts),length(peak));
%         for i = 1:length(binIndices)
%             ons_bin(binIndices(i),ons_idx(i)) = 1;
%         end
%         
%         stim = ons_bin';
        stim = peak;
        
        %get the neural data
        resp = double(EEG.data');
        
        if size(resp,1)>size(stim,1)
            resp = resp(1:size(stim,1),:);
        elseif size(resp,1)<size(stim,1)
            stim = stim(1:length(resp),:);
        end
        
        %loop over the test folds
        temp_w = [];
        temp_r = [];
        avg_tr = [];
        for tr = 1:nfold
            
            
            %split the data
            [strain,rtrain,stest,rtest] = mTRFpartition(stim,resp,nfold,tr);
            
            strainz = strain; %normalization occurs at the stim extraction fun
            stestz = stest;   %normalization occurs at the stim extraction fun
            
            
            rtrainz = cellfun(@(x) zscore(x,0,'all'),rtrain,'UniformOutput',false);
            rtestz = zscore(rtest,[],'all');
            
            %% use cross-validation
            fs = EEG.srate;
            
            
            cv = mTRFcrossval(strainz,rtrainz,fs,Dir,tmin,tmax,lambdas,'Verbose',0);
            
            %get the optimal regression parameter
            l = mean(cv.r,3); %over channels
            [l_val,l_idx] = max(mean(l,1));
            l_opt = lambdas(l_idx);
            
            %save the lambda
            %             l_save(s,k,ao,:) = lambda_opt;
            
            %train the neural model on the optimal regularization parameter
            model_train = mTRFtrain(strainz,rtrainz,fs,Dir,tmin,tmax,l_opt,'verbose',0);
            %                 weights(s,k,ao,:) = squeeze(mean(model_train.w,3));
            
            
            %save the weights
            temp_w(:,:,:,tr) = model_train.w;
            
            %predict the neural data
            [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
            
            %save the prediction values
            temp_r(:,tr) = STATS.r;
            
%             fprintf('DIST GRID: sbj = %s, condition: %s; number bin: %d\r',sbj{s},task{k},length(counts))
        end
        mlpt_weight{s} = squeeze(mean(temp_w,4));
        result_reg(s,:) = squeeze(mean(temp_r,2));
        
    %end
end

%% save the results and the whole script as well 
trf_time = model_train.t;
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\')

DNS_uni = struct();
DNS_uni.result_reg = result_reg;
% DNS_uni.sav_count = sav_count;
DNS_uni.numBins= numBins;
DNS_uni.trf_time = trf_time;
DNS_uni.binEdges = binEdges;
% DNS_grid.auditory = auditory;

DNS_uni.t = 'combined the two conditions to obtain more trainings data ';

save('DNS_dist_long_single.mat','-struct','DNS_uni')

%% plot the fucking results
bin_isi = load('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_long_binisi.mat')
isi_dat = squeeze(mean(bin_isi.result_reg,3));

res = mean(result_reg,2)
%sort the rest
[reso_sort s_idx] = sort(res,'ascend')

isi_dats = isi_dat(s_idx,:);
fig_pos = [121,222,504,437];
%compare these bitches statistically 
figure, hold on
set(gcf,'position',fig_pos)
for ed = 1:length(numBins)
    
    p_val(ed) = signrank(isi_dats(:,ed),reso_sort);
    plot(isi_dats(:,ed),'o-','Color', gradientColors_ISI(ed,:),'linew',2)
end
plot(reso_sort,'o-','Color',[0, 0.5, 0.7410],'linew',2)
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_val,0.05,'dep');

xlabel('Subjects');
ylabel('Pearson''s r');
legend([num2str(numBins'); num2str(1)],'Location', 'NorthWest','Box','off');
grid off;
set(gca,'FontSize',14)

% save_fig(gcf,fig_path,'DNS_ISI_verylongmodel')




figure
violinplot([mean(result_reg,2) isi_dat])

for i = 1:length(numBins)
    [p,h,stats] = signrank(mean(result_reg,2),isi_dat(:,i));
    p_long(i) = p;
    zval(i) = stats.zval;
    w_val(i) = stats.signedrank;
end

[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_long,0.05,'dep')

ef_size = zval./sqrt(size(isi_dat,1))
