%% DNS analysis, but random
OT_setup 

%TRF parameters
Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 6;
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

% % Bin the data
% binIndices = discretize(dns_dist, binEdges{i});
% [counts, ~, binIndices_env] = histcounts(dns_dist, binEdges{i});
% 
% figure 
% histogram(binIndices, numBins, 'Normalization', 'probability');
% xlabel('Bins');
% ylabel('Probability');
% title('Uniformly Distributed Bins');
% 
% figure
% histogram('BinEdges',binEdges{i},'BinCounts',counts)
% set(gca,'view',[90 -90])
% ylabel('Count (Samples)')
% xlabel('Amplitude Bins (dB)')
% set(gca,'FontSize',14)
% box off

ran_fold = 100;

for s=1:length(sbj)
    
    for k=1:2
        
        %% compute the envelopes
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        cd(PATH)
        
        
        novelty_ultm = load(sprintf('ons_ult_%s',task{k}));
        
        fs_new = EEG.srate;
        
        peak = smooth_peak(novelty_ultm.novelty_ultm,fs_new,'sigma',4);
        
        
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
        
        
        
        for ed = 1:length(binEdges)
            temp_w = [];

            
            [counts, bins, binIndices] = histcounts(ons_dif,binEdges{ed});
            
            %% let the randomization begin
            for rn = 1:ran_fold
                
                %randomize the indices
                binIndices_rand = binIndices(randperm(length(binIndices)));
                
                %create the feature vector based on the random entries
                ons_bin = zeros(length(counts),length(peak));
                for i = 1:length(binIndices)
                    ons_bin(binIndices_rand(i),ons_idx(i)) = 1;
                end
                
                stim = ons_bin';
                
                %get the neural data
                resp = double(EEG.data');
                
                if size(resp,1)>size(stim,1)
                    resp = resp(1:size(stim,1),:);
                elseif size(resp,1)<size(stim,1)
                    stim = stim(1:length(resp),:);
                end
                
                %loop over the test folds
                temp_r = [];
                for tr = 1:nfold
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
                    temp_w(:,:,:,tr,rn) = model_train.w;
                    
                    %predict the neural data
                    [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
                    
                    %save the prediction values
                    temp_r(:,tr) = STATS.r;
                    
                    fprintf('DIST GRID: sbj = %s, condition: %s; number bin: %d rand_iter: %d\r',sbj{s},task{k},length(counts),rn)
                end
                
                result_reg(s,k,ed,rn,:) = squeeze(mean(temp_r,2));
            end
            mlpt_weight{s,k,ed} = squeeze(mean(temp_w,4));

        end
    end
end

trf_time = model_train.t;
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\')

DNS_uni = struct();
DNS_uni.result_reg = result_reg;
DNS_uni.mlpt_weight = mlpt_weight;
% DNS_uni.sav_count = sav_count;
DNS_uni.numBins= numBins;
DNS_uni.trf_time = trf_time;
DNS_uni.binEdges = binEdges;
% DNS_grid.auditory = auditory;

DNS_uni.t = 'Here we applied a uniform distribution contstraint to the estimation optimal distance parameters randomizing the index values ';

save('DNS_dist_uniform_fulltr_rand.mat','-struct','DNS_uni')

