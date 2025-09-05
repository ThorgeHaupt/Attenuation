%% test whether these ISI binning does not work due to increased model dimensionality 
% i am gonna replace the binary vector by a weighted combination

OT_setup 
DNS_setup


%partition the data set
nfold = 6;
testfold = 1;

%number of bins
min_value = 1;       % Minimum value of the range
max_value = 1000;    % Maximum value of the range

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
        
        ons_bin = zeros(size(peak));
        
        
        ons_bin(ons_idx) = 1;

        stim = ons_bin';
        
        %normalize the ons_diff values
        ons_dif_norm = normalize(ons_dif,'range');
        ons_bin = zeros(size(peak));
        
        ons_bin(ons_idx) = ons_dif_norm;
        
        %get the neural data
        resp = double(EEG.data');
        
        if size(resp,1)>size(stim,1)
            resp = resp(1:size(stim,1),:);
        elseif size(resp,1)<size(stim,1)
            stim = stim(1:length(resp),:);
        end
        
        stim = [stim ons_bin(:,1:length(stim))'];
        
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
            
            
            mlpt_weight(s,k,tr,:,:,:) = model_train.w;
            
            %predict the neural data
            [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
            
            reg(s,k,tr,:) = STATS.r;
            
            
            fprintf('Participant %s, condition %s',sbj{s},task{k})
            
        end
    end
    
end

trf_time = model_train.t;
cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\')

DNS_SingleOnsets = struct();
DNS_SingleOnsets.reg = squeeze(mean(reg,3));
DNS_SingleOnsets.mlpt_weight = mlpt_weight;
DNS_SingleOnsets.trf_time= trf_time;
DNS_SingleOnsets.t = 'singular model of the onsets, with weighted onsets';


save('DNS_ISI_SingleOnsets_cleaned+weighted.mat','-struct','DNS_SingleOnsets')



%% plot the results
fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\SingleOnsets\'

temp_pred = squeeze(mean(reg,[2 3 4]));

figure
t = tiledlayout(1,2)
nexttile
violinplot(temp_pred,{'acoustic onsets'},...
    'ViolinColor',audi_colorsrgb('onset'),...
    'ViolinAlpha',0.5)
ylabel('Prediction Accuracy')
set(gca,'FontSize',16)

nexttile
weight_dat = squeeze(mean(mlpt_weight,[1 2 3 6]));
plot(trf_time,squeeze(mean(weight_dat,[3 4])),'Color',audi_colorsrgb('onset'),'linew',2)
ylabel('a.u.')
xlabel('Time Lags in ms.')
set(gca,'FontSize',16)
title(t,'Only Onsets')

save_fig(gcf,fig_path,'Single_onsets_clean')

%% contrast the results

single_ons = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_ISI_SingleOnsets_cleaned.mat')
single_dat = squeeze(mean(single_ons.reg,[2 3]))

figure
t = tiledlayout(1,2)
nexttile
violinplot([temp_pred single_dat],{'ons+weighted','ons'},...
    'ViolinColor',[0.9290    0.6940    0.1250;audi_colorsrgb('onset')],...
    'ViolinAlpha',0.5)
ylabel('Prediction Accuracy')
set(gca,'FontSize',14)
box off

p = signrank(temp_pred,single_dat)
sigstar({[1,2]},p)

nexttile
weight_dat = squeeze(mean(mlpt_weight,[1 2 3 6]));
p0= plot(trf_time,squeeze(mean(weight_dat,[3 4])),'Color',[0.9290    0.6940    0.1250],'linew',2)
hold on 
p1 = plot(trf_time,squeeze(mean(single_ons.mlpt_weight,[1 2 3 6])),'Color',audi_colorsrgb('onset'),'linew',2)
ylabel('a.u.')
box off
legend([p0(1);p1(1)],{'ons+weighted','ons'},'Box','off')
xlabel('Time Lags in ms.')
set(gca,'FontSize',14)
title(t,'Only Onsets')

save_fig(gcf,fig_path,'DNS_ISI_clean+weightesingleons')



