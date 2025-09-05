%% Test merely the onsets by themselves to get a frame of reference
%global paths
OT_setup 

%TRF parameters
Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 6;
testfold = 1;


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
        
        
        
        stim = peak';
        
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
cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\')

DNS_SingleOnsets = struct();
DNS_SingleOnsets.reg = squeeze(mean(reg,3));
DNS_SingleOnsets.mlpt_weight = mlpt_weight;
DNS_SingleOnsets.trf_time= trf_time;
DNS_SingleOnsets.t = 'singular model of the onsets';


save('DNS_SingleOnsets_trrepeat.mat','-struct','DNS_SingleOnsets')



%% plot the results
fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\SingleOnsets'

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

% save_fig(gcf,fig_path,'SingleOnsets')





