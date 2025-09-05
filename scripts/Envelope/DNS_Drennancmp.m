%% Model Comparison using variance paritioning
OT_setup

Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 6;
testfold = 1;
auditory = {'envelope'};
% auditory= {'melmenv','melonsmenv'}
for s=1:length(sbj)
    
    
    for k=1:2
        
        EEG = [];
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
%         
%         reg = zeros(length(auditory),EEG.nbchan);
%         raw = zeros(length(auditory),EEG.nbchan);
        
        
        %extract the stimulus
        menv = extract_stimulus2(EEG, PATH, auditory{:}, k,sbj{s},task);
        
        %normalize the envelope
        menv_norm = (menv - min(menv)) / ...
            (max(menv) - min(menv));
        
        % Define the bin edges in dB
        binEdges_dB = 40:8:80;  % Binning in 8 dB steps up to 64 dB (adjust as needed)
        nBins = length(binEdges_dB);
        
        % Convert dB edges to linear scale by taking 10^(binEdges_dB/20)
        binEdges_linear = 10.^(binEdges_dB / 20);
        
        % Normalize the bin edges to be between 0 and 1
        binEdges_normalized = (binEdges_linear - min(binEdges_linear)) / ...
            (max(binEdges_linear) - min(binEdges_linear));
        
        % Calculate the histogram counts and bin indices using histcounts
        [counts, ~, binIndices] = histcounts(menv_norm, binEdges_normalized);
        
        %save the counts
        sav_count{s,k} = counts;

        %binned envelope
        env_bin = zeros(length(counts),length(menv));
        
        for i = 1:length(binIndices)
            env_bin(binIndices(i),i) = menv_norm(i);
        end
        
        %normalize each bin
        env_bin_norm = normalize(env_bin,2,'range');

        stim_col = {menv,env_bin_norm'};
        
        for ao = 1:length(stim_col)
            stim = stim_col{1,ao};
            
            %get the neural data
            resp = double(EEG.data');
            
            if size(resp,1)>size(stim,1)
                resp = resp(1:size(stim,1),:);
            end
            
            [strain,rtrain,stest,rtest] = mTRFpartition(stim,resp,nfold,testfold);
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
            
            %save the weights
            if size(model_train.w,1)<2
                single_weight(s,k,:,:) = squeeze(model_train.w);
            else
                mlpt_weight{s,k} = model_train.w;
                                
            end

            %predict the neural data
            [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
            
            reg(ao,:) = STATS.r;
        end
        
        
        result_reg(s,k,:,:) = reg;
        %         result_raw(s,k,:,:) = raw;
    end
    
end

trf_time = model_train.t;
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results\')

%save it as a structure with the corresponding labels
analysis08_results = struct;
analysis08_results.result_reg = result_reg;
analysis08_results.auditory = auditory;
analysis08_results.single_weight = single_weight;
analysis08_results.mlpt_weight = mlpt_weight;
analysis08_results.sav_count = sav_count;
analysis08_results.binEdges_dB=binEdges_dB;



analysis08_results.trf_time = trf_time;
analysis08_results.auditory = auditory;

analysis08_results.descp = 'this analysis is the comparison to Drennans paper, including bins that also do not contain any information';
save('OT08_Drennancmp_without0.mat','-struct','analysis08_results')

load('OT08_Drennancmp_without0.mat')
%% plot the results

%set the figure path 
fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Drennancmp\'
typ = {'Envelope','AB envelope'};


%% Descriptives of the counts
counts = squeeze(cat(3,sav_count{:}));

figure
histogram('BinEdges',binEdges_dB,'BinCounts',mean(counts,2))
set(gca,'view',[90 -90])
ylabel('Count (Samples)')
xlabel('Amplitude Bins (dB)')
set(gca,'FontSize',14)
box off

save_fig(gcf,fig_path,'Descript_cmp_without0')





%% Prediction accuracy
temp_dat = squeeze(mean(mean(result_reg,2),4));
Env = temp_dat(:,1);
[Env sort_idx] = sort(Env,'ascend');
AB_Env = temp_dat(:,2);
AB_Env = AB_Env(sort_idx);

%test the two 
[p,h] = signrank(Env,AB_Env)

fig_pos = [548   333   692   645];

figure;
tiledlayout(1,2)
nexttile
hold on;

plot(Env, '-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Env');
plot(AB_Env, '-o', 'Color', [0.49, 0.18, 0.56], 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'AB Env');
% Customize the plot
xlabel('Subjects');
ylabel('Pearson''s r');
legend('Location', 'NorthWest');
grid on;
set(gca,'FontSize',14)


nexttile 
violinplot([Env AB_Env],typ,...
    'ViolinColor',[0, 0.45, 0.74;0.49, 0.18, 0.56])
box off
sigstar([1,2],p)
set(gca,'FontSize',14)

set(gcf,'Position',fig_pos);

save_fig(gcf,fig_path,'prediction_cmp_without0')


%% topos
temp_dat = squeeze(mean(mean(result_reg,2),1));

for i = 1:size(temp_dat,1)
    figure
    topoplot(temp_dat(i,:),EEG.chanlocs)
    title(typ{i})
    set(gca,'FontSize',14)

    save_fig(gcf,fig_path,sprintf('topo_%s_without0',typ{i}))
end


%% plot the weights

% figure
% plot(trf_time,squeeze(mean(mean(mean(single_weight,2),4),1)),'Color',[0, 0.45, 0.74],'linew',2)
% box off
% xlabel('Time Lag (ms) ')
% ylabel('a.u.')
% set(gca,'FontSize',14)
% title('Envelope Model','FontSize',18)
% 
% save_fig(gcf,fig_path,'env_model')


weight_nw = mlpt_weight{:,1};
weight_we = mlpt_weight{:,2};
combinedArray = cat(5,mlpt_weight{:,1} ,mlpt_weight{:,2});
combinedArray = reshape(combinedArray,size(weight_nw,1),size(weight_nw,2),size(weight_nw,3),2,length(sbj));
combinedArray = permute(combinedArray,[5 4 1 2 3]);
% temp_dat = squeeze(mean(mean(combinedArray,4),3));
temp_dat = squeeze(mean(mean(combinedArray,2),5));
bin_plot = binEdges_dB
%remove the 0 
bin_plot(1) = [];


for i=1:size(temp_dat,2)
    figure
    plot(squeeze(temp_dat(:,i,:))','Color',[1-(i/10) 0+(i/10) 1])
    set(gca,'Ylim',[-130 130])
end


%find the N1 and P2 peaks for the different bins 
N1 = [10 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

[n1_peak,n1_lat] = min(temp_dat(:,:,n1_idx(1):n1_idx(2)),[],3);
[p2_peak,p2_lat] = max(temp_dat(:,:,p2_idx(1):p2_idx(2)),[],3);

n1_lat = round(mean(n1_lat));
p2_lat = round(mean(p2_lat));

n1_peak = mean(n1_peak);
p2_peak = mean(p2_peak);

n1_time = trf_time(n1_idx(1):n1_idx(2));
p2_time = trf_time(p2_idx(1):p2_idx(2));


plot_dat = squeeze(mean(temp_dat,1));

N1 = [10 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

[n1_peak,n1_lat] = min(plot_dat(:,n1_idx(1):n1_idx(2)),[],2);
[p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);

% n1_lat = round(mean(n1_lat));
% p2_lat = round(mean(p2_lat));
% 
% n1_peak = mean(n1_peak);
% p2_peak = mean(p2_peak);

n1_time = trf_time(n1_idx(1):n1_idx(2));
p2_time = trf_time(p2_idx(1):p2_idx(2));




figure
t = tiledlayout(2,2)
nexttile
for i =1:size(temp_dat,2)
    plot(trf_time,plot_dat(i,:)+(i*10),'Color',[0 0.4470 0.7410],'linew',2)
    hold on
end
xlabel('Time Lag (ms)');
ylabel('Density Bins (dB)');
grid on 
set(gca,'FontSize',16)
box off

nexttile
imagesc(trf_time,bin_plot,plot_dat)
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
set(gca,'YDir','normal') 
set(gca,'FontSize',16)
box off

nexttile
plot(n1_time(n1_lat),bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([min(n1_time(n1_lat))-5 max(n1_time(n1_lat))+5])
% Adding a title or label similar to the "C"
title('N1 Peak latency')
set(gca,'FontSize',16,'YTick',linspace(1,n,n),'YtickLabel',binEdges_dB(2:end))
box off

% load('DNS_CI_rand_env_pp.mat');
% %get the 97%interval going
% l_bound = prctile(data,2.5);
% u_bound = prctile(data,97.5);
% m_dat = mean(data);



nexttile
plot((p2_peak-n1_peak), bin_plot, '-o', 'LineWidth', 1.5);
% hold on
% errorbar(m_dat, bin_plot, (l_bound'-m_dat'), 'horizontal', '-o', 'MarkerSize', 6, 'LineWidth', 1.5, 'CapSize', 10);

% Labeling the axes
xlabel('Magnitude a.u.');
ylabel('Amplitude Bins (dB)');
grid on;
ylim([bin_plot(1)-5 bin_plot(end)+5])
% xlim([15 45]);
% Adding a title or label similar to the "C"
title('N1-P2 peak to peak')
set(gca,'FontSize',16)
box off

title(t,'DNS based TRF','FontSize',28)
set(gca,'FontSize',16)

save_fig(gcf, fig_path,'Weight_cmp_without0')




