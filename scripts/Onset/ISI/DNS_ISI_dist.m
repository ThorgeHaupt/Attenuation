%% test out the different onset methods to obtain a somewhat stable measure of the sound events

MAINPATH = 'O:\projects\thh_ont\auditory-attention-in-complex-work-related-auditory-envrionments\data files'
addpath(genpath(MAINPATH));

OT_setup

Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 6;

testfold = 1;
auditory = {'envelope onset'};

% sbj =    {'P001', 'P002','P003','P006', 'P006','P007','P008','P009',...
%     'P010','P012','P013', 'P014','P016','P016','P017', 'P018',...
%     'P019','P020','P021','P022'};

 for s=1:length(sbj)

    for k=1:2
        
        %% compute the envelopes
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);

        cd(PATH)
        
        if k == 1
            wav = load(sprintf('%s_narrow_audio_strct.mat',sbj{s}));
        else
            wav = load(sprintf('%s_wide_audio_strct.mat',sbj{s}));
        end
        
        %% compute the onsets
        
        %energy novelty
        [novelty_enrgy, fs_new] = energy_novelty(double(wav.audio_strct.data)',wav.audio_strct.srate,'H',441);
                
        %complex novelty
        [novelty_cplx, fs_new] = complex_novelty(double(wav.audio_strct.data)',wav.audio_strct.srate,'H',441,'N',882);
        
        %phase novelty
        [novelty_phs, fs_new] = phase_novelty(double(wav.audio_strct.data)',wav.audio_strct.srate,'H',441,'N',882);

        %spectral novelty
        [novelty_spec, fs_new] = spectral_novelty(double(wav.audio_strct.data)',wav.audio_strct.srate,'H',441,'N',882);
        
        
        %average the three together for maximal information gain
        len = min([length(novelty_spec),length(novelty_spec),length(novelty_spec)]);
        novelty_ultm =mean(cat(1,novelty_spec(1,1:len), novelty_cplx(1,1:len), novelty_enrgy(1,1:len)),1);

        sec = 40;
        wav_time = linspace(0,EEG.xmax,length(wav.audio_strct.data));
        
        figure
        t = tiledlayout(5,1)
        nexttile
        plot(wav_time,wav.audio_strct.data)
        set(gca,'Xlim',[0 sec], 'XTickLabel', [])
        title('Audio')
        box off
        set(gca,'FontSize',14)
        
        
        times = linspace(0,wav.audio_strct.xmax,length(novelty_enrgy));
        nexttile
        plot(times, novelty_enrgy,'linew',2)
        title('Energy Novelty')
        set(gca,'Xlim',[0 sec], 'XTickLabel', [])
        box off
        set(gca,'FontSize',14)
        
        times = linspace(0,wav.audio_strct.xmax,length(novelty_spec));
        nexttile
        plot(times,novelty_spec,'linew',2)
        title('Spectral Novelty')
        set(gca,'Xlim',[0 sec], 'XTickLabel', [])
        box off
        set(gca,'FontSize',14)

        times = linspace(0,wav.audio_strct.xmax,length(novelty_cplx));
        nexttile
        plot(times,novelty_cplx,'linew',2)
        title('Complex Novelty')
        set(gca,'Xlim',[0 sec], 'XTickLabel', [])
        box off
        set(gca,'FontSize',14)
% 
%         times = linspace(0,wav.audio_strct.xmax,length(novelty_phs));
%         nexttile
%         plot(times,novelty_phs)
%         title('phase')
%         set(gca,'Xlim',[0 sec])
%         box off
%         set(gca,'FontSize',14)
        
        
        
        times = linspace(0,wav.audio_strct.xmax,length(novelty_ultm));
        nexttile
        plot(times,novelty_ultm,'linew',2)
        hold on 
        plot(times,peak,'linew',2)
        hold on 
%         plot(times,threshold_local)
        title('Combined Novelty')
        set(gca,'Xlim',[0 sec])
        ylim([0 0.4])
        box off
        set(gca,'FontSize',14)
        
        xlabel(t,'Time in s. ','FontSize',16)
        ylabel(t,'a.u.','FontSize',16)
        
        
        save_fig(gcf,fig_path,'method_novelty')
%       


        [peak,threshold_local] = smooth_peak(novelty_ultm,fs_new,'sigma',12,'plt',1);
        
       
       
        %get the indicies
        ons_idx = find(peak);
        
        %find the distance
        ons_dif = diff(ons_idx);
        
        %set the bin Edges
        max_dif = max(ons_dif)
        min_dif = min(ons_dif)
        
        
        
        
        
        %bin the distances
        [counts, binEdges_dB, binIndices] = histcounts(ons_dif,50);
      
        figure
        histogram('BinEdges',binEdges_dB,'BinCounts',counts)
        set(gca,'view',[90 -90])
        
        %remove first ons
        ons_idx(1) = [];
        
        ons_bin = zeros(length(binEdges_dB),length(peak));
        for i = 1:length(binIndices)
            ons_bin(binIndices(i),ons_idx(i)) = 1;
        end

        
        %exclude 0 entries
%         ons_bin(sum(ons_bin,2) == 0,:) = [];
%         binEdges_dB(sum(ons_bin,2) == 0) = [];
        
        stim_col = {peak',ons_bin'};
        
        for ao = 1:length(stim_col)
            stim = stim_col{1,ao};
            
            %get the neural data
            resp = double(EEG.data');
            
            if size(resp,1)>size(stim,1)
                resp = resp(1:size(stim,1),:);
            elseif size(resp,1)<size(stim,1)
                stim = stim(1:length(resp),:);
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
            %                 weights(s,k,ao,:) = squeeze(mean(model_train.w,3));
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
        
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results')
analysis08_results = struct;
analysis08_results.result_reg = result_reg;
analysis08_results.auditory = auditory;
analysis08_results.single_weight = single_weight;
analysis08_results.mlpt_weight = mlpt_weight;
analysis08_results.trf_time = trf_time;
analysis08_results.descp = 'sorted according to DNS based on improved onset detection';
save('DNS_ISI_dist.mat','-struct','analysis08_results')  
     


% plot the results
temp_dat = squeeze(mean(mean(result_reg,2),4));
Env = temp_dat(:,1);
AB_Env = temp_dat(:,2);
figure;
hold on;

plot(Env, '-o', 'Color', [0, 0.45, 0.74], 'MarkerFaceColor', 'auto', 'DisplayName', 'Env');
plot(AB_Env, '-o', 'Color', [0.49, 0.18, 0.56], 'MarkerFaceColor', 'auto', 'DisplayName', 'AB Env');
% Customize the plot
xlabel('Subjects');
ylabel('Pearson''s r');
legend('Location', 'NorthWest');
title('Plot B');
grid on;

temp_dat = squeeze(mean(mean(result_reg,2),1));

for i = 1:size(temp_dat,1)
    figure
    topoplot(temp_dat(i,:),EEG.chanlocs)
end


%% plot the weights
figure
plot(trf_time,squeeze(mean(mean(mean(single_weight,2),4),1)))

weight_nw = mlpt_weight{:,1};
weight_we = mlpt_weight{:,2};
combinedArray = cat(5,mlpt_weight{:,1} ,mlpt_weight{:,2});
combinedArray = reshape(combinedArray,size(weight_nw,1),size(weight_nw,2),size(weight_nw,3),2,length(sbj));
combinedArray = permute(combinedArray,[5 4 1 2 3]);
% temp_dat = squeeze(mean(mean(combinedArray,4),3));
temp_dat = squeeze(mean(mean(combinedArray,2),5));
bin_plot = binEdges_dB/EEG.srate;


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


%% plot the min and max values
figure
tiledlayout(2,2)
nexttile
plot(n1_time(n1_lat), bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('DNS Bins (dB)');
grid on;
xlim([-50 200]);
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('N1 Peak latency')

nexttile
plot(p2_time(p2_lat), bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([-50 200]);
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('P2 Peak latency')

nexttile
plot(n1_peak, bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Peak Amplitude');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([-22 1]);
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('N1 Peak amplitude')

nexttile
plot(p2_peak, bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Peak Amplitude');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([1 22]);
% Adding a title or label similar to the "C"
text(-90, 75, 'C', 'FontSize', 14, 'FontWeight', 'bold');  % Label 'C' in top-left
title('P2 Peak amplitude')


%% replicate the exact figure
plot_dat = squeeze(mean(temp_dat,1));
figure
t = tiledlayout(2,2)
nexttile
for i =1:size(plot_dat,1)
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
plot(n1_time(n1_lat), bin_plot, '-o', 'LineWidth', 1.5);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
grid on;
% ylim([-0.1 1.1])
xlim([0 400])
% Adding a title or label similar to the "C"
title('N1 Peak latency')
set(gca,'FontSize',16)
box off

load('DNS_CI_rand_env_pp.mat');
%get the 97%interval going
l_bound = prctile(data,2.5);
u_bound = prctile(data,97.5);
m_dat = mean(data);
nexttile
plot((p2_peak-n1_peak), bin_plot, '-o', 'LineWidth', 1.5);
hold on
errorbar(m_dat, bin_plot, (l_bound'-m_dat'), 'horizontal', '-o', 'MarkerSize', 6, 'LineWidth', 1.5, 'CapSize', 10);

% Labeling the axes
xlabel('Magnitude a.u.');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([15 45]);
ylim([-0.1 1.1])
% Adding a title or label similar to the "C"
title('N1-P2 peak to peak')
set(gca,'FontSize',16)
box off

title(t,'DNS based TRF','FontSize',28)
set(gca,'FontSize',16)

% save_fig(gcf,fig_path,'DNS_env')

%% 
temp_dat = flip(temp_dat,1);
figure;hold on;
tiledlayout(size(temp_dat,1),1)
for i =1:size(temp_dat,1)
    nexttile
    plot(model_train.t,temp_dat(i,:))
    axis off
    box off
end

%% per participant
temp_dat = squeeze(mean(mean(mlpt_weight,2),5));
bin_plot = binEdges_dB;
bad_idx = sum(temp_dat,3) ==0
bin_plot(:,bad_idx(1,:)) = [];
temp_dat(:,bad_idx(1,:),:) = [];


%find the N1 and P2 peaks for the different bins 
N1 = [30 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

[n1_peak,n1_lat] = squeeze(mean(min(temp_dat(:,:,n1_idx(1):n1_idx(2)),[],3)));
[p2_peak,p2_lat] = squeeze(mean(max(temp_dat(:,:,p2_idx(1):p2_idx(2)),[],3)));

n1_time = trf_time(n1_idx(1):n1_idx(2));
p2_time = trf_time(p2_idx(1):p2_idx(2));
        
        
        
        