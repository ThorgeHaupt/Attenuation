%% DNS_OnsRateofChange
%goal of this script is to sort the onsets according to the rate of change in
%the signal -> grid search to derive the optimal parameters and compare it
%the singular model. 

%global paths
OT_setup 

%TRF parameters
Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 10;
testfold = 1;

%lower bound
lo_bound = [0 8 16 24 32 40];

%upper bound
up_bound = [72 80 88 96 104 112 120];

%bin width 
bin_width = [4 8 12 16 24];


%start the loop 
auditory = {'envelope onset'};

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
                    
        %extract the stimulus
        menv = extract_stimulus2(EEG, PATH, auditory{:}, k,sbj{s},task);
        
        %normalize the envelope
        menv_norm = (menv - min(menv)) / ...
            (max(menv) - min(menv));
        
        %start the trifecta of happiness
        for lo = 1:length(lo_bound)
            
            for up = 1:length(up_bound)
                
                for bw = 1:length(bin_width)
                    %rather than binning the envelope -> get the values of
                    %the envelope onset at the onsets indicies and bin those
                    
                    %get the onset peaks
                    ons_idx = find(peak);
                    
                    %normalize those values -> will that lead to a
                    %different distribution accorss participants?
                    ons_env = menv_norm(ons_idx);
                    
%                     ons_env_norm = normalize(ons_env,'range');

                    % Define the bin edges in dB
                    binEdges_dB = lo_bound(lo):bin_width(bw):up_bound(up);  % Binning in 8 dB steps up to 64 dB (adjust as needed)
                    nBins = length(binEdges_dB);
                    
                    % Convert dB edges to linear scale by taking 10^(binEdges_dB/20)
                    binEdges_linear = 10.^(binEdges_dB / 20);
                    
                    % Normalize the bin edges to be between 0 and 1
                    binEdges_normalized = (binEdges_linear - min(binEdges_linear)) / ...
                        (max(binEdges_linear) - min(binEdges_linear));
                    
                    % Calculate the histogram counts and bin indices using histcounts
                    [counts, ~, binIndices] = histcounts(ons_env_norm, binEdges_normalized);
                    
%                     figure
%                     histogram('BinEdges',binEdges_normalized,'BinCounts',counts)
%                     set(gca,'view',[90 -90])
%                     ylabel('Count (Samples)')
%                     xlabel('Amplitude Bins (dB)')
%                     set(gca,'FontSize',14)
%                     box off
%                     
                    %save the counts
                    sav_count{s,k,lo,up,bw} = counts;
                    
                    %binned onsets
                    ons_bin = zeros(length(counts),length(peak));
                    
                    %!!! instead of binning the envelope we now take the
                    %onsets into considerations as well 
                    for i = 1:length(binIndices)
                        ons_bin(binIndices(i),ons_idx(i)) = 1;
                    end
                    
                    stim = ons_bin';
                    
                    
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
                    

                    mlpt_weight{s,k,lo,up,bw} = model_train.w;

                    %predict the neural data
                    [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
                    
                    reg(s,k,lo,up,bw,:) = STATS.r;
                
                
                    fprintf('Participant %s, condition %s low bound: %d; upper bound: %d; bin width: %d\r',sbj{s},task{k},lo_bound(lo),up_bound(up),bin_width(bw))
                   
                end
            end
        end
    end
    
end

trf_time = model_train.t;
cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\')

DNS_RateoChange = struct();
DNS_RateoChange.reg = reg;
DNS_RateoChange.mlpt_weight = mlpt_weight;
DNS_RateoChange.sav_count = sav_count;
DNS_RateoChange.auditory = auditory;
DNS_RateoChange.trf_time= trf_time;



DNS_RateoChange.t = 'Here we applied a grid search to the estimation of the AB envelope modle according to Drennan';

save('DNS_RateoChange.mat','-struct','DNS_RateoChange')

%% start plotting i guess
%% find the best prediction accuracy

fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\RateofChange\';

fig_pos = [448   293   792   685];

trf_time = tmin:10:tmax
temp_dat = squeeze(mean(reg,[1 2 6]));

[rows, cols, pages] = ind2sub(size(temp_dat),find(temp_dat == max(temp_dat,[],'all')));

%% get the corresponding bin size

best_lo = lo_bound(rows);

best_up = up_bound(cols);

best_width = bin_width(pages);

%remake the bin width 
binEdges_dB = best_lo:best_width:best_up;  % Binning in 8 dB steps up to 64 dB (adjust as needed)
nBins = length(binEdges_dB);

% Convert dB edges to linear scale by taking 10^(binEdges_dB/20)
binEdges_linear = 10.^(binEdges_dB / 20);

% Normalize the bin edges to be between 0 and 1
binEdges_normalized = (binEdges_linear - min(binEdges_linear)) / ...
    (max(binEdges_linear) - min(binEdges_linear));

%% get the counts
counts = squeeze(cat(3,sav_count{:,:,rows, cols, pages}));
counts = sum(counts,2);

% bin value
figure
set(gcf,'position',fig_pos)
nexttile
histogram('BinEdges',binEdges_dB,'BinCounts',counts)
% b_log = bar(bin_centers,log_count,'FaceColor','flat')
ylabel('Count (Samples)')
xlabel('Onset Rate of Change')
title('AB env Grid Counts')
set(gca,'FontSize',14,'view',[90 -90])
box off

save_fig(gcf,fig_path,'DNS_RateofChange_Counts')

%% compare prediction accuracies

cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results\')
ons_pred = load('OT08_Drennancmp_with0.mat','result_reg');
ons_pred = squeeze(mean(ons_pred.result_reg(:,:,1,:),[2 4]));

ABenv_pred = squeeze(mean(reg(:,:,rows,cols,pages,:),[2 6]));

[AB_env_sort,s_idx] = sort(ABenv_pred,'ascend');
env_pred_sort = ons_pred(s_idx);

figure;
hold on
set(gcf,'position',fig_pos)

plot(env_pred_sort, '-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons');
plot(AB_env_sort, '-o', 'Color', [0.49, 0.18, 0.56], 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'AB Env');
% Customize the plot
xlabel('Subjects');
ylabel('Pearson''s r');
legend('Location', 'NorthWest');
grid on;
set(gca,'FontSize',14)

save_fig(gcf,figure_path,'DNS_RateofChange_Prediction')

%% get the corresponding weights
figure
set(gcf,'position',fig_pos)
%plot the weights
t = tiledlayout(2,2)
weight_temp = []
for k = 1:length(task)
    weight_temp(:,:,:,:,k) = squeeze(cat(4,mlpt_weight{:,k,rows,cols,pages}));
end
plot_dat = squeeze(mean(weight_temp,[3 4 5]))';
nexttile
for i =1:size(plot_dat,2)
    plot(trf_time,plot_dat(:,i)+(i*14),'Color',[0 0.4470 0.7410],'linew',2)
    y_val(i) = mean(plot_dat(1:10,i)+(i*14));
    hold on
end
grid on
title('Model Weights')
set(gca,'FontSize',16,'YTick',round(y_val),'YtickLabel',binEdges_dB(2:end))
box off


%plot the imagesc of the weights
nexttile
imagesc('XData',trf_time,'Ydata',binEdges_dB,'CData',plot_dat')
yticks(binEdges_dB)
set(gca, 'YDir', 'normal','FontSize',16)
axis tight
title('Model Weights')


xlabel(t,'Time Lag (ms)');
ylabel(t,'Density Bins (dB)');

%plot the latency shift and peak to peak amplitude
%find the N1 and P2 peaks for the different bins 
N1 = [10 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

plot_dat = plot_dat';
[n1_peak,n1_lat] = min(plot_dat(:,n1_idx(1):n1_idx(2)),[],2);
[p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);

n1_time = trf_time(n1_idx(1):n1_idx(2));
p2_time = trf_time(p2_idx(1):p2_idx(2));



nexttile
plot(n1_time(n1_lat),binEdges_dB(2:end), '-o', 'LineWidth', 1.5);
n = length(binEdges_dB(2:end));
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([min(n1_time(n1_lat))-5 max(n1_time(n1_lat))+5])
ylim([min(binEdges_dB(2:end))-5 max(binEdges_dB(2:end))+5] )
% Adding a title or label similar to the "C"
title('N1 Peak latency')
set(gca,'FontSize',16,'YTick',binEdges_dB(2:end),'YtickLabel',binEdges_dB(2:end))
box off

nexttile
plot((p2_peak-n1_peak), binEdges_dB(2:end), '-o', 'LineWidth', 1.5);

% Labeling the axes
xlabel('Magnitude a.u.');
ylabel('Amplitude Bins (dB)');
grid on;
xlim([min(p2_peak-n1_peak)-5 max(p2_peak-n1_peak)+5])
ylim([min(binEdges_dB(2:end))-5 max(binEdges_dB(2:end))+5] )
% xlim([15 45]);
% Adding a title or label similar to the "C"
title('N1-P2 peak to peak')
set(gca,'FontSize',16,'YTick',binEdges_dB(2:end),'YtickLabel',binEdges_dB(2:end))
box off

title(t,'DNS based TRF','FontSize',28)
set(gca,'FontSize',16)


save_fig(gcf,figure_path,'DNS_RateofChange_Weights')



%% compare for the two conditions separately

cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\')
ons_preds = load('DNS_SingleOnsets.mat','reg');
ons_pred = squeeze(mean(ons_preds.reg(:,:,:),[2 3]));
for k = 1:length(task)
%     ons_pred = squeeze(mean(ons_preds.reg(:,k,1,:), 4));
    
    ABenv_pred = squeeze(mean(reg(:,k,rows,cols,pages,:), 6));
    
%     [AB_env_sort,s_idx] = sort(ABenv_pred,'ascend');
%     env_pred_sort = ons_pred(s_idx);
    
    figure;
    set(gcf,'position',fig_pos)
    hold on
    
    plot(ons_pred, '-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Env');
    plot(ABenv_pred, '-o', 'Color', [0.49, 0.18, 0.56], 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'AB Env');
    % Customize the plot
    xlabel('Subjects');
    ylabel('Pearson''s r');
    legend('Location', 'NorthWest','box','off');
    grid on;
    title(sprintf('Prediction Accuracy %s',task{k}))
    set(gca,'FontSize',14)
    [p h] = signrank(ons_pred,ABenv_pred);
    if p <0.05 && p > 0.01
        text(length(sbj), max(AB_env_sort), '*', 'FontSize', 18, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
    elseif p < 0.01 && p > 0.000
        text(length(sbj), max(AB_env_sort), '**', 'FontSize', 18, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
    elseif p< 0.000
        ext(length(sbj), max(AB_env_sort), '***', 'FontSize', 18, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
    end
    
%     save_fig(gcf,figure_path,sprintf('DNS_RateofChange_Prediction_%s',task{k}))
end

%compute the different weigths
for k = 1:length(task)
    figure
    set(gcf,'position',fig_pos)
    t = tiledlayout(2,2)

    weight_temp = squeeze(cat(4,mlpt_weight{:,k,rows,cols,pages}));
    plot_dat = squeeze(mean(weight_temp,[3 4]))';
    
    nexttile
    for i =1:size(plot_dat,2)
        plot(trf_time,plot_dat(:,i)+(i*6),'Color',[0 0.4470 0.7410],'linew',2)
        y_val(i) = mean(plot_dat(:,i)+(i*6));
        hold on
    end
    grid on
    title('Model Weights')
    set(gca,'FontSize',16,'YTick',round(y_val),'YtickLabel',binEdges_dB(2:end))
    box off
    
    
    %plot the imagesc of the weights
    nexttile
    imagesc('XData',trf_time,'Ydata',binEdges_dB,'CData',plot_dat')
    yticks(binEdges_dB)
    set(gca, 'YDir', 'normal','FontSize',16)
    axis tight
    title('Model Weights')
    
    
    xlabel(t,'Time Lag (ms)');
    ylabel(t,'Density Bins (dB)');
    
    %plot the latency shift and peak to peak amplitude
    %find the N1 and P2 peaks for the different bins
    N1 = [10 150];
    P2 = [90 250];
    
    n1_idx = dsearchn(trf_time',N1');
    p2_idx = dsearchn(trf_time',P2');
    
    plot_dat = plot_dat';
    [n1_peak,n1_lat] = min(plot_dat(:,n1_idx(1):n1_idx(2)),[],2);
    [p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);
    
    n1_time = trf_time(n1_idx(1):n1_idx(2));
    p2_time = trf_time(p2_idx(1):p2_idx(2));
    
    
    
    nexttile
    plot(n1_time(n1_lat),binEdges_dB(2:end), '-o', 'LineWidth', 1.5);
    n = length(binEdges_dB(2:end));
    % Labeling the axes
    xlabel('Time Lag (ms)');
    ylabel('Amplitude Bins (dB)');
    grid on;
    xlim([min(n1_time(n1_lat))-5 max(n1_time(n1_lat))+5])
    ylim([min(binEdges_dB(2:end))-5 max(binEdges_dB(2:end))+5] )
    % Adding a title or label similar to the "C"
    title('N1 Peak latency')
    set(gca,'FontSize',16,'YTick',binEdges_dB(2:end),'YtickLabel',binEdges_dB(2:end))
    box off
    
    nexttile
    plot((p2_peak-n1_peak), binEdges_dB(2:end), '-o', 'LineWidth', 1.5);
    
    % Labeling the axes
    xlabel('Magnitude a.u.');
    ylabel('Amplitude Bins (dB)');
    grid on;
    xlim([min(p2_peak-n1_peak)-5 max(p2_peak-n1_peak)+5])
    ylim([min(binEdges_dB(2:end))-5 max(binEdges_dB(2:end))+5] )
    % xlim([15 45]);
    % Adding a title or label similar to the "C"
    title('N1-P2 peak to peak')
    set(gca,'FontSize',16,'YTick',binEdges_dB(2:end),'YtickLabel',binEdges_dB(2:end))
    box off
    
    set(gca,'FontSize',16)
    title(t,sprintf('AB envelope %s',task{k}))
    
%     save_fig(gcf,figure_path,sprintf('DNS_RateofChange_Weights_%s',task{k}))
    
end
