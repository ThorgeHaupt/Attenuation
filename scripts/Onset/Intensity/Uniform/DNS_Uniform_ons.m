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

%% get the bins for normal distribtuion 

% save('OnsEnv_vals.mat','part_sum')

load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\OnsEnv_vals.mat')

% uniform distribution
numBins = linspace(2,8,7); % Number of bins
for i = 1:length(numBins)
    binEdges{i} = quantile(part_sum, linspace(0, 1, numBins(i) + 1));
end

% Bin the data
binIndices = discretize(part_sum, binEdges{i});
[counts, ~, binIndices_env] = histcounts(part_sum, binEdges{i});

figure 
histogram(binIndices, numBins, 'Normalization', 'probability');
xlabel('Bins');
ylabel('Probability');
title('Uniformly Distributed Bins');
% % %
figure
histogram('BinEdges',binEdges{i},'BinCounts',counts)
set(gca,'view',[90 -90])
ylabel('Count (Samples)')
xlabel('Amplitude Bins (dB)')
set(gca,'FontSize',14)
box off


%start the loop
auditory = {'envelope'};
% auditory= {'melmenv','melonsmenv'}

for s=1:length(sbj)
    
    
    for k=1:2
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        cd(PATH)
        
        %get the neural data
        resp = double(EEG.data');
        
        novelty_ultm = load(sprintf('ons_ult_%s',task{k}));
        
        fs_new = EEG.srate;
        
        peak_s = smooth_peak(novelty_ultm.novelty_ultm,fs_new,'sigma',4);
        
        if size(resp,1)>length(peak_s)
            resp = resp(1:length(peak_s),:);
        elseif size(resp,1)<length(peak_s)
            peak_s = peak_s(:,1:size(resp,1));
        end
        
        %extract the stimulus
        menv = extract_stimulus2(EEG, PATH, auditory{:}, k,sbj{s},task);
        
        %normalize the envelope
        menv_norm = (menv - min(menv)) / ...
            (max(menv) - min(menv));
        
        %get the onset peaks
        ons_idx = find(peak_s);
        peak = peak_s;
        %get the tones 
        %%get the percentage of tones in each bin
        alarm = extract_stimulus2(EEG, PATH, 'alarm', k,sbj{s},task);
        odd = extract_stimulus2(EEG, PATH, 'odd', k,sbj{s},task);
        irr = extract_stimulus2(EEG, PATH, 'irregular', k,sbj{s},task);
        tone = [alarm odd irr];
        
        %find overlap and remove it
        for t = 1:size(tone,2)
            tone_idx = find(tone(:,t));
            overlappingOnsets = [];
            for it = 1:length(ons_idx)
                % Find if any onset in onsets2 is within the threshold of current onset in onsets1
                if any(abs(tone_idx-ons_idx(it)) <= 10)
                    overlappingOnsets = [overlappingOnsets; ons_idx(it)];
                end
                
            end
            %compute the percentage of overlap
            
            peak(overlappingOnsets) = 0;
            
        end
        
        %rather than binning the envelope -> get the values of
        %the envelope at the onsets indicies and bin those
        ons_idx = find(peak);
        
        
        %normalize those values -> will that lead to a
        %different distribution accorss participants?
        ons_env = zeros(size(ons_idx));
        ons_idx_new =[]
        for o = 1:length(ons_idx)
            %max value or mean value?
            if ons_idx(o)+5 < EEG.pnts
                [ons_env(1,o), di] = max(menv_norm(ons_idx(o)-1:ons_idx(o)+5));
                ons_idx_new(o) = ons_idx(o) + di-2;
            else
                ons_env(1,o) = menv_norm(ons_idx(o));
                ons_idx_new(o) = ons_idx(o);
            end
        end
        
        %get the distribution of values to determine the edges
        ons_env_sav{s,k} = ons_env;
        
        
        
        for i = 1:length(numBins)
            
            %compare to the env binning
            %Calculate the histogram counts and bin indices using histcounts
            [counts, ~, binIndices_ons] = histcounts(ons_env, binEdges{i});
            
            
            %binned onsets
            ons_bin = zeros(length(counts),length(norm_env));
            
            %!!! instead of binning the envelope we now take the
            %onsets into considerations as well
            for io = 1:length(binIndices_ons)
                ons_bin(binIndices_ons(io),ons_idx(io)) = 1;
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
            
            
            mlpt_weight{s,k,i} = model_train.w;
            
            %predict the neural data
            [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
            
            reg(s,k,i,:) = STATS.r;

            
            
            ovlp = [];
            for n = 1:size(stim,2)
                %check if the onsets are overlapping
                stim_idx = find(stim(:,n));
                
                for t = 1:size(tone,2)
                    tone_idx = find(tone(:,t));
                    overlappingOnsets = [];
                    for it = 1:length(tone_idx)
                        % Find if any onset in onsets2 is within the threshold of current onset in onsets1
                        if any(abs(stim_idx - tone_idx(it)) <= 10)
                            overlappingOnsets = [overlappingOnsets; tone_idx(it)];
                        end
                        
                    end
                    %compute the percentage of overlap
                    ovlp(n,t) = (length(overlappingOnsets)/length(tone_idx))*100;
                end
                
            end
            ovlp_sum{s,k,i} = ovlp;
            
        end
        
    end
    
    
    %         fprintf('Participant %s, condition %s low bound: %d; upper bound: %d; bin width: %d\r',sbj{s},task{k},lo_bound(lo),up_bound(up),bin_width(bw))
    
    
end
trf_time = model_train.t;
cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\Uniform\')

DNS_Intensity = struct();
DNS_Intensity.reg = reg;
DNS_Intensity.mlpt_weight = mlpt_weight;
% DNS_Intensity.sav_count = sav_count;
DNS_Intensity.auditory = auditory;
DNS_Intensity.trf_time= trf_time;
DNS_Intensity.ovlp_sum= ovlp_sum;
DNS_Intensity.t = 'Semi Grid Search with Uniformly distributed values, removed the tones from consideration';

save('DNS_Intensity_uniformbins_remove.mat','-struct','DNS_Intensity')

%% Plotting

fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Uniform\';

fig_pos = [1          41        1920         963];

trf_time = tmin:10:tmax;
temp_dat = squeeze(mean(reg,[ 2 4]));
temp_dat(temp_dat==0) = [];

binEdges = cellfun(@(x) round(x,2),binEdges,'UniformOutput',false)

audi = {'alarm','odd','irregular'}

font_s = 14;

for b = 1:length(numBins)
    
    %get the accuracies
    pred_acc = temp_dat(:,b);
    
    
    %% compare prediction accuracies
    
    cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\')
    ons_pred = load('DNS_SingleOnsets.mat','reg');
    ons_pred = squeeze(mean(ons_pred.reg(:,:,:),[2 3]));
    
    ABenv_pred = pred_acc;
    
    %     [AB_env_sort,s_idx] = sort(ABenv_pred,'ascend');
    %     env_pred_sort = ons_pred(s_idx);
    [ons_pred_sort,s_idx] = sort(ons_pred,'ascend');
    AB_env_sort = ABenv_pred(s_idx);

    
    figure;
    t = tiledlayout(4,2)
    nexttile([1 2])
    hold on
    set(gcf,'position',fig_pos)
    
    plot(env_pred_sort, '-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons');
    plot(AB_env_sort, '-o', 'Color', [0.49, 0.18, 0.56], 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'AB Ons');
    % Customize the plot
    xlabel('Subjects');
    ylabel('Pearson''s r');
    legend('Location', 'NorthWest');
    grid on;
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
    
    % save_fig(gcf,fig_path,'DNS_Intensity_Prediction_linear')
    
    %% get the corresponding weights
%     figure
%     set(gcf,'position',fig_pos)
%     plot the weights
%     t = tiledlayout(2,2)
    weight_temp=[];
    for k = 1:length(task)
        weight_temp(:,:,:,:,k) = squeeze(cat(4,mlpt_weight{:,k,b}));
    end
    plot_dat = squeeze(mean(weight_temp,[3 4 5]))';
    nexttile
    y_val = [];
    for i =1:size(plot_dat,2)
        plot(trf_time,plot_dat(:,i)+(i*14),'Color',[0 0.4470 0.7410],'linew',2)
        y_val(i) = mean(plot_dat(1:10,i)+(i*14));
        hold on
    end
    grid on
    title('Model Weights')
    set(gca,'FontSize',font_s,'YTick',round(y_val),'YtickLabel',binEdges{b}(2:end))
    box off
    
    
    %plot the imagesc of the weights
    nexttile
    imagesc('XData',trf_time,'YData',linspace(1,b+1,b+1),'CData',plot_dat')
    yticks(linspace(1,b+1,b+1))
    yticklabels(binEdges{b}(2:end))
    set(gca, 'YDir', 'normal','FontSize',font_s)
    axis tight
    title('Model Weights')
    
    
%     xlabel(t,'Time Lag (ms)');
%     ylabel(t,'Density Bins (dB)','FontSize',font_s);
    
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
    plot(n1_time(n1_lat),linspace(1,b+1,b+1), '-o', 'LineWidth', 1.5);
    n = length(binEdges{b}(2:end));
    % Labeling the axes
    xlabel('Time Lag (ms)');
    % ylabel('Amplitude Bins (dB)');
    grid on;
    xlim([min(n1_time(n1_lat))-5 max(n1_time(n1_lat))+5])
    ylim([min(linspace(1,b+1,b+1))-0.1 max(linspace(1,b+1,b+1))+0.1] )
    % Adding a title or label similar to the "C"
    title('N1 Peak latency')
    set(gca,'FontSize',font_s,'YTick',linspace(1,b+1,b+1),'YtickLabel',binEdges{b}(2:end))
    box off
    
    nexttile
    plot((p2_peak-n1_peak), binEdges{b}(2:end), '-o', 'LineWidth', 1.5);
    
    % Labeling the axes
    xlabel('Magnitude a.u.');
    % ylabel('Amplitude Bins (dB)');
    grid on;
    xlim([min(p2_peak-n1_peak)-5 max(p2_peak-n1_peak)+5])
    ylim([min(binEdges{b}(2:end))-0.1 max(binEdges{b}(2:end))+0.1] )
    % xlim([15 45]);
    % Adding a title or label similar to the "C"
    title('N1-P2 peak to peak')
    set(gca,'FontSize',font_s,'YTick',binEdges{b}(2:end),'YtickLabel',binEdges{b}(2:end))
    box off
    
%     title(t,'DNS based TRF','FontSize',28)
    set(gca,'FontSize',font_s)
    title(t,sprintf('Number of Bins %s',num2str(numBins(b))))
    
    
    nexttile([1 2])
    prct_dat = squeeze(mean(cat(3,ovlp_sum{:,:,b}),3));
    ba = bar(prct_dat,'stacked','FaceColor','flat')
    for k = 1:size(audi,2)
        ba(k).CData = audi_colorsrgb(audi{k})
    end
    view([90 -90])
    legend(audi,'Box','off')
    box off
    title('Precentage of total tones in Bins')
    xticklabels(binEdges{b}(2:end))
    set(gca,'FontSize',font_s)

%     save_fig(gcf,fig_path,sprintf('Uniform_remove_%sbins',num2str(numBins(b))))
end

%save the results

%% plot the alarm odd and irrelevant tone










