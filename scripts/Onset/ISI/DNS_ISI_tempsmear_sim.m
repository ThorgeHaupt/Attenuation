OT_setup 
DNS_setup


%partition the data set
nfold = 6;
testfold = 1;

% % Bin the data
% binIndices = discretize(dns_dist, binEdges{i});
% [counts, ~, binIndices_env] = histcounts(dns_dist, binEdges{i});
% 
% figure 
% histogram(binIndices, numBins, 'Normalization', 'probability');
% xlabel('Bins');
% ylabel('Probability');
% title('Uniformly Distributed Bins');
% % 
% figure
% histogram('BinEdges',binEdges{i},'BinCounts',counts)
% set(gca,'view',[90 -90])
% ylabel('Count (Samples)')
% xlabel('Bin Widths (in s.)')
% set(gca,'FontSize',14,'XTickLabels',linspace(0,10,6))
% box off



for s=1:length(sbj)
    
    for k=1:2
        
        %% compute the envelopes
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
        
        cd(PATH)
        
        %% simulate the overlap here
        stim = extract_stimulus2(EEG, PATH,'onset', k, sbj{s});
        
        modelaep = mTRFtrain(stim',EEG.data',EEG.srate,1,tmin,tmax,0.05,'verbose',0);
        aep = squeeze(mean(modelaep.w,3));
      
        stim = extract_stimulus2(EEG, PATH,'onset', k, sbj{s});

        isi_idx = find(stim);
        
        onset=stim;
        y = zeros(size(onset))';
        
        %integrate the aeps
        for i = 1:length(isi_idx)
            % Randomly select a position to place the AEP
            
            position = isi_idx(i)-10;
            % Place the AEP at the selected position
            if position+length(aep) < EEG.pnts
                y(position:position+length(aep)-1) = y(position:position+length(aep)-1) + aep;
            end
            %     onset(position) = 1;
            
        end
        EEG.data = y;
        EEG.pnts = length(y);
        EEG.nbchan = 1;
        
        
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
            
            
            [counts, bins, binIndices] = histcounts(ons_dif,binEdges{ed});
            
            %                 figure
            %                 histogram('BinEdges',bins,'BinCounts',counts)
            %                 set(gca,'view',[90 -90])
            %                 ylabel('Count (Samples)')
            %                 xlabel('Onset Distance')
            %                 set(gca,'FontSize',14)
            %                 box off
            
            %                 sum(peak)-sum(counts)
            
            
            %save the counts
            sav_count{s,k,ed} = counts;
            
            ons_bin = zeros(length(counts),length(peak));
            for i = 1:length(binIndices)
                ons_bin(binIndices(i),ons_idx(i)) = 1;
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
            temp_w = [];
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
                temp_w(:,:,:,tr) = model_train.w;
                
                %predict the neural data
                [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
                
                %save the prediction values
                temp_r(:,tr) = STATS.r;
                
                fprintf('DIST GRID: sbj = %s, condition: %s; number bin: %d\r',sbj{s},task{k},length(counts))
            end
            mlpt_weight{s,k,ed} = squeeze(mean(temp_w,4));
            result_reg(s,k,ed,:) = squeeze(mean(temp_r,2));
        end
    end
end

%% plot the results
for nb = 1:length(numBins)
    
    
    
    edge = binEdges{nb};
    
    %% get the corresponding weights
    %get the y-values
    ylabels = round(edge(2:end)./100,2);
    figure
    %         set(gcf,'position',fig_pos)
    %plot the weights
    %         t = tiledlayout(2,2)
    weight_temp = [];
    for k = 1:length(task)
        weight_temp(:,:,:,k) = squeeze(cat(4,mlpt_weight{:,k,nb}));
    end
    
    plot_dat = squeeze(mean(weight_temp,[3 4]))';
    for i =1:size(plot_dat,2)
        plot(trf_time,plot_dat(:,i),'linew',2)
        hold on
    end
    grid off
    title('Model Weights')
    xlabel('Time Lag (ms)');
    %         set(gca,'FontSize',16,'YTick',round(y_val),'YtickLabel',ylabels)
    box off
end

figure
tiledlayout(1,2)
for nb = 1:length(numBins)
    
    edge = binEdges{nb};
    
    %% get the corresponding weights
    %get the y-values
    ylabels = round(edge(2:end)./100,2);
    weight_temp = [];
    for k = 1:2
        weight_temp(:,:,:,:,k) = squeeze(cat(4,mlpt_weight{:,k,nb}));
    end
    
    plot_dat = squeeze(mean(weight_temp,[3 4 5]))';
    %% plot the latency shift and peak to peak amplitude
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
    
    
    
    nexttile(1)
    plot(ylabels,n1_peak, '-o', 'LineWidth', 1.5,'Color',gradientColors_ISI(nb,:));
    hold on
    n = length(ylabels);
    % Labeling the axes
    xlabel('Time Lag (ms)');
    ylabel('a.u.')
    xlabel('Bins in seconds');
    xticks(ylabels)
    xticklabels(ylabels)
    ylabel('a.u.');
    %         grid on;
    xlim([0 ylabels(end)+1]);
    % Adding a title or label similar to the "C"
    title('N1 Peak amplitude')
    box off
    legend(num2str(numBins'),'Box','off')
    set(gca,'FontSize',14)
    
    %% P2
    [p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);
    
    
    nexttile(2)
    plot( ylabels,p2_peak, '-o', 'LineWidth', 1.5,'Color',gradientColors_ISI(nb,:));
    hold on
    
    xlabel('Time Lag (ms)');
    ylabel('a.u.')
    xlabel('Bins in seconds');
    xticks(ylabels)
    xticklabels(ylabels)
    ylabel('a.u.');
    %         grid on;
    xlim([0 ylabels(end)+1]);
    % Adding a title or label similar to the "C"
    title('P2 Peak amplitude')
    box off
    legend(num2str(numBins'),'Box','off','Location','northwest')
    set(gca,'FontSize',14)
    
    
end

rand = load('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_fulltr_rand.mat')
        
%extract the N1 and P2 values
N1 = [10 150];
P2 = [90 250];

n1_idx = dsearchn(trf_time',N1');
p2_idx = dsearchn(trf_time',P2');

for nb = 1:length(numBins)
    
    edge = binEdges{nb};
    weight_temp = [];
    weight_temp_rand = [];

    %Compute for the group average
    for k = 1:length(task)
        %average over conditions
        weight_temp(:,:,:,:,k) = squeeze(cat(5,mlpt_weight{:,k,nb}));
        %plot the random n1 peaks
        weight_temp_rand(:,:,:,:,:,k) = squeeze(cat(6,rand.mlpt_weight{:,k,nb,:}));
    end
    
    %average the results
    n1_dif(nb) = sum(diff(min(mean(weight_temp(:,n1_idx(1):n1_idx(2),:,:,:),[3 4 5]),[],2)));
    n1_rand_dif(nb,:) = squeeze(sum(diff(min(squeeze(mean(weight_temp_rand(:,n1_idx(1):n1_idx(2),:,:,:,:),[3 5 6])),[],2)),1));
    
    %compare the vlaues
    p_valn1(nb) = sum(n1_rand_dif(nb,:) < n1_dif(nb))/length(n1_rand_dif(nb,:));
    
    %do it for the P2
    p2_dif(nb) = sum(diff(max(mean(weight_temp(:,p2_idx(1):p2_idx(2),:,:,:),[3 4 5]),[],2)));
    p2_rand_dif(nb,:) = squeeze(sum(diff(max(squeeze(mean(weight_temp_rand(:,p2_idx(1):p2_idx(2),:,:,:,:),[3 5 6])),[],2)),1));
    p_valp2(nb) = sum(p2_rand_dif(nb,:) > p2_dif(nb))/length(p2_rand_dif(nb,:));

    
end

%apply FDR correction 
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh([p_valn1 p_valp2],0.05,'dep')

%plot the results
n1_p = adj_p(1:7)
figure
t=tiledlayout(1,2)
nexttile
for nb = 1:length(numBins)
    plot(numBins(nb),n1_dif(nb),'d','MarkerSize',20,'Color',gradientColors_ISI(nb,:),'MarkerFaceColor',gradientColors_ISI(nb,:))
    hold on 
    stande = std(n1_rand_dif(nb,:));
    mean_rand = mean(n1_rand_dif(nb,:));
    plot(numBins(nb),mean_rand,'ok','MarkerFaceColor','k')
    hold on
    errorbar(numBins(nb),mean_rand,stande,'k')
    box off
    if n1_p(nb) <0.05 && n1_p(nb) >0.01
        text(numBins(nb), mean_rand+stande+0.5, '*', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    elseif n1_p(nb) < 0.01 &&n1_p(nb) > 0.000
        text(numBins(nb), mean_rand+stande+0.5, '**', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    elseif n1_p(nb) == 0
        text(numBins(nb), mean_rand+stande+0.5, '***', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    else
        text(numBins(nb), mean_rand+stande+0.5, 'N.S.', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        
    end
    
end
xlim([numBins(1)-1 numBins(end)+1])
ylim([min(n1_dif)-1 4])
title('N1 Gradient Values')
set(gca,'FontSize',14)

p2_p = adj_p(length(numBins)+1:end);
nexttile
for nb = 1:length(numBins)
    p2(nb) = plot(numBins(nb),p2_dif(nb),'d','MarkerSize',20,'Color',gradientColors_ISI(nb,:),'MarkerFaceColor',gradientColors_ISI(nb,:))
    hold on 
    stande = std(p2_rand_dif(nb,:));
    mean_rand = mean(p2_rand_dif(nb,:));
    plot(numBins(nb),mean_rand,'ok','MarkerFaceColor','k')
    hold on
    errorbar(numBins(nb),mean_rand,stande,'k')
    box off
    
    if p2_p(nb) <0.05 && p2_p(nb) >0.01
        text(numBins(nb), p2_dif(nb)+0.5, '*', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    elseif p2_p(nb) < 0.01 &&p2_p(nb) > 0.000
        text(numBins(nb), p2_dif(nb)+0.5, '**', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    elseif p2_p(nb) == 0
        text(numBins(nb), p2_dif(nb)+0.5, '***', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    else
        text(numBins(nb), p2_dif(nb)+0.5, 'N.S.', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        
    end
    
    
end

xlim([numBins(1)-1 numBins(end)+1])
ylim([-2.5 max(p2_dif)+1])
title('P2 Gradient Values')
legend(p2,num2str(numBins'), 'Location', 'eastoutside','Box','off');
set(gca,'FontSize',14)


xlabel(t,'Number of Bins','FontSize',16)
ylabel(t,'a.u.','FontSize',16)

        
        