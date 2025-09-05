%% DNS search over uniform distribution of distance metrics
%global paths
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

trf_time = model_train.t;
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\')

DNS_uni = struct();
DNS_uni.result_reg = result_reg;
DNS_uni.mlpt_weight = mlpt_weight;
DNS_uni.sav_count = sav_count;
DNS_uni.numBins= numBins;
DNS_uni.trf_time = trf_time;
DNS_uni.binEdges = binEdges;
% DNS_grid.auditory = auditory;

DNS_uni.t = 'Here we applied a uniform distribution contstraint to the estimation optimal distance parameters between, testing on all trials ';

save('DNS_dist_uniform_fulltr.mat','-struct','DNS_uni')


%% plot the results
%condition differences
figure
tiledlayout(1,2)
for k = 1:2
    
    temp_dat = squeeze(mean(result_reg(:,k,:,:),[1 4]));
    
    nexttile
    plot(numBins, temp_dat','linew',2)
    xlabel('Number of Bins')
    ylabel('Prediction Accuracy')
    title(sprintf('Task: %s',task{k}))
    legend({'linear','log'})
    box off
    set(gca,'FontSize',16,'Ylim',[0.03 0.07])
%     save_fig(gcf,fig_path,sprintf('perform_%s',task{k}))  
    
end


fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\Uniform\';

fig_pos = [1          41        1920         963];

trf_time = tmin:10:tmax;
%load all the needed variables
ons_pred = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_ISI_SingleOnsets_cleaned.mat');
adj_ons = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_adjtr.mat')
rand = load('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_fulltr_rand.mat')

%Figure for all the bins and conditions
for k = 1:length(task)
    for nb = 1:length(numBins)
        
        edge = binEdges{nb};

        %% get the corresponding weights
        %get the y-values
        ylabels = round(edge(2:end)./100,2);
%         figure
        set(gcf,'position',fig_pos)
        %plot the weights
        t = tiledlayout(2,2)
        
        weight_temp = squeeze(cat(4,mlpt_weight{:,k,nb}));
        
        plot_dat = squeeze(mean(weight_temp,[3 4]))';
        nexttile
        for i =1:size(plot_dat,2)
            plot(trf_time,plot_dat(:,i)+(i*14),'Color',[0.49, 0.18, 0.56],'linew',2)
            y_val(i) = mean(plot_dat(1:10,i)+(i*14));
            hold on
        end
        grid off
        title('Model Weights')
        xlabel('Time Lag (ms)');
        set(gca,'FontSize',16,'YTick',round(y_val),'YtickLabel',ylabels)
        box off
        
        % compare prediction accuracies
        ons_pred_res = squeeze(mean(ons_pred.reg(:,k,:),[2 3]));
        
        %load the adjusted prediction accuracies
        adj_ons_res = squeeze(mean(adj_ons.result_reg(:,k,nb,:,:),[2 4 5]));
        
        ABenv_pred = squeeze(mean(result_reg(:,k,nb,:),4));
        
        [AB_env_sort,s_idx] = sort(ABenv_pred,'ascend');
        ons_pred_sort = ons_pred_res(s_idx);
        adj_ons_res = adj_ons_res(s_idx);
        
        p_sav(k,nb) = signrank(ons_pred_res,ABenv_pred);
        
        nexttile
        hold on
        
        plot(ons_pred_sort, '-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons');
        plot(AB_env_sort, '-o', 'Color', [0.49, 0.18, 0.56], 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'ISI Ons');
        plot(adj_ons_res ,'-o', 'Color', [0.3,0.615,0.818], 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'Ons adj');
        % Customize the plot
        xlabel('Subjects');
        ylabel('Pearson''s r');
        legend('Location', 'NorthWest','Box','off');
        grid off;
        set(gca,'FontSize',14)
        
        

        
        %plot the chance level for each participant
       
        rand_reg = squeeze(mean(rand.result_reg(:,k,nb,:,:),5));
        rand_reg = rand_reg(s_idx,:);
        mean_rand = mean(rand_reg,2);
        sem_rand = std(rand_reg,[],2)/sqrt(size(rand_reg,2));
        errorbar(1:length(sbj),mean_rand,sem_rand,'k','LineStyle','none','LineWidth',3, 'DisplayName', 'Random');
        
        dif_pred(k,nb,:) = AB_env_sort -  mean_rand;
%         
        cmp_ons = [adj_ons_res mean_rand];
        for tt = 1:2
            [p h] = signrank(cmp_ons(:,tt),AB_env_sort);
            if tt == 1;ad =0; else ad = 0.01; end
            if p <0.05 && p > 0.01
                text(length(sbj), max(cmp_ons(:,tt))-ad, '*', 'FontSize', 18, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            elseif p < 0.01 && p > 0.000
                text(length(sbj), max(cmp_ons(:,tt))-ad, '**', 'FontSize', 18, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            elseif p< 0.000
                ext(length(sbj), max(cmp_ons(:,tt))-ad, '***', 'FontSize', 18, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            end
        end
        %ylabel(t,'Distance Bins (in s.)','FontSize',16);
        
        %% plot the latency shift and peak to peak amplitude
        %find the N1 and P2 peaks for the different bins
        N1 = [10 150];
        P2 = [90 250];
        
        n1_idx = dsearchn(trf_time',N1');
        p2_idx = dsearchn(trf_time',P2');
        
        plot_dat = plot_dat';
        [n1_peak,n1_lat] = min(plot_dat(:,n1_idx(1):n1_idx(2)),[],2);
        [~,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);
        
        n1_time = trf_time(n1_idx(1):n1_idx(2));
        p2_time = trf_time(p2_idx(1):p2_idx(2));
        
        
        
        nexttile
        plot(ylabels,n1_peak, '-o', 'LineWidth', 1.5);
        hold on
        n = length(ylabels);
        % Labeling the axes
        xlabel('Time Lag (ms)');
        ylabel('a.u.')
        xlabel('Bins in seconds');
        xticks(ylabels)
        xticklabels(ylabels)
        ylabel('a.u.');
        grid on;
        xlim([0 ylabels(end)+1]);
        if i >1
            
            ylim([min(n1_peak)-1.5 max(n1_peak)+1.5]);
        end
        % Adding a title or label similar to the "C"
        title('N1 Peak amplitude')
        box off
        set(gca,'FontSize',14)
        
        %plot the random n1 peaks
        weight_temp = squeeze(cat(5,rand.mlpt_weight{:,k,nb,:}));
        weight_temp = reshape(weight_temp,nb+1,61,22,length(sbj),[]);
        plot_dat_rand = squeeze(mean(weight_temp,[3 4]));
        
        [n1_peak_rand,n1_lat_rand] = min(plot_dat_rand(:,n1_idx(1):n1_idx(2),:),[],2);
        mean_val = squeeze(mean(n1_peak_rand,2));
        ci_val = 1.96*squeeze(std(n1_peak_rand,[],3) / sqrt(size(n1_peak_rand,3)));
%         ci_int_1 = [mean(mean_val(1,:)-ci_val(1,:),2) mean(mean_val(1,:)+ci_val(1,:),2)];
%         ci_int_2 = [mean(mean_val(2,:)-ci_val(2,:),2) mean(mean_val(2,:)+ci_val(2,:),2)];
        mean_group = mean(mean_val,2);
        
        errorbar(ylabels, mean_group,ci_val,'o-', 'Color','k','LineWidth', 1.5, 'MarkerSize', 6, 'CapSize', 8)
        
        
        
        [p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);

        
        nexttile
        plot( ylabels,p2_peak, '-o', 'LineWidth', 1.5);
        hold on
        
        xlabel('Time Lag (ms)');
        ylabel('a.u.')
        xlabel('Bins in seconds');
        xticks(ylabels)
        xticklabels(ylabels)
        ylabel('a.u.');
        grid on;
        xlim([0 ylabels(end)+1]);
        if i >1
            
            ylim([min(p2_peak)-1.5 max(p2_peak)+1.5]);
        end
        % Adding a title or label similar to the "C"
        title('P2 Peak amplitude')
        box off
        set(gca,'FontSize',14)
        
        title(t,sprintf('DNS #bins:%d cond:%s',numBins(nb),task{k}),'FontSize',28)
        set(gca,'FontSize',16)
        [p2_peak_rand,p2_lat_rand] = max(plot_dat_rand(:,p2_idx(1):p2_idx(2),:),[],2);
        mean_val = squeeze(mean(p2_peak_rand,2));
        ci_val = 1.96*squeeze(std(p2_peak_rand,[],3) / sqrt(size(p2_peak_rand,3)));
%         ci_int_1 = [mean(mean_val(1,:)-ci_val(1,:),2) mean(mean_val(1,:)+ci_val(1,:),2)];
%         ci_int_2 = [mean(mean_val(2,:)-ci_val(2,:),2) mean(mean_val(2,:)+ci_val(2,:),2)];
        mean_group = mean(mean_val,2);
        
        errorbar(ylabels, mean_group,ci_val,'o-', 'Color','k','LineWidth', 1.5, 'MarkerSize', 6, 'CapSize', 8);
        
        
        
        
%         save_fig(gcf,fig_path,sprintf('DNS_dist_%s_%d',task{k},numBins(nb)))
    end
end

%% plot the difference between conditions
dif_preds = squeeze(mean(dif_pred,1));
figure
for i = 1:length(numBins)
    plot(dif_preds(i,:),'o-','Color',gradientColors_ISI(i,:),'linew',2)
    hold on
    p_sign(i) =signrank(dif_preds(i,:));
end
legend(num2str(numBins'),'Box','off','Location','northwest')
set(gca,'FontSize',16)

[h,p,ci,stats] = ttest(dif_preds')
[h, crit_p, adj_ci_cvrg, adj_p]= fdr_bh(p,0.05,'pdep')

%try it with a violinplot
figure
violinplot(dif_preds',num2str(numBins'),...
    'ViolinColor',gradientColors_ISI)
xlabel('Number of Bins')
ylabel('\Delta bin ISI - adj Ons.')
set(gca,'FontSize',16)
box off

sigstar({[1],[2],[3],[4],[5],[6],[7]},adj_p)

save_fig(gcf,fig_path,'DNS_pred_rand_cmp')

%% test whether narrow and wide condition amplitude values are different?

for nb = 1:length(numBins)
    peak_sumn1 = [];
    peak_sump2= [];
    for k = 1:length(task)
        
        edge = binEdges{nb};
        
        %% get the corresponding weights
        %get the y-values
        ylabels = round(edge(2:end)./100,2);
        
        weight_temp = squeeze(cat(4,mlpt_weight{:,k,nb}));
        
        plot_dat = squeeze(mean(weight_temp,3));
        %% plot the latency shift and peak to peak amplitude
        %find the N1 and P2 peaks for the different bins
        N1 = [10 150];
        P2 = [90 250];
        
        n1_idx = dsearchn(trf_time',N1');
        p2_idx = dsearchn(trf_time',P2');
        
        [n1_peak,n1_lat] = min(plot_dat(:,n1_idx(1):n1_idx(2),:),[],2);
        [p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2),:),[],2);
        
        peak_sumn1(:,:,k) = squeeze(n1_peak);
        peak_sump2(:,:,k) = squeeze(p2_peak);
        
        
        

    end
    
    %compare the two conditions
    pn1(nb) = signrank(reshape(peak_sumn1(:,:,1),[],1),reshape(peak_sumn1(:,:,2),[],1));
    pp2(nb) = signrank(reshape(peak_sump2(:,:,1),[],1),reshape(peak_sump2(:,:,2),[],1));

    
end
% fdr correction 
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh([pn1 pp2],0.05,'dep');


%% compare the different bin widths with respect to prediction accuracY
ons_pred = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\DNS_SingleOnsets_trrepeat.mat');
ons_pred_res = squeeze(mean(ons_pred.reg,[2 3]));
plot_dat = flip(cat(2,ons_pred_res,squeeze(mean(result_reg,[2 4]))),2);
figure
violinplot(plot_dat,num2str(flip([0; numBins'])))
xlabel('Number of Bins')
ylabel('Prediction Accuracy')
xticklabels(num2str(flip([0; numBins'])))
box off
set(gca,'FontSize',16)

data = plot_dat;
% Assuming your matrix is named `data` (size 20x7)
[numRows, numCols] = size(data);

% Initialize a matrix to store p-values
p_values = nan(numCols, numCols);

% Perform paired t-tests for each pair of columns
for i = 1:numCols
    for j = i+1:numCols
        [~, p_values(i, j)] = ttest(data(:, i), data(:, j)); % Perform paired t-test
    end
end
p_values(isnan(p_values)) = [];

% Display p-values
disp('P-values for paired t-tests between columns:');
disp(p_values);

% apply FDR correction 
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_values(:),0.05,'dep');
mat = zeros(size(plot_dat,2));
% Get the linear indices of the upper triangular part of the matrix
upper_tri_indices = find(triu(ones(size(plot_dat,2)), 1));

% Assign the vector values to the upper triangular part
mat(upper_tri_indices) = adj_p;

pv = diag(mat,1);

sigstar({[1 2],[2 3],[3 4],[4 5],[5 6], [6 7],[7 8]},pv')

% save_fig(gcf,fig_path,'DNS_ISI_uniformContrast')




%% computation of the peak amplitude values for N1 and P2
%save the values 
sav_ampn1 = [];
sav_ampp2 = [];
figure
tiledlayout(1,2)

for nb = 1:length(numBins)
    
    edge = binEdges{nb};
    
    %% get the corresponding weights
    %get the y-values
    ylabels = round(edge./100,2);
    yvals = (ylabels(1:end-1)+ylabels(2:end))/2;
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
    plot(yvals,n1_peak, '-o', 'LineWidth', 0.5,'Color',gradientColors_ISI(nb,:));
    hold on
%     n = length(ylabels);
%     % Labeling the axes
%     xlabel('Time Lag (ms)');
%     ylabel('a.u.')
%     xlabel('Bins in seconds');
%     xticks(ylabels)
%     xticklabels(ylabels)
%     ylabel('a.u.');
%     %         grid on;
%     xlim([0 ylabels(end)+1]);
%     % Adding a title or label similar to the "C"
%     title('N1 Peak amplitude')
%     box off
%     legend(num2str(numBins'),'Box','off')    
    sav_ampn1 = [sav_ampn1; yvals',n1_peak];

    
    %% P2
    [p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);
    
    
    nexttile(2)
    plot( yvals,p2_peak, '-o', 'LineWidth', 0.5,'Color',gradientColors_ISI(nb,:));
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
    
    
    sav_ampp2 = [sav_ampp2; yvals',p2_peak];
    
    
end

%fitfunction 
[best_model, params, rsq,relts] = fitfun(sav_ampn1(:,1),sav_ampn1(:,2))

%polyfit
n1_fit = polyval(params,yvals);

nexttile(1)
plot(yvals,n1_fit,'k','linew',2)
n = length(ylabels);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('a.u.')
xlabel('IOI');
xticks([ylabels(1:3:end) ylabels(end)])
xticklabels([ylabels(1:3:end) ylabels(end)])
ylabel('a.u.');
%         grid on;
xlim([0 ylabels(end)+1]);
% Adding a title or label similar to the "C"
title('N1 Peak amplitude')
box off
legend(num2str(numBins'),'Box','off')
set(gca,'FontSize',12)

%P2 
%fitfunction 
[best_model, params, rsq,relts] = fitfun(sav_ampp2(:,1),sav_ampp2(:,2));
L = params(1);
k = params(2);
x0 = params(3);

% Compute predicted y using the logistic function
p2_fit = L ./ (1 + exp(-k * (yvals - x0)));

nexttile(2)
plot(yvals,p2_fit,'k','linew',2)
n = length(ylabels);
% Labeling the axes
xlabel('Time Lag (ms)');
ylabel('a.u.')
xlabel('IOI');
xticks([ylabels(1:3:end) ylabels(end)])
xticklabels([ylabels(1:3:end) ylabels(end)])
ylabel('a.u.');
%         grid on;
xlim([0 ylabels(end)+1]);
% Adding a title or label similar to the "C"
title('P2 Peak amplitude')
box off
legend(num2str(numBins'),'Box','off','location','northeast')
set(gca,'FontSize',12)


% save_fig(gcf,fig_path,'DNS_ISI_ampvals_log_bettermod')

%% Compare the values to chance
%compare to the null distribution
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
set(gca,'FontSize',14,'Ylim',[-2.3 11])


xlabel(t,'Number of Bins','FontSize',16)
ylabel(t,'\nabla values','FontSize',16)

save_fig(gcf,fig_path,'DNS_ISI_Gradients')

%% The same as above, but the P2-N1 distance
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
    
    %peak to peak ratio
    n1p2_dif(nb) = sum(diff( max(mean(weight_temp(:,p2_idx(1):p2_idx(2),:,:,:),[3 4 5]),[],2)- min(mean(weight_temp(:,n1_idx(1):n1_idx(2),:,:,:),[3 4 5]),[],2)    ));
    n1p2_rand_dif(nb,:) = sum(diff( squeeze(max(squeeze(mean(weight_temp_rand(:,p2_idx(1):p2_idx(2),:,:,:,:),[3 5 6])),[],2)) - squeeze(min(squeeze(mean(weight_temp_rand(:,n1_idx(1):n1_idx(2),:,:,:,:),[3 5 6])),[],2))   ,1),1);
    
    %compare the vlaues
    p_valn1p2(nb) = sum(n1p2_rand_dif(nb,:) > n1p2_dif(nb))/length(n1p2_rand_dif(nb,:));
    

    
end

%apply FDR correction 
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_valn1p2,0.05,'dep')



%plot the results
n1_p = adj_p(1:7)
figure
t=tiledlayout(1,1)
nexttile
for nb = 1:length(numBins)
    p(nb) = plot(numBins(nb),n1p2_dif(nb),'d','MarkerSize',20,'Color',gradientColors_ISI(nb,:),'MarkerFaceColor',gradientColors_ISI(nb,:))
    hold on 
    stande = std(n1p2_rand_dif(nb,:));
    mean_rand = mean(n1p2_rand_dif(nb,:));
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
% ylim([min(n1_dif)-1 3])
title('P2-N1 Gradient Values')
set(gca,'FontSize',14)
legend(p,num2str(numBins'), 'Location', 'eastoutside','Box','off');


xlabel(t,'Number of Bins','FontSize',16)
ylabel(t,'a.u.','FontSize',16)



save_fig(gcf,fig_path,'DNS_ISI_P2N1Gradients')

%% Permutation testing for single participants
for s = 1:length(sbj)
    for nb = 1:length(numBins)
        
        edge = binEdges{nb};
        weight_temp = [];
        weight_temp_rand = [];

        for k = 1:length(task)
            %average over conditions
            weight_temp(:,:,:,k) = squeeze(cat(4,mlpt_weight{s,k,nb}));
            %plot the random n1 peaks
            weight_temp_rand(:,:,:,:,k) = squeeze(cat(4,rand.mlpt_weight{s,k,nb,:}));
        end
        weight_temp = squeeze(mean(weight_temp,[3 4]));
        weight_temp_rand = squeeze(mean(weight_temp_rand,[3 5]));
        
        %% get the corresponding weights
        %get the y-values
        ylabels = round(edge(2:end)./100,2);
        
        
        %get the difference score
        
        difn1 = sum(diff(min(weight_temp(:,n1_idx(1):n1_idx(2)),[],2)));
        difp2 = sum(diff(max(weight_temp(:,p2_idx(1):p2_idx(2)),[],2)));
        
        
        %get the random difference score
        difn1_rand = sum(diff(squeeze(min(weight_temp_rand(:,n1_idx(1):n1_idx(2),:),[],2)),1,1),1);
        difp2_rand = sum(diff(squeeze(min(weight_temp_rand(:,p2_idx(1):p2_idx(2),:),[],2)),1,1),1);
        
        %compare the total number 
        pn1(s,nb) = sum(difn1_rand <= difn1)/size(weight_temp_rand,3);
        pp2(s,nb) = sum(difp2_rand >= difp2)/size(weight_temp_rand,3);
        
    end
end

%apply binominal testing

%% Lets do some accuracy comparison 
fig_pos = [121,222,504,437];
[ons_pred_sort, sidx] = sort(ons_pred_res,'ascend')
bin_isi = squeeze(mean(result_reg,[2 4]));
figure, hold on
set(gcf,'Position',fig_pos)
for nb = 1:length(numBins)
    plot(bin_isi(sidx,nb),'-o', 'Color', gradientColors_ISI(nb,:),'linew',2, 'MarkerFaceColor', 'auto'); 
    p(nb) = signrank(ons_pred_sort,bin_isi(sidx,nb));
end
labels = num2cell(numBins');
%contrast the prediction accuracy
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p,0.05,'dep');
for i = 1:length(adj_p)
    if adj_p(i) < 0.05 && adj_p(i) > 0.01
        labels{i} = strcat(num2str(labels{i}), ' *');
    elseif adj_p(i) < 0.01 && adj_p(i) > 0.0001
        labels{i} = strcat(num2str(labels{i}), ' **');
    elseif  adj_p(i) < 0.0001
        labels{i} = strcat(num2str(labels{i}), ' ***');
    else 
        labels{i} = strcat(num2str(labels{i}), ' N.S.');
    end
end
legend(labels,'Box','off','Location','northwest')

plot(ons_pred_sort,'-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons Orig.');

box off
xlabel('Participants')
ylabel('Prediction Accuracy')
set(gca,'FontSize',16,'Ylim',[0.01 0.08])

save_fig(gcf,fig_path,'DNS_ISI_condition_pred')

%% against the chance level
[ons_pred_sort, sidx] = sort(ons_pred_res,'ascend')
bin_isi = squeeze(mean(result_reg,[2 4]));
figure, hold on
for nb = 1:length(numBins)
    plot(bin_isi(sidx,nb),'-o', 'Color', gradientColors_ISI(nb,:),'linew',2, 'MarkerFaceColor', 'auto'); 
    p(nb) = signrank(ons_pred_sort,bin_isi(sidx,nb));
end
labels = num2cell(numBins');
%contrast the prediction accuracy
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p,0.05,'dep');
for i = 1:length(adj_p)
    if adj_p(i) < 0.05 && adj_p(i) > 0.01
        labels{i} = strcat(num2str(labels{i}), ' *');
    elseif adj_p(i) < 0.01 && adj_p(i) > 0.0001
        labels{i} = strcat(num2str(labels{i}), ' **');
    elseif  adj_p(i) < 0.0001
        labels{i} = strcat(num2str(labels{i}), ' ***');
    else 
        labels{i} = strcat(num2str(labels{i}), ' N.S.');
    end
end
legend(labels,'Box','off','Location','northwest')

plot(ons_pred_sort,'-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons Orig.');

box off
xlabel('Participants')
ylabel('Prediction Accuracy')
set(gca,'FontSize',16)

%% significance testing against onset vector and random vector

for bin = 1:length(numBins)
    for r = 1:2
        if r == 1
            %test isi model against onsets
            p_val(r,bin) = signrank(squeeze(mean(result_reg(:,:,bin,:),[2 4])),squeeze(mean(ons_pred.reg,[2 3])));
        else
           p_val(r,bin) =  signrank(squeeze(mean(result_reg(:,:,bin,:),[2 4])),squeeze(mean(rand.result_reg(:,:,bin,:,:),[2 4 5])));
        end
    end
end

[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_val(:),0.05,'dep');

p_int = reshape(adj_p,size(p_val));

%% paper plots
bin_int = [2 4 6];

fig_path2 = [fig_path 'comb_fig\']

fig_pos = [268   270   440   650]
figure
set(gcf,'position',fig_pos)
%plot the weights
t = tiledlayout(2,3);
y_val = [];
%do sig testing before doing anyhting else
for bi=1:length(bin_int)
    
    bin = bin_int(bi);
    %get the model weights
    edge = binEdges{bin};
    
    %% get the corresponding weights
    %get the y-values
    ylab = round(edge./100,2);
    ylabels= ylab(1:end-1)+(diff(ylab)/2);
    weight_temp = [];
    for k = 1:2
        weight_temp(:,:,:,:,k) = squeeze(cat(4,mlpt_weight{:,k,bin}));
    end
    
    plot_dat = squeeze(mean(weight_temp,[3 4 5]))';
    topo_dat = squeeze(mean(weight_temp,[4 5]));
    
    if bi == 1; add = 30; else add = 14; end
    
    nexttile(bi)
    for i =1:size(plot_dat,2)
        plot(trf_time,plot_dat(:,i)+(i*add),'Color',gradientColors_ISI(bin,:),'linew',2.5)
        y_val(i) = mean(plot_dat(1:10,i))+(i*add);
        hold on     
    end
    y_vals = [0 y_val];
    y_mrk = y_vals(2:end)+(diff(y_vals)/2);
    grid off
    title('Model Weights')
    xlabel('Time Lag (ms)');
    set(gca,'FontSize',16,'YTick',[0 y_mrk],'YtickLabel',ylab)
    box off
    
    %% plot the prediction accuracies
    %load the adjusted prediction accuracies
    ons_pred_res = squeeze(mean(ons_pred.reg,[2 3]));
    
    bin_isi_pred = squeeze(mean(result_reg(:,:,bin,:),[2 4]));
    
    [ons_pred_sort,s_idx] = sort(ons_pred_res,'ascend');
    bin_isi_sort = bin_isi_pred(s_idx);

    nexttile(bi+3)
    hold on
       
    %plot(ons_pred_sort, '-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons');
    plot(bin_isi_sort, '-o', 'Color', gradientColors_ISI(bin,:), 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'ISI Ons');
    box off
    
    xlabel('Subjects');
    ylabel('Pearson''s r');
    legend('Location', 'NorthWest','Box','off');
    grid off;
    set(gca,'FontSize',14)
    
    %add the random data
    rand_reg = squeeze(mean(rand.result_reg(:,:,bin,:,:),[2 5]));
    rand_reg = rand_reg(s_idx,:);
    mean_rand = mean(rand_reg,2);
    sem_rand = std(rand_reg,[],2)/sqrt(size(rand_reg,2));
    errorbar(1:length(sbj),mean_rand,sem_rand,'k','LineStyle','none','LineWidth',3, 'DisplayName', 'Random');
    
    cmp_ons = [ons_pred_sort mean_rand];
    mrk = {'^','*'};
    %add the significance 
    for tt = 2
       
        p = p_int(tt,bin);
        if tt == 1;ad =-0.005; else ad = 0.005; end
        if p <0.05 && p > 0.01
            text(length(sbj), max(cmp_ons(:,tt))-ad,  mrk{tt}, 'FontSize', 18, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
        elseif p < 0.01 && p > 0.001
            text(length(sbj), max(cmp_ons(:,tt))-ad, [mrk{tt} mrk{tt}], 'FontSize', 18, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
        elseif p< 0.001
            text(length(sbj), max(cmp_ons(:,tt))-ad, [mrk{tt} mrk{tt} mrk{tt}], 'FontSize', 18, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
        end
    end
end


% save_fig(gcf,fig_path2,'DNS_ISI_comb_fig_all2')





    

    




