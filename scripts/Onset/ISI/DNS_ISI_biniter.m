%% DNS distance computation over pre-defined bin widths
%% Grid search for the optimal distance parameter setup 

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

%number of bins
min_value = 1;       % Minimum value of the range
max_value = 1000;    % Maximum value of the range

% uniform distribution
numBins = linspace(3,9,7); % Number of bins


typ_space = {'lin','log'};
%start the loop 

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
        
        for t = 1:length(typ_space)
            typ = typ_space{t};
            
            for ed = 1:length(numBins)
                
                n_bins = numBins(ed);
                
                switch typ
                    
                    case 'lin'
                        bin_edge = linspace(1,max_value,n_bins);
                    case 'log'
                        
                        % Generate a custom spaced vector using a power transformation
                        bin_edge = linspace(0, 1, n_bins) .^ 2;   % Squaring to concentrate bins at the higher end
                        bin_edge = round(bin_edge * (max_value - min_value) + min_value);
                        bin_edge = unique(bin_edge);

                end

                [counts, bins, binIndices] = histcounts(ons_dif,bin_edge);
                
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
                temp_w = [];
                temp_r = [];
                for tr= 1:nfold
                    
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
                end
                fprintf('DIST GRID: sbj = %s, condition: %s; number bin: %d\r',sbj{s},task{k},n_bins)
                mlpt_weight{s,k,t,ed} = squeeze(mean(temp_w,4));
                result_reg(s,k,t,ed,:) = squeeze(mean(temp_r,2));
            end
        end
    end
end
 

trf_time = model_train.t;
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\')

DNS_grid = struct();
DNS_grid.result_reg = result_reg;
DNS_grid.mlpt_weight = mlpt_weight;
DNS_grid.sav_count = sav_count;
DNS_grid.numBins= numBins;
DNS_grid.trf_time = trf_time;
% DNS_grid.auditory = auditory;

DNS_grid.t = 'Here we applied a grid search to the estimation optimal distance parameters between ';

save('DNS_dist_biniter.mat','-struct','DNS_grid')

fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Dist_grid\'

%% lets look into the data

%condition differences
figure
tiledlayout(1,2)
for k = 1:2
    
    temp_dat = squeeze(mean(result_reg(:,k,:,:,:),[1 5]));
    
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



fig_pos = [448   293   792   685];

trf_time = tmin:10:tmax;
temp_dat = squeeze(mean(result_reg(:,:,:,:,:),[1 2 5]))'
[~,m_idx] = max(temp_dat);

fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\'


for k = 1:length(task)
    for nb = 1:length(numBins)
        for ty = 1:length(typ_space)
            %get bin edges
            
            switch typ_space{ty}
                
                case 'lin'
                    bins = numBins(nb );
                    edge = linspace(1,1000,bins);
                    %get the counts
                    count = sum(squeeze(cat(3,sav_count{:,1,nb}))');
                    
                case 'log'
                    bin_edge = linspace(0, 1, numBins(nb)) .^ 2;   % Squaring to concentrate bins at the higher end
                    bin_edge = round(bin_edge * (max_value - min_value) + min_value);
                    edge = unique(bin_edge);
                    count = sum(squeeze(cat(3,sav_count{:,1,nb}))');
                    
            end
            

            
            %% get the corresponding weights
            %get the y-values
            ylabels = round(edge(2:end)./100,2);
            figure
            set(gcf,'position',fig_pos)
            %plot the weights
            t = tiledlayout(2,2)
            
            weight_temp = squeeze(cat(4,mlpt_weight{:,k,ty,nb}));
            
            plot_dat = squeeze(mean(weight_temp,[3 4]))';
            nexttile
            for i =1:size(plot_dat,2)
                plot(trf_time,plot_dat(:,i)+(i*14),'Color',[0 0.4470 0.7410],'linew',2)
                y_val(i) = mean(plot_dat(1:10,i)+(i*14));
                hold on
            end
            grid on
            title('Model Weights')
            xlabel('Time Lag (ms)');
            set(gca,'FontSize',16,'YTick',round(y_val),'YtickLabel',ylabels)
            box off
            
            % compare prediction accuracies
            
            cd('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\')
            ons_pred = load('DNS_SingleOnsets_trrepeat.mat','reg');
            ons_pred = squeeze(mean(ons_pred.reg(:,k,:),[2 3]));
            
            ABenv_pred = squeeze(mean(result_reg(:,k,ty,nb,:),5));
            
            [AB_env_sort,s_idx] = sort(ABenv_pred,'ascend');
            env_pred_sort = ons_pred(s_idx);
            
            
            nexttile
            hold on
            
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
            
            
            
            % ylabel(t,'Distance Bins (in s.)','FontSize',16);
            
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
            
            
            
            nexttile
            plot(ylabels,n1_peak, '-o', 'LineWidth', 1.5);
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
            
            nexttile
            plot( ylabels,p2_peak, '-o', 'LineWidth', 1.5);
            
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
            
            title(t,sprintf('DNS typ: %s, #bins:%d cond:%s',typ_space{ty},numBins(nb),task{k}),'FontSize',28)
            set(gca,'FontSize',16)
            
            
            save_fig(gcf,[fig_path typ_space{ty} filesep],sprintf('DNS_dist_%s_%d',task{k},numBins(nb)))
        end
    end
end


%% line fitting of the amplitude values
for k = 1:length(task)
    for nb = 1:length(numBins)
        for ty = 1:length(typ_space)
            %get bin edges
            
            switch typ_space{ty}
                
                case 'lin'
                    bins = numBins(nb );
                    edge = linspace(1,1000,bins);
                    %get the counts
                    count = sum(squeeze(cat(3,sav_count{:,1,nb}))');
                    
                case 'log'
                    bin_edge = linspace(0, 1, numBins(nb)) .^ 2;   % Squaring to concentrate bins at the higher end
                    bin_edge = round(bin_edge * (max_value - min_value) + min_value);
                    edge = unique(bin_edge);
                    count = sum(squeeze(cat(3,sav_count{:,1,nb}))');
                    
            end
            

            
            %% get the corresponding weights
            %get the y-values
            ylabels = round(edge(2:end)./100,2);
            
            weight_temp = squeeze(cat(4,mlpt_weight{:,k,ty,nb}));
            
            plot_dat = squeeze(mean(weight_temp,[3 4]))';
            
            % ylabel(t,'Distance Bins (in s.)','FontSize',16);
            
            %% plot the latency shift and peak to peak amplitude
            %find the N1 and P2 peaks for the different bins
            N1 = [10 150];
            P2 = [90 250];
            
            n1_idx = dsearchn(trf_time',N1');
            p2_idx = dsearchn(trf_time',P2');
            
            plot_dat = plot_dat';
            [n1_peak,n1_lat] = min(plot_dat(:,n1_idx(1):n1_idx(2)),[],2);
            [p2_peak,p2_lat] = max(plot_dat(:,p2_idx(1):p2_idx(2)),[],2);
            
            [p S] = polyfit(ylabels,n1_peak,1);
            n1_coef(k,nb,ty) = p(1);
            
            n1_time = trf_time(n1_idx(1):n1_idx(2));
            p2_time = trf_time(p2_idx(1):p2_idx(2));
        end
    end
end

%plot the results
for k = 1:2
    figure
    s1 = scatter(numBins,squeeze(n1_coef(k,:,1)),'filled');
    hold on 
    s2= scatter(numBins,squeeze(n1_coef(k,:,2)),'filled');
    title(task{k})
    legend([s1 s2],typ_space)
    lsline
end


%% what the is the best working bin size

[~,m_idx] = max(squeeze(mean(result_reg(:,:,:,:,:),[1 2 5]))');

%compare to the singluar model 

%get the weights
bins = numBins(m_idx(1,1));
edge = linspace(1,1000,bins);

%get the counts 
count = sum(squeeze(cat(3,sav_count{:,1,m_idx(1,1)}))');

% bin value
figure
tiledlayout(1,2)
nexttile
histogram('BinEdges',edge,'BinCounts',count)
ylabel('Count (Samples)')
xlabel('Onset Distance')
title('Linear Count')
set(gca,'FontSize',14,'view',[90 -90])
box off


%get the weights
temp_dat=[];
for k = 1:2
    
    temp_dat(:,:,:,:,k) = cat(4, mlpt_weight{:,k,1,m_idx(1,1)});
end
temp_dat = squeeze(mean(temp_dat,[3 5]));

weight_dat = squeeze(mean(temp_dat,3));

le = size(weight_dat,2);
nexttile
for i = 1:size(weight_dat,1)
    tem_dat = squeeze(temp_dat(i,:,:))';
    dnsMean = weight_dat(i,:);
    h(i) = plot(dnsMean,'linew',2);
    c(i,:) = get(h(i),'color');
    hold on
    N = size(tem_dat,1);
    ySEM = std(tem_dat,1)/sqrt(N);
    CI95 = tinv([0.025 0.975],N-1);
    yCI95 = bsxfun(@times,ySEM,CI95(:));
    conv = yCI95 + dnsMean ;
    x2 = [linspace(1,le,le) fliplr(linspace(1,le,le))];
    inbe = [conv(1,:) fliplr(conv(2,:))];
    f = fill(x2,inbe,c(i,:));
    f.FaceAlpha = 0.2;
    f.EdgeAlpha = 0.4;
    f.LineWidth = 0.5;
end
title('Onset Distance Model Weights')
box off
xlabel('Time Lags in ms.')
ylabel('a.u.')
legend(h,num2str(round(edge'/EEG.srate,2)),'Box','off')
set(gca,'FontSize',14)

% save_fig(gcf,fig_path,'LinCount_weights')






                    
%% log stuff

% Generate a custom spaced vector using a power transformation
log_bins = numBins(m_idx(1,2));
bin_edge = linspace(0, 1, log_bins) .^ 2;   % Squaring to concentrate bins at the higher end
bin_edge = round(bin_edge * (max_value - min_value) + min_value);

log_edge = unique(bin_edge);

bin_centers = (bin_edge(1:end-1) + bin_edge(2:end)) / 2;

%get the counts 
log_count = sum(squeeze(cat(3,sav_count{:,2,m_idx(1,1)}))');

% bin value
figure
tiledlayout(1,2)
nexttile
histogram('BinEdges',log_edge,'BinCounts',log_count)
% b_log = bar(bin_centers,log_count,'FaceColor','flat')
ylabel('Count (Samples)')
xlabel('Onset Distance')
title('Log Count')
set(gca,'FontSize',14,'view',[90 -90])
box off


%get the weights
temp_dat=[];
for k = 1:2
    
    temp_dat(:,:,:,:,k) = cat(4, mlpt_weight{:,k,2,m_idx(1,1)});
end
temp_dat = squeeze(mean(temp_dat,[3 5]));

weight_dat = squeeze(mean(temp_dat,3));

le = size(weight_dat,2);
nexttile
for i = 1:size(weight_dat,1)
    tem_dat = squeeze(temp_dat(i,:,:))';
    dnsMean = weight_dat(i,:);
    h(i) = plot(dnsMean,'linew',2);
    c(i,:) = get(h(i),'color');
    hold on
    N = size(tem_dat,1);
    ySEM = std(tem_dat,1)/sqrt(N);
    CI95 = tinv([0.025 0.975],N-1);
    yCI95 = bsxfun(@times,ySEM,CI95(:));
    conv = yCI95 + dnsMean ;
    x2 = [linspace(1,le,le) fliplr(linspace(1,le,le))];
    inbe = [conv(1,:) fliplr(conv(2,:))];
    f = fill(x2,inbe,c(i,:));
    f.FaceAlpha = 0.2;
    f.EdgeAlpha = 0.4;
    f.LineWidth = 0.5;
end
title('Onset Distance Model WEights')
box off
xlabel('Time Lags in ms.')
ylabel('a.u.')
legend(h,num2str(round(edge'/EEG.srate,2)),'Box','off')
set(gca,'FontSize',14)

save_fig(gcf,fig_path,'LogCount_weights')


for i = 1:size(c,1)
    b_log(i).CData  = c(i,:);
end
% 
% nexttile
% plot(trf_time,weight_dat','linew',2)
% legend(num2str(log_edge'))
% xlabel('Time Lags in ms.')
% ylabel('a.u.')
% title('Weights for linear sorted SOA')
% set(gca,'FontSize',14)
% 
% 
% for i = 1:4
%     figure
%     plot(squeeze(temp_dat(i,:,:)),'k')
% end







%iterate through participants
temp_dat = squeeze(mean(result_reg(:,k,:,:,:),[2 5]));

for s = 1:length(sbj)
    
    [s_val(s,:), s_idx(s,:)] = max(squeeze(temp_dat(s,:,:))')
end


figure
plot(s_val)
xlabel('Participants')
ylabel('Prediction Accuracy')
title('Participants')
legend({'linear','log'})

% bin value
figure
histogram('BinEdges',bins,'BinCounts',s_idx)
set(gca,'view',[90 -90])
ylabel('Count (Samples)')
xlabel('Onset Distance')
set(gca,'FontSize',14)
box off




 
 