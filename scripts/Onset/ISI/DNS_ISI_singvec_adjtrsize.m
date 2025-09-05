%% equalize the amount of trainingsdata
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
                temp_w = [];
                temp_r = [];
                for tr = 1:nfold
                    
                    
                    %split the data
                    [strain,rtrain,stest,rtest] = mTRFpartition(stim,resp,nfold,tr);
                    
                    %count the total onsets in the trainings data
                    count_tr = cell2mat(cellfun(@sum,strain,'UniformOutput',false));
                    
                    %delte the onsets 
                    for on = 1:size(strain,1)
                        
                        %get average training per segment over bins
                        sum_seg = sum(count_tr(on,:));
                        
                        %get average number of single bin 
                        avg_bin = round(sum_seg/size(count_tr,2));
                        
                        %ons to delete to have avg num of ons per bin 
                        nr_ons_del = sum_seg-avg_bin;
                       
                        %get the singular onset vector
                        temp_dat = sum(strain{on,1},2);
                        
                        %search for the onsets
                        temp_idx = find(temp_dat);
                        
                        %delete half of those
                        temp_del_idx = randperm(length(temp_idx),nr_ons_del);
                        
                        temp_dat(temp_idx(temp_del_idx),1) = 0;
                        
                        strain{on,1} = temp_dat;
                    end

                    
                    strainz = strain; %normalization occurs at the stim extraction fun
                    stestz = sum(stest,2);   %normalization occurs at the stim extraction fun
                    
                    
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

                    %predict the neural data
                    [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
                    
                    %save the prediction values
                    temp_r(:,tr) = STATS.r;
                    
                    fprintf('DIST GRID: sbj = %s, condition: %s; number bin: %d rand_iter: %d\r',sbj{s},task{k},length(counts),rn)
                end
                
                result_reg(s,k,ed,rn,:) = squeeze(mean(temp_r,2));
            end
        end
    end
end




trf_time = model_train.t;
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\')

DNS_uni = struct();
DNS_uni.result_reg = result_reg;
% DNS_uni.sav_count = sav_count;
DNS_uni.numBins= numBins;
DNS_uni.trf_time = trf_time;
DNS_uni.binEdges = binEdges;
% DNS_grid.auditory = auditory;

DNS_uni.t = 'Here we applied a uniform distribution contstraint to the estimation optimal distance parameters reducing the training data to that of the binned model for the singular onset vector ';

save('DNS_dist_uniform_adjtr.mat','-struct','DNS_uni')


%% plot the results

fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\Uniform\adjTraining\'

%plot the descriptives
temp_dat = squeeze(mean(result_reg,[2 4 5]));
figure
violinplot(temp_dat,numBins)
xlabel('Number of bins in the reference model')
ylabel('Prediction Accuracy')
box off
title('Trainingsdata adjusted Single Onset Vector')
set(gca,'FontSize',16)

% save_fig(gcf,fig_path,'DNS_ISI_singlevec_adjustedtraining')

%% plot over participants
% compare prediction accuracies
ons_pred = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_ISI_SingleOnsets_cleaned.mat');
ons_pred = squeeze(mean(ons_pred.reg(:,:,:),[2 3]));

endColor = [0, 0.8, 0.9]; % Convert 8-bit RGB to MATLAB RGB
baseColor = [0, 0.5, 0.7410]; % End color for gradient

% Number of desired colors
numColors = 7;

% Generate linearly spaced colors along the gradient
gradientColors = zeros(numColors, 3);
for i = 1:numColors
    alpha = (i-1) / (numColors-1); % Blending factor (0 to 1)
    gradientColors(i, :) = (1 - alpha) * baseColor + alpha * endColor;
end

[ons_pred_sort s_idx] = sort(ons_pred,'ascend');
temp_dat_sort = temp_dat(s_idx,:);
figure
for i = 1:length(numBins)
    plot(1:length(sbj),temp_dat_sort(:,i),'o-','Color',gradientColors(i,:),'linew',2,'MarkerFaceColor', 'auto')
    hold on
end
legend(num2str(numBins'),'Location','northwest','box','off')
hold on
plot(ons_pred_sort,'-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons Orig.');
xlabel('Participants')
ylabel('Prediction Accuracy')
box off
title('Trainingsdata adjusted Single Onset Vector')
set(gca,'FontSize',16)

save_fig(gcf,fig_path,'DNS_ISI_singlevec_overparticipants')


%plot the gradient over participants
[~,s_idx] = sort(mean(temp_dat,2),'descend');
temp_dat = temp_dat(s_idx,:);
x = 1:7;
y = temp_dat; % Random data for 20 lines (Replace with your actual data)


% Define the new, higher-resolution X-axis
x_highres = linspace(min(x), max(x), 100); % Interpolated X values (100 points)

% Interpolate Y values for higher resolution
y_highres = zeros(size(y, 1), length(x_highres));
for i = 1:size(y, 1)
    y_highres(i, :) = interp1(x, y(i, :), x_highres, 'linear'); % Linear interpolation
end

figure
imagesc(y_highres)

% Define the new, higher-resolution X-axis
x_highres = linspace(min(x), max(x), 100); % Interpolated X values (100 points)

% Interpolate Y values for higher resolution
y_highres = zeros(size(y, 1), length(x_highres));
for i = 1:size(y, 1)
    y_highres(i, :) = interp1(x, y(i, :), x_highres, 'linear'); % Linear interpolation
end

% Flatten all Y values to compute global min and max
y_global = y_highres(:); % Flatten to a single array

% Define global min and max for normalization
global_min = min(y_global);
global_max = max(y_global);

% Normalize y-values globally to [0, 1]
y_highres_normalized = (y_highres - global_min) / (global_max - global_min);

% Define colormap
cmap = parula(256); % Choose your desired colormap (e.g., 'jet', 'hot', etc.)

% Map normalized y-values to colormap
colorIndices = round(y_highres_normalized * 255) + 1;

% Prepare the figure
figure;
hold on;

for i = 1:size(y_highres, 1)
    % Plot each segment with color changing based on global y-value
    for j = 1:length(x_highres) - 1
        % Define color for the current segment based on global y-value
        segmentColor = cmap(colorIndices(i, j), :);

        % Create a patch for each segment
        patch([x_highres(j) x_highres(j+1) x_highres(j+1) x_highres(j)], ...
              [y_highres(i, j) y_highres(i, j+1) y_highres(i, j+1)-0.001 y_highres(i, j)-0.001], ...
              segmentColor, 'EdgeColor', 'none'); % No edge color for smoother look
    end
end

% Add a colorbar for global y-axis mapping
colormap(cmap);
cbar = colorbar;
cbar.Label.String = 'Prediction Accuracy';
caxis([global_min global_max]); % Colorbar range based on the global y-values

xlabel('Number of bins in the reference model')
ylabel('Prediction Accuracy')
box off
title('Trainingsdata adjusted Single Onset Vector')
set(gca,'FontSize',16)

% save_fig(gcf,fig_path,'DNS_ISI_singlevec_adjustedtraining_colored')


%% plot the prediction accuracy per bin and condition
figure
plot(temp_dat')

fig_pos = [1          41        1920         963];

for k = 1:length(task)
    for nb = 1:length(numBins)
        
        edge = binEdges{nb};

        %% get the corresponding weights
        %get the y-values
        ylabels = round(edge(2:end)./100,2);
        figure
        set(gcf,'position',fig_pos)
        %plot the weights
        t = tiledlayout(1,2)
        
        
        % compare prediction accuracies
        ons_pred = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_ISI_SingleOnsets_cleaned.mat');
        ons_pred = squeeze(mean(ons_pred.reg(:,k,:),[2 3]));
        
        ABons_struct = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_fulltr.mat')
        ABenv_pred = squeeze(mean(ABons_struct.result_reg(:,k,nb,:),4));
        
        [AB_env_sort,s_idx] = sort(ABenv_pred,'ascend');
        ons_pred_sort = ons_pred(s_idx);
        
        
        nexttile
        hold on
        
        plot(ons_pred_sort, '-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons');
        plot(AB_env_sort, '-o', 'Color', [0.49, 0.18, 0.56], 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'ISI Ons');
        % Customize the plot
        xlabel('Subjects');
        ylabel('Pearson''s r');
        legend('Location', 'NorthWest');
        grid on;
        set(gca,'FontSize',14)
        
        

        
        %plot the singular reduced values
        reduc_reg = squeeze(mean(result_reg(:,k,nb,:,:),5));
        reduc_reg = reduc_reg(s_idx,:);
        mean_reduc = mean(reduc_reg,2);
        sem_rand = std(reduc_reg,[],2)/sqrt(size(reduc_reg,2));
        plot(mean_reduc,'-o','Color','k','linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'IOns adj');
%         errorbar(1:length(sbj),mean_reduc,sem_rand,'k','LineStyle','none','LineWidth',3, 'DisplayName', 'Size Adjusted');
        
        cmp_ons = [ons_pred_sort mean_reduc];
        for tt = 1:2
            [p h] = signrank(cmp_ons(:,tt),AB_env_sort);
            if tt == 1;ad =0; else ad = 0.01; end
            if p <0.05 && p > 0.01
                text(length(sbj), max(cmp_ons(:,tt))-ad, '*', 'FontSize', 18, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            elseif p < 0.01 && p > 0.001
                text(length(sbj), max(cmp_ons(:,tt))-ad, '**', 'FontSize', 18, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            elseif p< 0.001
                text(length(sbj), max(cmp_ons(:,tt))-ad, '***', 'FontSize', 18, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            end
        end
        
        title(t,sprintf('DNS #bins:%d cond:%s',numBins(nb),task{k}),'FontSize',28)
        
        %compare directly the binned versus the reduced 
        nexttile
        violinplot([AB_env_sort mean_reduc],{'ISI Bins','Adj onset'})
        
        sigstar({[1 2]},signrank(AB_env_sort,mean_reduc))
        
%         save_fig(gcf,fig_path,sprintf('DNS_dist_adjtr_%s_%d',task{k},numBins(nb)))

    end
end