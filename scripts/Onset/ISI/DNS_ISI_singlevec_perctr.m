OT_setup
DNS_setup

% plot the results
reduc_vec = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_singlevec_perctr.mat');
reduc_vec_dat = squeeze(mean(reduc_vec.result_reg,[2 4 5]));
perc_tr = reduc_vec.perc_tr;

bin_dat = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_fulltr.mat');
bin_res = flip(squeeze(mean(bin_dat.result_reg,[2 4])),2);

fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\Uniform\adjTraining\'
%% plot the actual accurcies
% figure out how much data on average is being used for training depending
% on the 
%get the tr percentages
perc_tr_bin = round(flip(1./linspace(2,8,7)),3);
figure; hold on
% Boxplot with improved visuals
b1 = boxplot(bin_res, perc_tr_bin, 'Positions', perc_tr_bin, ...
    'Color', flip(gradientColors_ISI,1), 'Widths',0.01);


% Customize boxplot appearance
set(b1, 'LineWidth', 1.5); % Thicker lines for better visibility
set(gca, 'XTick', perc_tr_bin, 'XTickLabel', num2str(perc_tr_bin'.*100)); % Customize tick labels

% test for significance
for i=1:length(numBins)
    
    %i need to interpolate the data -> fuck my life
    tar_perc = perc_tr_bin(i);
    
    %get the correct Perc_indices
    [~,s_idx] = sort(abs(perc_tr-tar_perc),'ascend');
    
    %get the weight
    w1 = (perc_tr(s_idx(1))- tar_perc) / (perc_tr(s_idx(1))-perc_tr(s_idx(2)));
    w2 = 1-w1;
    
    temp_dat(:,i) = [(reduc_vec_dat(:,s_idx(1)).*w1)+ (reduc_vec_dat(:,s_idx(2)).*w2)];

    [p,h,stats] = signrank(temp_dat(:,i),bin_res(:,i));
    p_isi(i) = p;
    zval(i) = stats.zval;
    wval(i) = stats.signedrank;
end
%do multiple testing correction
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_isi,0.05,'dep','yes');
ef_size = zval./sqrt(size(bin_res,1))

%stats_summary
stats_sum = [wval' zval' p_isi' ef_size']; 

%get also the t values and w and all the other important test statistics


for i = 1:length(numBins)
    if adj_p(i) <0.05 && adj_p(i) >0.01
        text(perc_tr_bin(i), max(bin_res(:,i))+0.005, '*', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    elseif adj_p(i) < 0.01 &&adj_p(i)> 0.000
        text(perc_tr_bin(i), max(bin_res(:,i))+0.005, '**', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    elseif adj_p(i)== 0
        text(perc_tr_bin(i), max(bin_res(:,i))+0.005, '***', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    else
        text(perc_tr_bin(i), max(bin_res(:,i))+0.005, 'N.S.', 'FontSize', 18, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        
    end
end

ylim([0.02 0.06])
xlim([0.01 0.52])

% Add significance markers

% Overlay the reduced training data with mean and SEM
dim = 1;
meanData = mean(reduc_vec_dat, dim);
SEM = std(reduc_vec_dat, 0, dim) ./ sqrt(size(reduc_vec_dat, dim));

% Plot mean and SEM as a shaded region
fill([perc_tr, fliplr(perc_tr)], ...
    [meanData - SEM, fliplr(meanData + SEM)], ...
    [0, 0.5, 0.7410], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Shaded SEM

% Plot mean line
m1 = plot(perc_tr, meanData, 'Color', [0, 0.5, 0.7410], 'LineWidth', 2);

% Adjust plot aesthetics
ylim([0 0.08]);
xlabel('Training Proportion (%)');
ylabel('Prediction Accuracy');
title('Prediction Accuracy vs Training Proportion');
grid on; % Add gridlines
box off; % Add box around the plot
set(gca,'FontSize',16)

%%plot a grey shaded area
fill([0.125 0.50 0.50 0.125], [0 0 0.08 0.08], [0.5, 0.5, 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

save_fig(gcf,fig_path,'DNS_ISI_singlevec_perctr_cmp_tronly')

%% create a time line 
figure
t = tiledlayout(2,length(numBins),'TileSpacing','Compact')
ylabel(t,'Prediction Accuracy','FontSize',16);
for i = 1:length(numBins)
    nexttile
    violinplot([temp_dat(:,i) bin_res(:,i)],{'Ons','Bin Isi'},...
        'ViolinColor',[gradientColors_svec(8-i,:);gradientColors_ISI(8-i,:)])
    sigstar({[1 2]},stats_sum(i,3))
    set(gca,'Ylim',[0.01 0.08],'FontSize',12)
    title(sprintf('%0.1f%%',round(perc_tr_bin(i)*100,1)))
    box off
    
end

%increase the resolution 
% data2 = mean(reduc_vec_dat,1);
% [x, y] = meshgrid(1:size(data2, 2), 1:size(data2, 1)); % Original grid
% [xq, yq] = meshgrid(linspace(1, size(data2, 2), 200), linspace(1, size(data2, 1), 200)); % High-res grid
% data2_interp = interp2(x, y, data2, xq, yq, 'linear'); % Linear interpolation
nexttile([1 6])
% imagesc(data2_interp)
% colorbar
% Overlay the reduced training data with mean and SEM
dim = 1;
meanData = mean(reduc_vec_dat, dim);
SEM = std(reduc_vec_dat, 0, dim) ./ sqrt(size(reduc_vec_dat, dim));

% Plot mean and SEM as a shaded region
fill([perc_tr, fliplr(perc_tr)], ...
    [meanData - SEM*2, fliplr(meanData + SEM*2)], ...
    [0, 0.5, 0.7410], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Shaded SEM
hold on
% Plot mean line
m1 = plot(perc_tr, meanData, 'Color', [0, 0.5, 0.7410], 'LineWidth', 2);
hold on
% Adjust plot aesthetics
ylim([0 0.08]);
xlabel('Bin adjusted % of data used for Training');

% title('Prediction Accuracy vs Training Proportion');
% grid on; % Add gridlines
box off; % Add box around the plot
set(gca,'FontSize',12,'xlim',[0.04 0.51],'XTick', perc_tr_bin, 'XTickLabel', num2str(perc_tr_bin'.*100)); % Customize tick labels)
n = 0.0005
area = [perc_tr_bin'-n perc_tr_bin'+n perc_tr_bin'+n perc_tr_bin'-n]
%%plot a grey shaded area
for i = 1:length(numBins)
   f(i) = fill(area(i,:), [0 0 0.08 0.08], [0.5, 0.5, 0.5], 'FaceAlpha', 0.8, 'EdgeColor', 'none');
end
legend([m1 f(1)],{'Ons','POI'},'Box','off','FontSize',12,'Location','northwest')

    
nexttile
% Plot mean and SEM as a shaded region
fill([perc_tr, fliplr(perc_tr)], ...
    [meanData - SEM*2, fliplr(meanData + SEM*2)], ...
    [0, 0.5, 0.7410], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Shaded SEM
hold on
% Plot mean line
m1 = plot(perc_tr, meanData, 'Color', [0, 0.5, 0.7410], 'LineWidth', 2);
hold on
% Adjust plot aesthetics
ylim([0 0.08]);
% title('Prediction Accuracy vs Training Proportion');
% grid on; % Add gridlines
box off; % Add box around the plot
set(gca,'FontSize',12,'xlim',[0.98 1.01],'XTick', perc_tr_bin, 'XTickLabel', num2str(perc_tr_bin'.*100));
ax = gca; % Get current axes
ax.YAxis.Visible = 'off'; % Turn off the y-axis

% save_fig(gcf,fig_path,'DNS_reduc_tr_cmp')