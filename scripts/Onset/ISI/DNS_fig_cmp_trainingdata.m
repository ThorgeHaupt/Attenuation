%% plot the line plots next to each other, to see the effect of trainingsdata
DNS_setup
OT_setup

%load the data
ons_pred = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_ISI_SingleOnsets_cleaned.mat');
ons_pred_res = squeeze(mean(ons_pred.reg,[2 3]));

%sort the data
[ons_pred_sort s_idx] = sort(ons_pred_res,'ascend');

adj_ons = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_adjtr.mat');
adj_ons_res = squeeze(mean(adj_ons.result_reg,[2 4 5]));
adj_ons_sort = adj_ons_res(s_idx,:);

isi_bin = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_fulltr.mat');
isi_bin_res = squeeze(mean(isi_bin.result_reg,[2 4]));
isi_bin_sort = isi_bin_res(s_idx,:);

fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\Uniform\';

%% do statistical testing
isi = {isi_bin_sort,adj_ons_sort};
ons = ons_pred_sort;


%do the sig testing
for m =1:3
    
    if m < 3
        temp_ons = ons;
        temp_isi = isi{1,m};
        for i = 1:length(numBins)
            
            %do the testing
            [p,h,stats] = signrank(temp_ons,temp_isi(:,i))
             p_val(m,i) = p;
             w_val(m,i) = stats.signedrank;
             z_val(m,i) = stats.zval;
             e_val(m,i) = stats.zval/sqrt(size(temp_ons,1)*2)
             
        end
    else
        
        for i = 1:length(numBins)
            
            %do the testing
            [p,h,stats] = signrank((isi_bin_sort(:,i)- adj_ons_sort(:,i)));
             p_val(m,i) = p;
             w_val(m,i) = stats.signedrank;
             z_val(m,i) = stats.zval;
             e_val(m,i) = stats.zval/sqrt(size(temp_ons,1))
        end
    end
    
   
end
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_val(:),0.05,'dep');
p_int = reshape(adj_p,size(p_val));
isi_stat = [w_val(1,:)' z_val(1,:)' p_int(1,:)' e_val(1,:)']
ons_stat = [w_val(2,:)' z_val(2,:)' p_int(2,:)' e_val(2,:)']
dif_stat = [w_val(3,:)' z_val(3,:)' p_int(3,:)' e_val(3,:)']

%% 
ylim = [-0.005 0.075];
%plot the results
figure
tiledlayout(1,4,'TileSpacing','Compact')

nexttile
for i = 1:length(numBins)
    plot(1:length(sbj),adj_ons_sort(:,i),'o-','Color',gradientColors_svec(i,:),'linew',2,'MarkerFaceColor', 'auto')
    hold on
end
labels = [];
labels = num2cell(numBins');
%contrast the prediction accuracy
for i = 1:size(p_int,2)
    if p_int(2,i) < 0.05 && p_int(2,i) > 0.01
        labels{i} = strcat(num2str(labels{i}), ' *');
    elseif p_int(2,i) < 0.01 && p_int(2,i) > 0.001
        labels{i} = strcat(num2str(labels{i}), ' **');
    elseif  p_int(2,i) < 0.001
        labels{i} = strcat(num2str(labels{i}), ' ***');
    else 
        labels{i} = strcat(num2str(labels{i}), ' N.S.');
    end
end
legend(labels,'Box','off','Location','northwest')
hold on
plot(ons_pred_sort,'-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons Orig.');
xlabel('Participants')
ylabel('Prediction Accuracy')
box off
grid on 
title('adjusted Single Onset Vector')
set(gca,'FontSize',16,'Ylim',ylim)


nexttile
for i = 1:length(numBins)
    plot(1:length(sbj),isi_bin_sort(:,i),'o-','Color',gradientColors_ISI(i,:),'linew',2,'MarkerFaceColor', 'auto')
    hold on
end
labels = [];
labels = num2cell(numBins');
%contrast the prediction accuracy
for i = 1:size(p_int,2)
    if p_int(1,i) < 0.05 && p_int(1,i) > 0.01
        labels{i} = strcat(num2str(labels{i}), ' *');
    elseif p_int(1,i) < 0.01 && p_int(1,i) > 0.001
        labels{i} = strcat(num2str(labels{i}), ' **');
    elseif  p_int(1,i) < 0.001
        labels{i} = strcat(num2str(labels{i}), ' ***');
    else 
        labels{i} = strcat(num2str(labels{i}), ' N.S.');
    end
end
legend(labels,'Box','off','Location','northwest')
hold on
plot(ons_pred_sort,'-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons Orig.');
xlabel('Participants')
title('Bin Isi Model')
box off
set(gca,'FontSize',16,'Ylim',ylim,'Yticklabels',[])
grid on 


nexttile
for i = 1:length(numBins)
    temp_dat = isi_bin_sort(:,i)- adj_ons_sort(:,i);
    plot(1:length(sbj),temp_dat,'o-','Color',gradientColors_ISI(i,:),'linew',2,'MarkerFaceColor', 'auto')
    hold on
    
end
labels = [];
labels = num2cell(numBins');
%contrast the prediction accuracy
for i = 1:size(p_int,2)
    if p_int(3,i) < 0.05 && p_int(3,i) > 0.01
        labels{i} = strcat(num2str(labels{i}), ' *');
    elseif p_int(3,i) < 0.01 && p_int(3,i) > 0.001
        labels{i} = strcat(num2str(labels{i}), ' **');
    elseif  p_int(3,i) < 0.001
        labels{i} = strcat(num2str(labels{i}), ' ***');
    else 
        labels{i} = strcat(num2str(labels{i}), ' N.S.');
    end
end
legend(labels,'Box','off','Location','northwest')
hold on
yline(0,'linew',2)
fill([0 20 20 0], [0 0 -0.08 -0.08], [0.5, 0.5, 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
xlabel('Participants')
ylabel('\Delta Bin ISI - adj Ons.')
box off
title('Difference')
set(gca,'FontSize',16,'Ylim',[-0.005 0.02])
ylim_ref = get(gca,'Ylim');

nexttile
for i = 1:length(numBins)
    temp_dat = isi_bin_sort(:,i)- adj_ons_sort(:,i);
    [f, xi] = ksdensity(temp_dat);
    fill(xi, f, gradientColors_ISI(i,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold on
    plot(xi, f, 'Color',gradientColors_ISI(i,:),'LineWidth', 2);
    
end
ylabel('\Delta Bin ISI - adj Ons.')
set(gca,'Xlim',ylim_ref,'view',[-90 90],'XAxisLocation','top','YTick',[],'FontSize',16)
box off
xline(0,'linew',2)
fill([0 0 -0.08 -0.08],[0 150 150 0], [0.5, 0.5, 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');


% save_fig(gcf,fig_path,'DNS_fig_cmp_trainingdata')