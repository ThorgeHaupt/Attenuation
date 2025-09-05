%% script to plot the different models next to each other ... 
DNS_setup
OT_setup
%load the condition data
cond = load('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_uniform_fulltr.mat')
cond_isi = squeeze(mean(cond.result_reg,[2 4]));
ons_pred = load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_ISI_SingleOnsets_cleaned.mat');
cond_ons = squeeze(mean(ons_pred.reg,[2 3]));

%load the subject data
subject = load('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_long_binisi.mat')
sub_isi = squeeze(mean(subject.result_reg,3));
subject = load('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_dist_long_single.mat')
sub_ons = squeeze(mean(subject.result_reg,2));

%load the generic data
generic = load('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\DNS_ISI_generic.mat')
temp_dat = mean(generic.result_reg,3);
gen_isi = temp_dat(1:7,:)';
gen_ons = temp_dat(end,:)';

fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\Uniform\';





%% do the sig testing

isi = {cond_isi,sub_isi,gen_isi};
ons = [cond_ons sub_ons gen_ons]

for m =1:size(ons,2)
    temp_ons = ons(:,m);
    temp_isi = isi{m};
    
    for i = 1:length(numBins)
        
        %do the testing
        p_val(m,i) = signrank(temp_ons,temp_isi(:,i))
    end
end
[h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p_val(:),0.05,'dep');
p_int = reshape(adj_p,size(p_val));

%% 
tit = {'Condition','Subject','Generic'};

%plot the results
figure,hold on;

t = tiledlayout(1,size(ons,2),'TileSpacing','Tight')
for m =1:size(ons,2)
    temp_ons = ons(:,m);
    temp_isi = isi{m};
    
    %sort the data according to ons
    [temp_ons_sort, sidx] = sort(temp_ons,'ascend');
    temp_isi_sort = temp_isi(sidx,:);
    
    nexttile 
    for nb = 1:length(numBins)
        plot(temp_isi_sort(:,nb),'-o', 'Color', gradientColors_ISI(nb,:),'linew',2, 'MarkerFaceColor', 'auto');
        hold on
    end
    labels = num2cell(numBins');
    %contrast the prediction accuracy
    p_temp = p_int(m,:);
    for i = 1:length(p_temp)
        if p_temp(i) < 0.05 && p_temp(i) > 0.01
            labels{i} = strcat(num2str(labels{i}), ' *');
        elseif p_temp(i) < 0.01 && p_temp(i) > 0.001
            labels{i} = strcat(num2str(labels{i}), ' **');
        elseif  p_temp(i) < 0.001
            labels{i} = strcat(num2str(labels{i}), ' ***');
        else
            labels{i} = strcat(num2str(labels{i}), ' N.S.');
        end
    end
    legend(labels,'Box','off','Location','northwest')
    plot(temp_ons_sort,'-o', 'Color', [0, 0.45, 0.74],'linew',2, 'MarkerFaceColor', 'auto', 'DisplayName', 'Ons');
    
    box off
    title(tit{m})
    set(gca,'FontSize',14,'Ylim',[0.01 0.08])
end
xlabel(t,'Participants','FontSize',16)
ylabel(t,'Prediction Accuracy','FontSize',16)

save_fig(gcf,fig_path,'DNS_ISI_all_pred')