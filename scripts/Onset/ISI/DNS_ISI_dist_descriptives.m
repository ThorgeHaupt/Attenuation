%Descriptives of the onset distances


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

%Wang et al.
bin_edge{1,1} = [0.1 0.2000    0.3000    0.4000    0.5000    0.6000    0.7000    0.8000    0.9000    1.0000];
%Caballero et al. 
bin_edge{2,1} = [0.25 0.5000    1.0000    2.0000    4.0000    8.0000];
%Zacharias et. al.
bin_edge{3,1} =  [0.2500    0.5000    0.7500    1.0000    1.5000    2.0000    3.0000    4.0000    5.0000    7.0000];

bin_edge_label{1,1} = {'0.1 - 0.2',... 
    '0.2 - 0.3',...
    '0.3-0.4',...
    '0.4-0.5',...
    '0.5-0.6',...
    '0.6-0.7',...
    '0.7-0.8',...
    '0.8-0.9',...
    '0.9-1'};

bin_edge_label{2,1} = {'0.25 - 0.5',... 
    '0.5 - 1',...
    '1-2',...
    '2-4',...
    '4-8'};

bin_edge_label{3,1} = {'0.25 - 0.5',... 
    '0.5 - 0.75',...
    '0.75 - 1',...
    '1-2',...
    '2-3',...
    '3-4',...
    '4-5',...
    '5-5'};

    
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
     
        %save the novel ultimate
        save(sprintf('ons_ult_%s.mat',task{k}),'novelty_ultm')
        
        
        peak = smooth_peak(novelty_ultm,fs_new,'plt',1);
        
       
       
        %get the indicies
        ons_idx = find(peak);
        
        %find the distance
        ons_dif = diff(ons_idx);
        
        %remove first ons
        ons_idx(1) = [];
        
        for ed = 1:size(bin_edge,1)
            
            %get the bin edge and adjust to sample rate
            bin_ed = bin_edge{ed,1}*EEG.srate;
            
            %get the extreme bin Edges
            max_dif = max(ons_dif);
            min_dif = min(ons_dif);
            
            bin_idx = ones(size(bin_ed));
            
            %make sure that the smallest diff, does not fall out of the
            %range
            if bin_ed(1,1)> min_dif
                
                %add another grabage bin
                bin_ed = [min_dif-1 bin_ed];
                bin_idx = [0 bin_idx];
            end
            
            if  bin_ed(end) < max_dif
                %add garbage bin, where all the values above fall
                bin_ed(end+1) = max_dif+1;
                bin_idx(end) =  0;
            end
            
            %bin the distances
            [counts, binEdges_dB, binIndices] = histcounts(ons_dif,bin_ed);
            
            %save the counts
            sav_count{s,k,ed} = counts;
            
            %save the pg version
            sav_count_pg{s,k,ed} = counts(logical(bin_idx));
            
            
            
            ons_bin = zeros(length(counts),length(peak));
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
                    mlpt_weight{s,k,ed} = model_train.w;
                    
                    %save the pg version 
                    mlpt_weight_pg{s,k,ed} = model_train.w(logical(bin_idx),:,:);
                    
                end
                
                
                %predict the neural data
                [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
                
                reg(ao,:) = STATS.r;
            end
            
            
            result_reg(s,k,ed,:,:) = reg;
            %         result_raw(s,k,:,:) = raw;
        end
    end
 end
 
 
 %% save the files
 trf_time = model_train.t;
        
cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results')
analysis08_results = struct;
analysis08_results.result_reg = result_reg;
analysis08_results.single_weight = single_weight;
analysis08_results.mlpt_weight = mlpt_weight;
analysis08_results.mlpt_weight_pg = mlpt_weight_pg;
analysis08_results.sav_count = sav_count;
analysis08_results.sav_count_pg = sav_count_pg;
analysis08_results.trf_time = trf_time;
analysis08_results.descp = 'tested different binning methods based on three papers';
save('DNS_BIN_dist.mat','-struct','analysis08_results')  

%% plot the descriptives
fig_path = '\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\BIN_dist\'

%origins of bin sizes
bin_typ = {'Wang et. al ','Caballero et al. ','Zacharias et al.'}

%the colors 
ons_col = [0.31, 0.80, 0.77];
colors = [ 1.00, 0.42, 0.42;  % Coral
    1.00, 0.82, 0.40;  % Goldenrod
    0.23, 0.53, 1.00]  % Slate Blue

%figure position
fig_pos = [680   286   984   692];

%get the pg_counts
for i = 1:size(bin_edge,1)

    %get the data
    temp_dat = cell2mat(reshape(cellfun(@(x) reshape(x,1,1,length(bin_edge{i,1})-1), sav_count_pg(:,:,i), 'UniformOutput', false), 20, 2));
    averageArray = squeeze(mean(temp_dat, [1 2]));
    sumArray = squeeze(sum(temp_dat,[1 2]));
    
    figure
    histogram('BinEdges',bin_edge{i,1},'BinCounts',sumArray,'FaceColor',colors(i,:))
    set(gca,'view',[90 -90])
    title(bin_typ{i})
    xlabel('Counts')
    ylabel('Bins in seconds')
    set(gca,'FontSize',15)
    set(gcf,'position',fig_pos)
    box off
    %save_fig(gcf,fig_path,sprintf('descp_sum_%s',bin_typ{i}))
end

count = [];
%get the non pg_counts
for i = 2:size(bin_edge,1)
    if i >1
        temp_dat = cell2mat(reshape(cellfun(@(x) reshape(x,1,1,length(bin_edge{i,1})+1), sav_count(:,:,i), 'UniformOutput', false), 20, 2));
        sum_count = sum(count,2);
        
        figure
        histogram('BinEdges',[0 bin_edge{i,1} 20],'BinCounts',sum_count,'FaceColor',colors(i,:))
        set(gca,'view',[90 -90])
        title(bin_typ{i})
        xlabel('Counts')
        ylabel('Bins in seconds')
        set(gca,'FontSize',15)
        set(gcf,'position',fig_pos)
        box off
    else
        
        
        for s = 1:length(sbj)
            for k = 1:length(task)
                
                temp_dat = sav_count{s,k,i};
                
                count(1:length(temp_dat),k) = cat(2,count,temp_dat);
                
                
            end
        end
    end
end

%% plot the prediction accuracy

fig_pos = [379   162   861   816];
for i = 1:size(bin_edge,1)
    
    temp_dat = squeeze(mean(result_reg(:,:,i,:,:),[1 2]));
    
    Ons = temp_dat(1,:);
    [Ons,s_idx] =sort(Ons,'ascend');
    AB_Ons = temp_dat(2,:);
    AB_Ons = AB_Ons(s_idx);
    
    [p,h] = signrank(Ons, AB_Ons);
    
    
    figure;
    t = tiledlayout('flow')
    nexttile
    hold on;
    
    plot(Ons, '-o', 'Color', ons_col, 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'Ons');
    plot(AB_Ons, '-o', 'Color', colors(i,:), 'linew',2,'MarkerFaceColor', 'auto', 'DisplayName', 'AB Ons');
    % Customize the plot
    xlabel('Subjects');
    legend('Location', 'NorthWest');
    title(bin_typ{:,i},'FontSize',16);
    set(gca,'FontSize',14)
    grid on;
    
    nexttile
    violinplot(temp_dat',{'Ons','AB Ons'},...
        'ViolinColor',[ons_col;colors(i,:)])
    set(gca,'FontSize',14)
    
    sigstar([1 2],p)
    
    ylabel(t,'Pearson''s r','FontSize',14);
    
    set(gcf,'position',fig_pos);
    
%     save_fig(gcf,fig_path,sprintf('predictacc_%s',bin_typ{1,i}))
end
    
    

%% plot the weights

%origins of bin sizes
bin_typ = {'Wang et. al ','Caballero et al. ','Zacharias et al.'};

%get the channels of the original paper
orig_chan = {'Cz','Fz','Cz'};

mrk = {'o','x','*','.','x','_','<','>','s','d'};

%base window
base_win = [-100 -10];

base_idx = dsearchn(trf_time',base_win');

%get the counts
for i = 1:size(bin_edge,1)
      
      
    
    combinedArray = [];
    % Loop over the two conditions (columns)
    for cond = 1:2
       %get data 
       temp_dat = cat(4, mlpt_weight_pg{:, cond, i});
       
       %remove baseline
       for s = 1:length(sbj)
           for ch = 1:EEG.nbchan
               base_dat = mean(temp_dat(:,base_idx(1):base_idx(2),ch,s),2);
               for bi = 1:length(bin_edge{i,1})-1
                   %subtract from data
                   temp_dat(bi,:,ch,s) =  temp_dat(bi,:,ch,s) - base_dat(bi);
                   
               end
               
           end
       end
           
        
        % Concatenate the matrices for each participant along the 4th dimension
        combinedArray(:,:,:,:,cond) = temp_dat;
        
    end
    
    
    temp_dat = squeeze(mean(temp_dat,5));
    
    %select the approriate channel 
    chan_idx = strcmp({EEG.chanlocs.labels},orig_chan{1,i});
%     
%     if i < 2
%         %plot the single weight
%         for k = 1:2
%             ons_w = squeeze(mean(mean(single_weight(:,k,:,:),1),2));
%             figure
%             plot(trf_time,ons_w','Color',ons_col,'linew',2)
%             box off
%             xlabel('Time Lag (ms)')
%             ylabel('a.u.')
%             title('Onset Model')
%             set(gca,'FontSize',14)
%         end
% 
%     end
%     
    %mean over everything
    w_avg = squeeze(mean(temp_dat(:,:,chan_idx,:),4));
    bin_ed = bin_edge{i,:};
    
    
%     figure
%     for t = 1:size(w_avg,1)
%         plot(trf_time,w_avg(t,:),'Color',colors(i,:),'Marker',mrk{t})
%         hold on
%     end
%     legend(num2str(reshape(bin_edge{i,:},[],1)))
%     title(bin_typ{1,i});
%     xlabel('Time Lag (ms)')
%     ylabel('a.u.')
%     set(gca,'FontSize',14)
%     save_fig(gcf,fig_path,sprintf('DNS_distance_weightcomb_%s',bin_typ{i}))

    
    %plot the weights
    %get the min max distance of the weights for separation purposes
%     w_avg = zscore(w_avg,[],'all')
    dist = (max(w_avg,[],'all') - min(w_avg,[],'all'))/2;
    ylabels = bin_edge{i,1}
    fig_pos = [466    80   774   700]
    % Plot each row as a separate line
    figure;
    set(gcf,'Position',fig_pos)
    tiledlayout(2,2)
    nexttile(1)
    hold on;
    y_val = [];
    for ed = 1:size(w_avg,1)
        plot(trf_time, w_avg(ed, :) + (ed-1) * dist, 'Color', colors(i,:), 'LineWidth', 1.5); % Offset each line for clarity
        y_val(ed) = mean(w_avg(ed,1:10)+(ed-1) * dist);
    end
    hold off;
    xlim([-50 250])
%     ylim([0 2e-3])
    % Add labels and title
    xlabel('Time Lag (ms)');
    ylabel('a.u.');
    title(bin_typ{1,i});
    set(gca,'FontSize',15,'YTick',y_val,'YtickLabel',ylabels)
    
    nexttile(2)
    imagesc('XData',trf_time,'CData',w_avg)
    yticks(linspace(1,length(ylabels),length(ylabels)))
    yticklabels(ylabels)
    set(gca, 'YDir', 'normal','FontSize',16)
    xlabel('Time Lag (ms)');

    axis tight
    title('Model Weights')
    
    

    
    %plot the N1 amplitude over for the different
    N1 = [10 150];
    P2 = [90 250];
    
    n1_idx = dsearchn(trf_time',N1');
    p2_idx = dsearchn(trf_time',P2');
    
    [n1_peak,n1_lat] = min(temp_dat(:,n1_idx(1):n1_idx(2),chan_idx,:),[],2);
    [p2_peak,p2_lat] = max(temp_dat(:,p2_idx(1):p2_idx(2),chan_idx,:),[],2);
    
%     n1_lat = round(mean(n1_lat));
%     p2_lat = round(mean(p2_lat));
%     
%     n1_peak = mean(n1_peak);
%     p2_peak = mean(p2_peak);
    
    n1_time = trf_time(n1_idx(1):n1_idx(2));
    p2_time = trf_time(p2_idx(1):p2_idx(2));
    
    %% plot the N1 amplitude values 
    nexttile(3)
    %get the data
    n1_dat = squeeze(n1_peak);
    n1_mean = mean(n1_dat,2);
    n1_std = std(n1_dat,1,2);
    sem = n1_std/size(n1_dat,2);
    
    %plot it 
    e = errorbar(linspace(1,size(n1_peak,1),size(n1_peak,1))',n1_mean,sem);
    e.Color = colors(i,:);
    e.LineWidth = 2;
    % Labeling the axes
    xlabel('Bins in seconds');
    xticks(linspace(1,size(n1_peak,1),size(n1_peak,1)))
    xticklabels(bin_edge_label{i,1})
    ylabel('a.u.');
    grid on;
    xlim([0 size(n1_peak,1)+1]);
    if i >1
        
        ylim([min(n1_mean)-1.5 max(n1_mean)+1.5]);
    end
    % Adding a title or label similar to the "C"
    title('N1 Peak amplitude') 
    box off
    set(gca,'FontSize',14)
    
    %% plot the P2 amplitude values
    nexttile(4)
    %get the data
    p2_dat = squeeze(p2_peak);
    p2_mean = mean(p2_dat,2);
    p2_std = std(p2_dat,1,2);
    sem = n1_std/size(n1_dat,2);
    
    %plot it 
    e = errorbar(linspace(1,size(p2_peak,1),size(p2_peak,1))',p2_mean,sem);
    e.Color = colors(i,:);
    e.LineWidth = 2;
    % Labeling the axes
    xlabel('Bins in seconds');
    xticks(linspace(1,size(n1_peak,1),size(n1_peak,1)))
    xticklabels(bin_edge_label{i,1})
    ylabel('a.u.');
    grid on;
    xlim([0 size(p2_peak,1)+1]);
    if i>1
        
        ylim([min(p2_mean)-1.5 max(p2_mean)+1.5]);
    end
    % Adding a title or label similar to the "C"
    title('P2 Peak amplitude') 
    box off
    set(gca,'FontSize',14)
    
    save_fig(gcf,fig_path,sprintf('DNS_distance_%s',bin_typ{i}))
    
end




%% for the conditions separate
   
    



 
