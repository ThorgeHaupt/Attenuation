%testing of the different onset thresholds and how it impacts the
%computation for the different dns conditions
MAINPATH = 'O:\projects\thh_ont\auditory-attention-in-complex-work-related-auditory-envrionments\data files'
addpath(genpath(MAINPATH));

OT_setup

thresh = logspace(log(0.22),log(0.8),18);

Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

auditory = {'envelope','mel'}

fig_path = '\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Label\'

%% load the Yament
downloadFolder = fullfile(tempdir,'YAMNetDownload');
loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/yamnet.zip');
YAMNetLocation = tempdir;
unzip(loc,YAMNetLocation)
addpath(fullfile(YAMNetLocation,'yamnet'))

net = yamnet

%%


for s=1:length(sbj)
    try
        for k=1:2
            
            addpath(genpath('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\eeglab2021'))
            [EEG,PATH] = OT_preprocessing(s,k,sbj,20);
            
            cd(PATH)
            
            %get the audio files
            if k == 1
                [audioIn,fs] =audioread(sprintf('narrow_audio_game_%s.wav',sbj{s}));
            else
                [audioIn,fs] =audioread(sprintf('wide_audio_game_%s.wav',sbj{s}));
                
            end
            
            rmpath(genpath('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\eeglab2021'))
            
            %         [sounds,timeStamps] = classifySound(audioIn,fs);
            %
            melSpectYam = yamnetPreprocess(audioIn,fs,'OverlapPercentage',0);
            classes = classify(net,melSpectYam);
            
            %every classification is roughly 960ms long, divide the indices
            %according to speech and non-speech
            %         cls_time = [0:0.96:1096.45-0.96;0.96:0.96:1096.45]' ;
            
            for ao = 1:length(auditory)
                stim = extract_stimulus2(EEG, PATH,auditory{ao}, k, sbj{s},task);
                
                %get the neural data
                resp = double(EEG.data);
                
                if size(resp,2)>size(stim,1)
                    resp = resp(:,1:size(stim,1));
                end
                
                lins = length(resp)/EEG.srate;
                
                cls_time = [];
                cls_time(1,:) = linspace(0,lins-0.96,length(classes));
                cls_time(2,:) = linspace(0.96,lins,length(classes));
                
                speechidx = ismember(classes,'Speech');
                
                %find the start and ending values of clusters
                d = diff([0 speechidx' 0]);
                
                % Indices where a cluster of 1s starts and ends
                startIdx = find(d == 1);  % Starting indices of clusters
                endIdx = find(d == -1) - 1;  % Ending indices of clusters
                
                %save them
                cls_speech = [startIdx' endIdx'];
                
                %         cls_idx = find(cls_speech(:,1) == cls_speech(:,2));
                %         cls_speech(cls_idx,:) = [];
                
                %extract the timepoints in seconds
                speech_time = [cls_time(1,cls_speech(:,1))' cls_time(2,cls_speech(:,2))']*EEG.srate;
                
                %select the time periods that do not contain these indices
                mod_start = [];
                mod_start = startIdx';
                
                %4 conditions
                if cls_speech(end) == length(classes) && cls_speech(1,1) == 1
                    mod_start = mod_start-1;
                    mod_start(1) = [];
                    endIdx = endIdx'+1;
                    endIdx(end) = [];
                    
                elseif cls_speech(end)  ~= length(classes) && cls_speech(1,1) == 1
                    mod_start = mod_start-1;
                    mod_start(1) = [];
                    mod_start(end+1) = length(classes);
                    endIdx = endIdx'+1;
                    
                elseif cls_speech(end)  == length(classes) && cls_speech(1,1) ~= 1
                    mod_start = mod_start-1;
                    endIdx = endIdx'+1;
                    endIdx = [1 ;endIdx];
                    endIdx(end) = [];
                    
                else
                    mod_start = mod_start-1;
                    mod_start(end+1) = length(classes);
                    endIdx = endIdx'+1;
                    endIdx = [1 ;endIdx];
                end
                cls_non_speech = [endIdx mod_start];
                
                %extract the non-speech time segments
                non_speech_time = [cls_time(1,cls_non_speech(:,1))' cls_time(2,cls_non_speech(:,2))']*EEG.srate;
                
                %segment the data according to speech and non-speech data
                eeg_time = EEG.times(1:length(resp))/10;
                
                
                
                %for speech
                clear eeg_speech stim_speech
                for t = 1:length(speech_time)
                    
                    %find the closest value
                    idx = dsearchn(eeg_time',speech_time(t,:)');
                    
                    eeg_speech{t,:} = resp(:,idx(1):idx(2))';
                    stim_speech{t,:} = stim(idx(1):idx(2),:);
                    
                end
                
                %non_speech
                clear eeg_nspeech stim_nspeech
                for t = 1:length(non_speech_time)
                    
                    %find the closest value
                    idx = dsearchn(eeg_time',non_speech_time(t,:)');
                    
                    eeg_nspeech{t,:} = resp(:,idx(1):idx(2))';
                    stim_nspeech{t,:} = stim(idx(1):idx(2),:);
                    
                end
                
                
                %% spit the data
                %#of folds
                n_perm = min([length(stim_speech) length(stim_nspeech)]);
                nr_tr = round(length(stim_speech)*0.8);
                nr_ts = length(stim_speech)-nr_tr;
                
                %random selection
                ts_idx = randperm(n_perm,nr_ts);
                tr_idx = setdiff(1:n_perm, ts_idx);
                
                %split the data
                %speech
                eeg_sp_tr = eeg_speech(tr_idx,:);
                stim_sp_tr = stim_speech(tr_idx,:);
                eeg_sp_ts  = eeg_speech(ts_idx,:);
                stim_sp_ts = stim_speech(ts_idx,:);
                
                %non_speech
                eeg_nsp_tr = eeg_nspeech(tr_idx,:);
                stim_nsp_tr = stim_nspeech(tr_idx,:);
                eeg_nsp_ts  = eeg_nspeech(ts_idx,:);
                stim_nsp_ts = stim_nspeech(ts_idx,:);
                
                %train the models
                mtr_speech = mTRFtrain(stim_sp_tr,eeg_sp_tr,EEG.srate,Dir,tmin,tmax,0.05,'verbose',0);
                mtr_nspeech = mTRFtrain(stim_nsp_tr,eeg_nsp_tr,EEG.srate,Dir,tmin,tmax,0.05,'verbose',0);
                
                if ndims(squeeze(mtr_speech.w))<3
                    weight_sav(s,k,ao,1,:,:) = squeeze(mtr_speech.w);
                    weight_sav(s,k,ao,2,:,:) = squeeze(mtr_nspeech.w);
                end
                
                
                %predict speech on speech
                [PRED_sp_sp,STATS_sp_sp] = mTRFpredict(stim_sp_ts,eeg_sp_ts,mtr_speech ,'verbose',0);
                result_sp_sp(s,k,ao,:) = mean(STATS_sp_sp.r,'all');
                
                %predict speech on non-speech
                [PRED_sp_nsp,STATS_sp_nsp] = mTRFpredict(stim_nsp_ts,eeg_nsp_ts,mtr_speech ,'verbose',0);
                result_sp_nsp(s,k,ao,:) = mean(STATS_sp_nsp.r,'all');
                
                %predict non -speech on non-speech
                [PRED_nsp_nsp,STATS_nsp_nsp] = mTRFpredict(stim_nsp_ts,eeg_nsp_ts,mtr_nspeech ,'verbose',0);
                result_nsp_nsp(s,k,ao,:) = mean(STATS_nsp_nsp.r,'all');
                
                %predict non-speech on speech
                [PRED_nsp_sp,STATS_nsp_sp] = mTRFpredict(stim_sp_ts,eeg_sp_ts,mtr_nspeech ,'verbose',0);
                result_nsp_sp(s,k,ao,:) = mean(STATS_nsp_sp.r,'all');
                
            end
            
        end
        
    catch ME
        % Some error occurred if you get here.
        errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
            ME.stack(1).name, ME.stack(1).line, ME.message);
        fprintf(1, '%s\n', errorMessage);
        uiwait(warndlg(errorMessage));
    end
end


cd('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\prelim_results\')
DNS_epoch = struct()
DNS_epoch.weight_sav = weight_sav;
DNS_epoch.result_sp_sp = result_sp_sp;
DNS_epoch.result_nsp_sp = result_nsp_sp;
DNS_epoch.result_sp_nsp= result_sp_nsp;
DNS_epoch.result_nsp_nsp = result_nsp_nsp;

DNS_epoch.t = 'labeling of segments -> discrepance between speech vs. no speech';
save('DNS_spvnsp.mat','-struct','DNS_epoch')


%% plotting
%prepare the data
figure
sp_w = squeeze(weight_sav(:,:,:,1,:,:));
le = size(sp_w,3);
temp_dat = squeeze(mean(mean(sp_w,2,'omitnan'),4,'omitnan'));
dnsMean = squeeze(mean(temp_dat,1,'omitnan'));
h = plot(dnsMean,'b','linew',2);
hold on
N = size(sp_w,1);
ySEM = std(temp_dat,1)/sqrt(N);
CI95 = tinv([0.025 0.975],N-1);
yCI95 = bsxfun(@times,ySEM,CI95(:));
conv = yCI95 + dnsMean ;
x2 = [linspace(1,le,le) fliplr(linspace(1,le,le))];
inbe = [conv(1,:) fliplr(conv(2,:))];
f = fill(x2,inbe,'b');
f.FaceAlpha = 0.2;
f.EdgeAlpha = 0.4;
f.LineWidth = 0.5;
hold on

nps_w = squeeze(weight_sav(:,:,:,2,:,:));
temp_datl = squeeze(mean(mean(nps_w,2,'omitnan'),4,'omitnan'));
dnslMean = squeeze(mean(temp_datl,1,'omitnan'));
l = plot(dnslMean,'r','linew',3);
hold on
ylSEM = std(temp_datl,1)/sqrt(N);
CI95 = tinv([0.025 0.975],N-1);
yCI95 = bsxfun(@times,ylSEM,CI95(:));
conv = yCI95 + dnslMean;
x2 = [linspace(1,le,le) fliplr(linspace(1,le,le))];
inbe = [conv(1,:) fliplr(conv(2,:))];
f = fill(x2,inbe,'r');
f.FaceAlpha = 0.1;
f.EdgeAlpha = 0.2;
f.LineWidth = 0.5;

l = legend([h,l],'Speech','no Speech','Fontsize',24,'Location','southeast')
set(l,'Box','off')

set(gca,'XTick', linspace(1,le,7),'XTickLabel',linspace(tmin,tmax,7),'Fontsize',24)
title(sprintf('TRF %s',auditory{1,1}),'FontSize',30)

xlabel('Time (ms)','Fontsize',24)
ylabel('a.u.', 'Interpreter', 'tex','Fontsize',24);

set(gcf,'Position',fig_pos)
box off


%% print the prediction accuracies
%speech vs. no-speech 
temp_dat = [squeeze(mean(result_sp_sp,2)) squeeze(mean(result_nsp_nsp,2))]


figure
h = violinplot(temp_dat ,[auditory, auditory],... %data and labels
    'ViolinAlpha',0.45,...       %sets the color to be more transparent
    'ShowMean', true)            %hightlights the mean 
xline(5.5,'--k','linew',2)       %the line to separate single and multi feature models   
hold on

%% print the cross prediction 

sp_sp = squeeze(mean(result_sp_sp(:,:,1),2));
sp_nsp = squeeze(mean(result_sp_nsp(:,:,1),2));
nsp_sp = squeeze(mean(result_nsp_sp(:,:,1),2));
nsp_nsp = squeeze(mean(result_nsp_nsp(:,:,1),2));


%non-speech on speech
[cor,p] = corrcoef(sp_sp,nsp_sp)

ylim= [-0.01 0.1]
xlim = [-0.01 0.1]

figure
scatter(sp_sp,nsp_sp,'filled')
hold on
ls = lsline(gca)
ls.LineWidth = 2;
ls.Color = 'k';
hold on
plot(xlim,ylim,'--','linew',2)
set(gca,'Ylim',ylim,'Xlim',xlim)%,'FontSize',26)
axis square

%speech on non-speech
[cor,p] = corrcoef(nsp_nsp,sp_nsp)

ylim= [-0.01 0.1]
xlim = [-0.01 0.1]

figure
scatter(nsp_nsp,sp_nsp,'filled')
hold on
ls = lsline(gca)
ls.LineWidth = 2;
ls.Color = 'k';
hold on
plot(xlim,ylim,'--','linew',2)
set(gca,'Ylim',ylim,'Xlim',xlim)%,'FontSize',26)
axis square






