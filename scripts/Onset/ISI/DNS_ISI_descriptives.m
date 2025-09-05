%% this script is computing the descriptive of the distance of all participants and conditions

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

dns_dist = [];
 for s=1:length(sbj)

    for k=1:2
        
        %% compute the envelopes
        [EEG,PATH] = OT_preprocessing(s,k,sbj,20);

        cd(PATH)
        
        %get the neural data
        resp = double(EEG.data');
        
        novelty_ultm = load(sprintf('ons_ult_%s',task{k}));
        
        fs_new = EEG.srate;
        
        peak = smooth_peak(novelty_ultm.novelty_ultm,fs_new,'sigma',4);
        
        if size(resp,1)>length(peak)
            resp = resp(1:length(peak),:);
        elseif size(resp,1)<length(peak)
            peak = peak(:,1:size(resp,1));
        end
       
        %get the indicies
        ons_idx = find(peak);
        
        %find the distance
        ons_dif = diff(ons_idx);
        
        %set the bin Edges
        max_dif = max(ons_dif);
        min_dif = min(ons_dif);

        %save the results
        dns_dist = [dns_dist ons_dif];
    end
 end
 
%save the descriptives
save('dns_dist_descriptives.mat','dns_dist')

%plot the results
figure
histogram(dns_dist(dns_dist<1000),20)
view([90 -90])
xticklabels(linspace(0,10,11))
box off
ylabel('Counts')
xlabel('Distance in s.')
set(gca,'FontSize',16)

save_fig(gcf,'\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\figures\Onset\ISI\Descriptive\','DNS_isi_dist_descriptives')


