%% PATH 
addpath(genpath('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\functions\'))



%% color Gradient for the ISI
%set the colors right
endColor = [0.91, 0.75, 0.95]; % Convert 8-bit RGB to MATLAB RGB
baseColor = [0.49, 0.18, 0.56]; % End color for gradient

% Number of desired colors
numColors = 7;

% Generate linearly spaced colors along the gradient
gradientColors_ISI = zeros(numColors, 3);
for i = 1:numColors
    alpha = (i-1) / (numColors-1); % Blending factor (0 to 1)
    gradientColors_ISI(i, :) = (1 - alpha) * baseColor + alpha * endColor;
end

%% Color Gradient for the single vec
endColor = [0, 0.8, 0.9]; % Convert 8-bit RGB to MATLAB RGB
baseColor = [0, 0.5, 0.7410]; % End color for gradient

% Number of desired colors
numColors = 7;

% Generate linearly spaced colors along the gradient
gradientColors_svec = zeros(numColors, 3);
for i = 1:numColors
    alpha = (i-1) / (numColors-1); % Blending factor (0 to 1)
    gradientColors_svec(i, :) = (1 - alpha) * baseColor + alpha * endColor;
end

%% Bin Edges

%load the distance collection vector
load('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\DNS_exploration\Results\Onsets\ISI\dns_dist_descriptives.mat')
%number of bins
min_value = 1;       % Minimum value of the range
max_value = 1000;    % Maximum value of the range

%clean it first
dns_dist(dns_dist > max_value) = [];
% uniform distribution
numBins = linspace(2,8,7); % Number of bins

% uniform distribution
for i = 1:length(numBins)
    binEdges{i} = quantile(dns_dist, linspace(0, 1, numBins(i) + 1));
end

%% model Parameters
%TRF parameters
Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);

%partition the data set
nfold = 6;
testfold = 1;
